from __future__ import annotations

import io
import math
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import MultiPoint

from .contracts import AnalysisResult, Point
from .migan import MIGANInpainting, MODEL_SIZE


def compose_final_image(
    source: Image.Image,
    analysis: AnalysisResult,
    translations: dict[int, str],
    reconstructor: str,
    migan: MIGANInpainting,
) -> tuple[Image.Image, list[str]]:
    """Complete repair and text rendering before returning any image.

    MI-GAN is invoked once with the complete working image and one combined
    mask.  The generated image is blended only through that mask, so pixels
    outside detected text quads remain owned by the source image.
    """

    warnings: list[str] = []
    original = source.convert("RGB")
    if reconstructor == "migan-512" and analysis.regions:
        known_mask = build_text_mask(original, analysis)
        generated_512 = migan.inpaint(original, known_mask)
        generated = generated_512.resize(original.size, Image.Resampling.BICUBIC)
        hole_mask = Image.eval(known_mask, lambda value: 255 - value)
        repaired = Image.composite(generated, original, hole_mask)
    elif reconstructor in {"none", "original"}:
        repaired = original.copy()
        if reconstructor == "none":
            warnings.append("未运行背景修复；最终图仍会在所有阶段完成后一次性返回。")
    else:
        raise ValueError(f"未知背景修复器：{reconstructor}")

    final = repaired.copy()
    regions_by_id = {region.id: region for region in analysis.regions}
    for group in analysis.groups:
        text = translations.get(group.id, "")
        if not text.strip():
            continue
        translated_lines = split_translation_lines(text, len(group.lines))
        if translated_lines is None:
            warnings.append(f"翻译块 G{group.id} 的行边界无法安全还原；保留该块原图。")
            continue
        for index, line in enumerate(group.lines):
            line_text = translated_lines[index]
            line_regions = [regions_by_id[region_id] for region_id in line.region_ids if region_id in regions_by_id]
            if not line_regions:
                continue
            render_translation(
                final,
                _line_quad(line_regions),
                line_text,
                line_regions[0].layout.writing_mode,
                estimate_foreground_color(original, _line_quad(line_regions)),
            )
    return final, warnings


def split_translation_lines(text: str, count: int) -> list[str] | None:
    """Map one paragraph translation back to its physical source lines.

    Newlines are authoritative.  If a provider removes them, distribute the
    complete paragraph by stable text units instead of repeating the last
    translated line over every remaining region.
    """
    if count <= 0 or not text.strip():
        return None
    if count == 1:
        return [text.strip()]
    explicit = [line.strip() for line in text.splitlines()]
    if len(explicit) == count and all(explicit):
        return explicit

    units: list[str] = []
    for chunk in re.findall(r"\S+", text):
        if _is_cjk_chunk(chunk):
            units.extend(character for character in chunk if not character.isspace())
        else:
            units.append(chunk)
    if len(units) < count:
        return None

    result: list[str] = []
    start = 0
    for index in range(count):
        remaining_units = len(units) - start
        remaining_lines = count - index
        take = remaining_units if index == count - 1 else max(1, math.ceil(remaining_units / remaining_lines))
        end = min(len(units), start + take)
        result.append(_join_translation_units(units[start:end]))
        start = end
    return result if len(result) == count and start == len(units) else None


def _is_cjk_chunk(chunk: str) -> bool:
    has_cjk = any(_is_cjk_character(character) for character in chunk)
    has_ascii_word = any(character.isascii() and character.isalnum() for character in chunk)
    return has_cjk and not has_ascii_word


def _is_cjk_character(character: str) -> bool:
    value = ord(character)
    return (
        0x3040 <= value <= 0x30FF
        or 0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xAC00 <= value <= 0xD7AF
    )


def _join_translation_units(units: list[str]) -> str:
    result = ""
    for unit in units:
        if not result:
            result = unit
        elif _is_cjk_chunk(result[-1]) or _is_cjk_chunk(unit[0]):
            result += unit
        else:
            result += f" {unit}"
    return result


def build_text_mask(image: Image.Image, analysis: AnalysisResult) -> Image.Image:
    """Build a known-pixel mask from glyphs, not from the whole detector box.

    Google boxes need a little search margin because they can hug the glyph
    edges.  Local boxes can be deliberately loose, so their empty paper must
    not become an MI-GAN hole.  Both routes therefore use the box only as a
    search area and derive the actual hole from original-pixel contrast.
    """
    holes = Image.new("L", image.size, color=0)
    for region in analysis.regions:
        if not region.text.strip():
            continue
        search_quad = (
            expanded_quad(region.quad, region.layout.writing_mode)
            if analysis.detector == "google-cloud-vision"
            else region.quad
        )
        glyph, box = glyph_mask_from_pixels(image, search_quad)
        if glyph is None or box is None:
            continue
        left, top, right, bottom = box
        # A small dilation catches antialiasing and tiny strokes outside the
        # segmentation threshold without expanding an entire loose local box.
        short_side = min(glyph.width, glyph.height)
        radius = max(1, min(6, round(short_side * 0.06)))
        dilated = glyph.filter(ImageFilter.MaxFilter(radius * 2 + 1))
        existing = holes.crop(box)
        holes.paste(ImageChops.lighter(existing, dilated), (left, top))
    # The Core ML contract is fixed at 512.  Keep the high-resolution mask
    # authoritative for the final blend, while the adapter downsamples it for
    # the model input using nearest-neighbour semantics.
    return Image.eval(holes, lambda value: 255 - value)


def glyph_mask_from_pixels(
    image: Image.Image,
    quad: list[Point],
) -> tuple[Image.Image | None, tuple[int, int, int, int] | None]:
    if len(quad) != 4:
        return None, None
    left = max(0, math.floor(min(point.x for point in quad)))
    top = max(0, math.floor(min(point.y for point in quad)))
    right = min(image.width - 1, math.ceil(max(point.x for point in quad)))
    bottom = min(image.height - 1, math.ceil(max(point.y for point in quad)))
    if right <= left or bottom <= top:
        return None, None

    width, height = right - left + 1, bottom - top + 1
    local_quad = [(point.x - left, point.y - top) for point in quad]
    outer = Image.new("1", (width, height), 0)
    ImageDraw.Draw(outer).polygon(local_quad, fill=1)
    center_x = sum(point[0] for point in local_quad) / 4
    center_y = sum(point[1] for point in local_quad) / 4
    inner_quad = [
        (center_x + (point[0] - center_x) * 0.72, center_y + (point[1] - center_y) * 0.72)
        for point in local_quad
    ]
    inner = Image.new("1", (width, height), 0)
    ImageDraw.Draw(inner).polygon(inner_quad, fill=1)

    pixels = image.load()
    border: list[tuple[int, int, int]] = []
    candidates: list[tuple[float, int, int, tuple[int, int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not outer.getpixel((x, y)):
                continue
            value = pixels[left + x, top + y]
            if inner.getpixel((x, y)):
                candidates.append((0, x, y, value))
            else:
                border.append(value)
    if not candidates:
        return None, None

    background = _mean_rgb(border or [candidate[3] for candidate in candidates])
    variance = sum(_rgb_distance(value, background) ** 2 for value in border) / max(1, len(border)) / (255 * 255)
    threshold = max(0.075 * 255, math.sqrt(variance) * 3.5 * 255)
    ink: list[tuple[int, int, int, int, int]] = []
    for _, x, y, value in candidates:
        if _rgb_distance(value, background) >= threshold:
            ink.append((x, y, value[0], value[1], value[2]))
    coverage = len(ink) / max(1, len(candidates))
    if coverage < 0.008 or coverage > 0.65:
        return None, None

    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()
    for x, y, *_ in ink:
        mask_pixels[x, y] = 255
    return mask, (left, top, right + 1, bottom + 1)


def _line_quad(regions) -> list[Point]:
    points = [point for region in regions for point in region.quad]
    if len(points) < 4:
        return []
    if len(points) == 4:
        return [Point(x=x, y=y) for x, y in _ordered_quad([(point.x, point.y) for point in points])]
    rectangle = MultiPoint([(point.x, point.y) for point in points]).minimum_rotated_rectangle
    if not rectangle.is_empty and hasattr(rectangle, "exterior"):
        coords = list(rectangle.exterior.coords)[:4]
        if len(coords) == 4:
            return [Point(x=x, y=y) for x, y in _ordered_quad(coords)]
    return [Point(x=point.x, y=point.y) for point in points[:4]]


def expanded_quad(quad: list[Point], writing_mode: str, padding_ratio: float = 0.14) -> list[Point]:
    """Return an erase-only quad with a small stroke safety margin.

    The detector quad remains the click/recognition contract.  This expanded
    quad is only used for the MI-GAN hole mask, so Google boxes that hug glyph
    edges do not leave antialiased strokes behind.
    """
    if len(quad) != 4:
        return quad
    top_left, top_right, bottom_right, bottom_left = quad
    baseline = math.hypot(top_right.x - top_left.x, top_right.y - top_left.y)
    cross = math.hypot(bottom_left.x - top_left.x, bottom_left.y - top_left.y)
    short_side = min(baseline, cross)
    if baseline <= 0 or cross <= 0 or short_side <= 0:
        return quad
    ux = (top_right.x - top_left.x) / baseline
    uy = (top_right.y - top_left.y) / baseline
    vx = (bottom_left.x - top_left.x) / cross
    vy = (bottom_left.y - top_left.y) / cross
    reading_padding = short_side * 0.05
    cross_padding = short_side * max(0.08, padding_ratio)
    along_u = reading_padding if writing_mode == "horizontal" else cross_padding
    along_v = cross_padding if writing_mode == "horizontal" else reading_padding

    def offset(point: Point, du: float, dv: float) -> Point:
        return Point(
            x=point.x + ux * du + vx * dv,
            y=point.y + uy * du + vy * dv,
        )

    return [
        offset(top_left, -along_u, -along_v),
        offset(top_right, along_u, -along_v),
        offset(bottom_right, along_u, along_v),
        offset(bottom_left, -along_u, along_v),
    ]


def encode_jpeg(image: Image.Image, quality: int = 94) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_translation(
    image: Image.Image,
    quad: list[Point],
    text: str,
    writing_mode: str,
    fill: tuple[int, int, int] = (24, 30, 38),
) -> None:
    if not text.strip() or len(quad) != 4:
        return
    width = max(
        1,
        round(max(
            math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y),
            math.hypot(quad[2].x - quad[3].x, quad[2].y - quad[3].y),
        )),
    )
    height = max(
        1,
        round(max(
            math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y),
            math.hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y),
        )),
    )
    padding = max(2, round(min(width, height) * 0.12))
    available_width = max(1, width - padding * 2)
    available_height = max(1, height - padding * 2)
    content = text.strip()
    if writing_mode == "vertical":
        content = "\n".join(character for character in content if not character.isspace())
    font = fitting_font(content, available_width, available_height, writing_mode == "vertical")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bbox = draw.multiline_textbbox((0, 0), content, font=font, spacing=0, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max(0, (width - text_width) // 2) - bbox[0]
    y = max(0, (height - text_height) // 2) - bbox[1]
    draw.multiline_text((x, y), content, font=font, fill=(*fill, 255), spacing=0, align="center")

    angle = math.degrees(math.atan2(quad[1].y - quad[0].y, quad[1].x - quad[0].x))
    rotated = layer.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
    center_x = sum(point.x for point in quad) / 4
    center_y = sum(point.y for point in quad) / 4
    image.paste(
        rotated,
        (round(center_x - rotated.width / 2), round(center_y - rotated.height / 2)),
        rotated,
    )


def estimate_foreground_color(image: Image.Image, quad: list[Point]) -> tuple[int, int, int]:
    """Estimate ink colour from the original pixels before background repair."""
    if len(quad) != 4:
        return (24, 30, 38)
    min_x = max(0, math.floor(min(point.x for point in quad)))
    max_x = min(image.width - 1, math.ceil(max(point.x for point in quad)))
    min_y = max(0, math.floor(min(point.y for point in quad)))
    max_y = min(image.height - 1, math.ceil(max(point.y for point in quad)))
    if max_x < min_x or max_y < min_y:
        return (24, 30, 38)

    outer = Image.new("1", image.size, 0)
    ImageDraw.Draw(outer).polygon([(point.x, point.y) for point in quad], fill=1)
    center = Point(
        x=sum(point.x for point in quad) / 4,
        y=sum(point.y for point in quad) / 4,
    )
    inner = Image.new("1", image.size, 0)
    ImageDraw.Draw(inner).polygon(
        [
            (center.x + (point.x - center.x) * 0.72, center.y + (point.y - center.y) * 0.72)
            for point in quad
        ],
        fill=1,
    )
    pixels = image.load()
    border: list[tuple[int, int, int]] = []
    ink: list[tuple[int, int, int]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if not outer.getpixel((x, y)):
                continue
            value = pixels[x, y]
            if inner.getpixel((x, y)):
                ink.append(value)
            else:
                border.append(value)
    if not ink:
        return (24, 30, 38)
    background = _mean_rgb(border or ink)
    ranked = sorted(ink, key=lambda value: _rgb_distance(value, background), reverse=True)
    selected = ranked[: max(1, len(ranked) // 8)]
    return tuple(round(sum(value[index] for value in selected) / len(selected)) for index in range(3))


def _mean_rgb(values: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    count = max(1, len(values))
    return tuple(sum(value[index] for value in values) / count for index in range(3))


def _rgb_distance(value: tuple[int, int, int], background: tuple[float, float, float]) -> float:
    return math.sqrt(sum((value[index] - background[index]) ** 2 for index in range(3)))


def fitting_font(text: str, width: int, height: int, vertical: bool):
    font_path = _font_path()
    upper = max(8, min(72, round(height * (0.72 if vertical else 0.78))))
    lower = 8
    for size in range(upper, lower - 1, -1):
        try:
            font = ImageFont.truetype(str(font_path), size=size) if font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        box = probe.multiline_textbbox((0, 0), text, font=font, spacing=0, align="center")
        if box[2] - box[0] <= width and box[3] - box[1] <= height:
            return font
    return ImageFont.load_default()


def _font_path() -> Path | None:
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Hiragino Sans W3.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    return next((Path(path) for path in candidates if Path(path).is_file()), None)


def _ordered_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    ordered = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1])
    ordered = ordered[start:] + ordered[:start]
    if len(ordered) == 4:
        cross = (
            (ordered[1][0] - ordered[0][0]) * (ordered[2][1] - ordered[1][1])
            - (ordered[1][1] - ordered[0][1]) * (ordered[2][0] - ordered[1][0])
        )
        if cross < 0:
            ordered.reverse()
            start = min(range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1])
            ordered = ordered[start:] + ordered[:start]
    return ordered
