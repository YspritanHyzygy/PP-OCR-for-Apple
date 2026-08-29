from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import DetectedTextRegion, TextFlowGroup, TextFlowLine


MAX_GAP_RATIO = 1.6
MIN_OVERLAP_RATIO = 0.3
MAX_ANGLE_DELTA = 0.15
# Translation context may include a heading and smaller body copy.  Rendering
# still keeps every detector line independent, so this can be looser than the
# visual-block grouping rule used by the iOS overlay.
MAX_CROSS_SIZE_RATIO = 1.65
MAX_LINES_PER_GROUP = 60


@dataclass
class _LogicalLine:
    regions: list[DetectedTextRegion]

    @property
    def writing_mode(self) -> str:
        return self.regions[0].layout.writing_mode

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        points = [point for region in self.regions for point in region.quad]
        return (
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    @property
    def cross_size(self) -> float:
        return max(_cross_size(region) for region in self.regions)

    @property
    def angle(self) -> float:
        return sum(_angle(region) for region in self.regions) / len(self.regions)

    @property
    def text(self) -> str:
        ordered = sorted(
            self.regions,
            key=(
                (lambda region: _bounds(region)[0])
                if self.writing_mode == "horizontal"
                else (lambda region: _bounds(region)[1])
            ),
        )
        return _join_fragments([region.text for region in ordered])


def group_regions(regions: list[DetectedTextRegion]) -> list[TextFlowGroup]:
    for region in regions:
        region.group_id = None
        region.line_index = None

    logical_lines = _assemble_physical_lines(
        [region for region in regions if _carries_content(region.text)]
    )
    groups: list[list[_LogicalLine]] = []
    for mode in ("horizontal", "vertical"):
        lines = [line for line in logical_lines if line.writing_mode == mode]
        lines.sort(key=_reading_key)
        for line in lines:
            candidates = [
                group
                for group in groups
                if group[0].writing_mode == mode
                and _same_block(group[-1], line)
                and _compatible_with_group(group, line)
            ]
            if not candidates:
                groups.append([line])
                continue
            best = min(
                candidates,
                key=lambda group: max(
                    0,
                    _block_gap(group[-1].bounds, line.bounds, mode),
                ),
            )
            best.append(line)

    groups.sort(key=lambda lines: _reading_key(lines[0]))
    output: list[TextFlowGroup] = []
    for group_id, lines in enumerate(groups, start=1):
        flow_lines: list[TextFlowLine] = []
        for line_index, line in enumerate(lines, start=1):
            ordered_regions = sorted(
                line.regions,
                key=(
                    (lambda region: _bounds(region)[0])
                    if line.writing_mode == "horizontal"
                    else (lambda region: _bounds(region)[1])
                ),
            )
            for region in ordered_regions:
                region.group_id = group_id
                region.line_index = line_index
            flow_lines.append(
                TextFlowLine(
                    index=line_index,
                    region_ids=[region.id for region in ordered_regions],
                    text=line.text,
                )
            )
        output.append(
            TextFlowGroup(
                id=group_id,
                writing_mode=lines[0].writing_mode,
                lines=flow_lines,
                text="\n".join(line.text for line in lines),
            )
        )
    return output


def _assemble_physical_lines(regions: list[DetectedTextRegion]) -> list[_LogicalLine]:
    ordered = sorted(
        regions,
        key=lambda region: (
            0 if region.layout.writing_mode == "horizontal" else 1,
            _bounds(region)[1],
            _bounds(region)[0],
        ),
    )
    lines: list[_LogicalLine] = []
    for region in ordered:
        candidates = [line for line in lines if _same_physical_line(line, region)]
        if not candidates:
            lines.append(_LogicalLine([region]))
            continue
        best = min(candidates, key=lambda line: _inline_gap(line.bounds, _bounds(region), line.writing_mode))
        best.regions.append(region)
    return lines


def _same_physical_line(line: _LogicalLine, region: DetectedTextRegion) -> bool:
    if line.writing_mode != region.layout.writing_mode:
        return False
    if _angle_delta(line.angle, _angle(region)) > MAX_ANGLE_DELTA:
        return False
    sizes = [line.cross_size, _cross_size(region)]
    if min(sizes) <= 0 or max(sizes) / min(sizes) > MAX_CROSS_SIZE_RATIO:
        return False
    lhs, rhs = line.bounds, _bounds(region)
    cross_overlap, narrower = _cross_overlap(lhs, rhs, line.writing_mode)
    same_band = narrower > 0 and cross_overlap / narrower >= 0.55
    gap = _inline_gap(lhs, rhs, line.writing_mode)
    return same_band and gap <= max(sizes) * MAX_GAP_RATIO


def _same_block(lhs: _LogicalLine, rhs: _LogicalLine) -> bool:
    if lhs.writing_mode != rhs.writing_mode:
        return False
    if _angle_delta(lhs.angle, rhs.angle) > MAX_ANGLE_DELTA:
        return False
    sizes = [lhs.cross_size, rhs.cross_size]
    if min(sizes) <= 0 or max(sizes) / min(sizes) > MAX_CROSS_SIZE_RATIO:
        return False
    overlap, narrower = _inline_overlap(lhs.bounds, rhs.bounds, lhs.writing_mode)
    if narrower <= 0 or overlap / narrower < MIN_OVERLAP_RATIO:
        return False
    return _block_gap(lhs.bounds, rhs.bounds, lhs.writing_mode) <= max(sizes) * MAX_GAP_RATIO


def _compatible_with_group(lines: list[_LogicalLine], candidate: _LogicalLine) -> bool:
    if len(lines) >= MAX_LINES_PER_GROUP:
        return False
    anchor = lines[0]
    sizes = [anchor.cross_size, candidate.cross_size]
    if min(sizes) <= 0 or max(sizes) / min(sizes) > MAX_CROSS_SIZE_RATIO:
        return False
    overlap, narrower = _inline_overlap(anchor.bounds, candidate.bounds, anchor.writing_mode)
    return narrower > 0 and overlap / narrower >= MIN_OVERLAP_RATIO


def _reading_key(line: _LogicalLine) -> tuple[float, float]:
    x0, y0, x1, _ = line.bounds
    return (y0, x0) if line.writing_mode == "horizontal" else (-x1, y0)


def _bounds(region: DetectedTextRegion) -> tuple[float, float, float, float]:
    xs = [point.x for point in region.quad]
    ys = [point.y for point in region.quad]
    return min(xs), min(ys), max(xs), max(ys)


def _angle(region: DetectedTextRegion) -> float:
    first, second = region.quad[0], region.quad[1]
    return math.atan2(second.y - first.y, second.x - first.x)


def _angle_delta(lhs: float, rhs: float) -> float:
    delta = abs(lhs - rhs) % math.pi
    return min(delta, math.pi - delta)


def _cross_size(region: DetectedTextRegion) -> float:
    first, fourth = region.quad[0], region.quad[3]
    return math.hypot(fourth.x - first.x, fourth.y - first.y)


def _cross_overlap(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
    mode: str,
) -> tuple[float, float]:
    if mode == "horizontal":
        return min(lhs[3], rhs[3]) - max(lhs[1], rhs[1]), min(lhs[3] - lhs[1], rhs[3] - rhs[1])
    return min(lhs[2], rhs[2]) - max(lhs[0], rhs[0]), min(lhs[2] - lhs[0], rhs[2] - rhs[0])


def _inline_overlap(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
    mode: str,
) -> tuple[float, float]:
    if mode == "horizontal":
        return min(lhs[2], rhs[2]) - max(lhs[0], rhs[0]), min(lhs[2] - lhs[0], rhs[2] - rhs[0])
    return min(lhs[3], rhs[3]) - max(lhs[1], rhs[1]), min(lhs[3] - lhs[1], rhs[3] - rhs[1])


def _inline_gap(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
    mode: str,
) -> float:
    if mode == "horizontal":
        return max(0, max(lhs[0], rhs[0]) - min(lhs[2], rhs[2]))
    return max(0, max(lhs[1], rhs[1]) - min(lhs[3], rhs[3]))


def _block_gap(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
    mode: str,
) -> float:
    return rhs[1] - lhs[3] if mode == "horizontal" else lhs[0] - rhs[2]


def _carries_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _join_fragments(parts: list[str]) -> str:
    output = ""
    for part in parts:
        if not output:
            output = part
        elif _is_cjk(output[-1]) or _is_cjk(part[0]):
            output += part
        else:
            output += f" {part}"
    return output


def _is_cjk(character: str) -> bool:
    value = ord(character)
    return (
        0x3040 <= value <= 0x30FF
        or 0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xAC00 <= value <= 0xD7AF
        or 0xFF00 <= value <= 0xFF60
    )
