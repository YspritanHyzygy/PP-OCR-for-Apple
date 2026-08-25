#!/usr/bin/env python3
"""Generate the locked French, Spanish, and German accent corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SEED = 20260824
FONT_SHA256 = "b85c38ecea8a7cfb39c24e395a4007474fa5a4fc864f6ee33309eb4948d232d5"
SAMPLES_PER_LANGUAGE = 60
PHRASES = {
    "fr": [
        "Café déjà ouvert", "Crème brûlée à emporter", "Été, Noël et maïs",
        "Où est l'hôtel ?", "Garçon, un thé s'il vous plaît", "Cœur de bœuf",
        "À bientôt, chère Élise", "Forêt près de Besançon", "Ça coûte 12,50 €",
        "L'œuvre est achevée", "Pêche fraîche du marché", "Numéro réservé aux élèves",
    ],
    "es": [
        "El niño pidió piña", "Mañana a las 9:30", "¿Dónde está el baño?",
        "¡Última función!", "Canción número veintidós", "Pingüino en el océano",
        "Café, azúcar y limón", "Información turística", "Señalización pública",
        "Árboles junto al río", "Qué difícil decisión", "Precio: 18,75 €",
    ],
    "de": [
        "Grüße aus Köln", "Fußgängerüberweg", "Öffnungszeiten geändert",
        "Äpfel und süße Brötchen", "Straße zum Hauptbahnhof", "Für Gäste geöffnet",
        "Große Auswahl", "Münchner Küche", "Später zurück, danke",
        "Übermäßiger Lärm", "Zwölf Bücher", "Preis: 19,90 €",
    ],
}
PALETTES = [
    ((245, 222, 82), (20, 61, 122)),
    ((42, 92, 77), (244, 236, 213)),
    ((210, 72, 58), (250, 244, 225)),
    ((226, 235, 247), (101, 42, 124)),
    ((40, 44, 52), (238, 188, 64)),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(font_path: Path, output: Path) -> Path:
    if sha256(font_path) != FONT_SHA256:
        raise SystemExit(f"{font_path}: does not match the locked Noto Sans SHA-256")
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    samples = []
    randomizer = random.Random(SEED)
    for language, phrases in PHRASES.items():
        for index in range(SAMPLES_PER_LANGUAGE):
            text = unicodedata.normalize("NFC", phrases[index % len(phrases)])
            background, foreground = PALETTES[randomizer.randrange(len(PALETTES))]
            size = randomizer.choice((42, 48, 56, 64))
            font = ImageFont.truetype(str(font_path), size=size)
            image = Image.new("RGB", (1024, 256), background)
            draw = ImageDraw.Draw(image)
            bounds = draw.textbbox((0, 0), text, font=font)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            x = randomizer.randint(36, max(36, 1024 - width - 36))
            y = randomizer.randint(28, max(28, 256 - height - 28)) - bounds[1]
            draw.text((x, y), text, font=font, fill=foreground)
            polygon = [[x, y + bounds[1]], [x + width, y + bounds[1]],
                       [x + width, y + bounds[3]], [x, y + bounds[3]]]
            name = f"{language}-{index:03d}.png"
            image.save(image_output / name, format="PNG", optimize=False, compress_level=9)
            samples.append({
                "id": f"latin-diacritic-v1:{language}:{index:03d}",
                "dataset": "latin-diacritic-v1",
                "language": language,
                "scenario": "screen-or-ui",
                "imagePath": f"images/{name}",
                "groundTruth": [{"polygon": polygon, "text": text, "ignore": False}],
            })
    corpus = output / "corpus.json"
    corpus.write_text(
        json.dumps({"schemaVersion": 1, "samples": samples}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(generate(args.font.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
