"""Crop the raw app screenshots into the shots the video shows.

Run from the repository root:  uv run python video/scripts/crop_shots.py
Reads video/scripts/shots.json; writes video/assets/shots/<name>.png and
video/assets/shots/shots.js (pixel sizes, so the compositions can keep aspect ratios).
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "raw"
OUT = ROOT / "assets" / "shots"


def main():
    spec = json.loads((ROOT / "scripts" / "shots.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, item in spec.items():
        if name.startswith("_"):
            continue
        image = Image.open(RAW / item["source"]).convert("RGB")
        left, top, right, bottom = item["box"]
        if right > image.width or bottom > image.height:
            raise SystemExit(f"{name}: box {item['box']} exceeds {image.size}")
        crop = image.crop((left, top, right, bottom))
        crop.save(OUT / f"{name}.png", optimize=True)
        sizes[name] = {"width": crop.width, "height": crop.height, "source": item["source"]}
        print(f"{name}: {item['source']} {item['box']} -> {crop.size}")
    (OUT / "shots.js").write_text("window.SHOTS = " + json.dumps(sizes, indent=1) + ";\n")


if __name__ == "__main__":
    main()
