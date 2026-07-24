"""Locate the ColorChecker with SAM3, segment the 24 video-page squares,
recover their (row, col) grid layout from pixel geometry, and sanity-check
that layout with reference-free colorimetric signals.

Unlike the classic page, there's no publicly published reference table for
this chart's video-production patches (Calibrite-proprietary), so this
can't do the Hungarian identity match against known values yet — only
self-consistency checks:

- two low-saturation columns expected ("grey balance" and "highlights and
  shadows"), each independently monotonic in luminance
- the chromatic column (green/cyan/blue/magenta/red/yellow, per the
  product spec) expected to step ~60 degrees around the hue wheel each row
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker import detect as vlm
from premiere_ai.colorchecker.pages import get_page_config
from premiere_ai.colorchecker.patch_grid import assign_grid, column_profiles, mean_patch_color, verify_hue_progression

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20
OUTPUT_PATH = Path(__file__).parent / "video_patch_grid_check.png"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=vlm.IMAGE_PATH)
    args = parser.parse_args()

    page = get_page_config("video")
    with Image.open(args.image) as img:
        image = img.convert("RGB")
    image_width, image_height = image.size

    print(f"Loading {MODEL_ID}...")
    model_path = get_model_path(MODEL_ID)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=COLORCHECKER_SCORE_THRESHOLD)

    chart_result = predictor.predict(image, text_prompt=COLORCHECKER_PROMPT)
    if len(chart_result.scores) == 0:
        print("No ColorChecker found in the image.")
        return

    best = int(np.argmax(chart_result.scores))
    x1, y1, x2, y2 = [int(v) for v in chart_result.boxes[best]]
    x1 = max(0, x1 - CROP_PADDING)
    y1 = max(0, y1 - CROP_PADDING)
    x2 = min(image_width, x2 + CROP_PADDING)
    y2 = min(image_height, y2 + CROP_PADDING)
    crop = image.crop((x1, y1, x2, y2))
    crop_array = np.array(crop)
    print(f"ColorChecker box: ({x1}, {y1}, {x2}, {y2}), score={chart_result.scores[best]:.2f}")

    square_result = predictor.predict(
        crop, text_prompt=page.square_prompt, score_threshold=page.square_score_threshold
    )
    all_boxes = [tuple(float(v) for v in box) for box in square_result.boxes]
    print(f"Found {len(all_boxes)} raw color-square candidate(s)")

    # The "color square" prompt also weakly matches the grayscale strip on
    # this page's other target panel (low score, but above threshold) — its
    # boxes are tall/narrow rather than square, so an aspect-ratio filter
    # cleanly separates the false positives from the real patches.
    boxes = []
    for box in all_boxes:
        w, h = box[2] - box[0], box[3] - box[1]
        aspect = w / h
        if 0.6 <= aspect <= 1.6:
            boxes.append(box)
        else:
            print(f"  discarding non-square candidate box={[round(v, 1) for v in box]} aspect={aspect:.2f}")
    n = len(boxes)
    print(f"Kept {n} color squares after aspect-ratio filtering")

    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    try:
        grid_coords = assign_grid(centroids)
    except ValueError as e:
        print(f"Grid recovery failed: {e}")
        return

    grid_colors = {}
    for (row, col), box in zip(grid_coords, boxes):
        grid_colors[(row, col)] = mean_patch_color(crop_array, box)

    n_rows = max(r for r, c in grid_colors) + 1
    n_cols = max(c for r, c in grid_colors) + 1
    print(f"Recovered a clean {n_rows}x{n_cols} grid ({n} patches)")
    print("\nGrid (row, col): mean RGB")
    for row in range(n_rows):
        line = "  " + "  ".join(f"({row},{c})={grid_colors[(row, c)]}" for c in range(n_cols))
        print(line)

    profiles = column_profiles(grid_colors)
    print("\nColumn profiles (mean saturation, luminance sequence, monotonicity):")
    for p in sorted(profiles, key=lambda p: p["mean_saturation"]):
        print(
            f"  col {p['col']}: mean_sat={p['mean_saturation']:.3f} "
            f"monotonic={p['monotonic']} ({p['direction']}) luminances={p['luminances']}"
        )

    low_sat_cols = sorted(profiles, key=lambda p: p["mean_saturation"])[:2]
    print("\nExpected 2 neutral-ish columns (grey balance + highlights/shadows):")
    all_monotonic = True
    for p in low_sat_cols:
        status = "OK" if p["monotonic"] else "WARNING: not monotonic"
        print(f"  col {p['col']} (mean_sat={p['mean_saturation']:.3f}): {status}")
        all_monotonic = all_monotonic and p["monotonic"]
    if not all_monotonic:
        print("  WARNING: a neutral-ish column isn't monotonic — grid orientation may be wrong.")

    chromatic_col = max(profiles, key=lambda p: p["mean_saturation"])["col"]
    chromatic_colors = [grid_colors[(r, chromatic_col)] for r in range(n_rows)]
    hue_check = verify_hue_progression(chromatic_colors)
    print(f"\nChromatic column (col {chromatic_col}) hue progression check:")
    print(f"  hues: {hue_check['hues']}")
    print(f"  forward diffs: {hue_check['forward_diffs']}")
    print(f"  consistent ~60deg/step: {hue_check['consistent']} (direction={hue_check['direction']})")
    if not hue_check["consistent"]:
        print("  WARNING: chromatic column doesn't show the expected ~60deg hue steps —")
        print("  grid orientation may be wrong, or this isn't the chromatic column.")

    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    for (row, col), box in zip(grid_coords, boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        draw.text((cx - 10, cy - 6), f"{row},{col}", fill=(255, 0, 0))
    overlay.save(OUTPUT_PATH)
    print(f"\nSaved annotated grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
