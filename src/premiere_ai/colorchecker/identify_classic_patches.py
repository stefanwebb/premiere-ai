"""Locate the ColorChecker with SAM3, segment the 24 Classic-page squares,
recover their (row, col) grid layout from pixel geometry alone, and
sanity-check that layout against a colorimetric self-consistency signal
(the neutral grayscale ramp should be monotonic) — all without needing the
manufacturer's reference Lab values yet.

This also renders an annotated image with each patch's recovered (row, col)
so the grid can be checked by eye against the physical chart's known
layout.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker import detect as vlm
from premiere_ai.colorchecker.classic_reference import ORIG_SHAPE, REFERENCE_NAMES, REFERENCE_POSITIONS, REFERENCE_RGBS
from premiere_ai.colorchecker.pages import get_page_config
from premiere_ai.colorchecker.patch_grid import assign_grid, best_axis_swap_transform, match_to_reference, mean_patch_color, verify_neutral_ramp

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20
OUTPUT_PATH = Path(__file__).parent / "classic_patch_grid_check.png"


def main() -> None:
    page = get_page_config("classic")
    image_path = vlm.IMAGE_PATH
    with Image.open(image_path) as img:
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
    boxes = [tuple(float(v) for v in box) for box in square_result.boxes]
    n = len(boxes)
    print(f"Found {n} color squares")

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

    verification = verify_neutral_ramp(grid_colors)
    print(f"\nNeutral ramp check: {verification['axis']} {verification['index']}")
    print(f"  luminances: {verification['luminances']}")
    print(f"  monotonic: {verification['monotonic']} ({verification['direction']})")
    if not verification["monotonic"]:
        print("  WARNING: neutral ramp is not monotonic — grid orientation may be wrong,")
        print("  or clustering mis-assigned a chromatic row/col as the neutral one.")

    assignment, residuals = match_to_reference(grid_colors, REFERENCE_RGBS)
    transform_name, transform_result = best_axis_swap_transform(assignment, REFERENCE_POSITIONS, ORIG_SHAPE)
    print(f"\nChromatic identity check (Hungarian match vs. published Classic reference RGBs):")
    print(f"  best-fitting orientation: {transform_name}")
    print(f"  {transform_result['matches']}/{len(assignment)} patches consistent with that orientation")
    if transform_result["mismatches"]:
        print("  WARNING: inconsistent patches (grid-recovery or color-match error):")
        for m in transform_result["mismatches"]:
            name = REFERENCE_NAMES[m["ref_index"]]
            print(
                f"    cell {m['cell']} matched to {name!r} (ΔRGB={residuals[m['cell']]:.1f}), "
                f"but that orientation predicts it belongs at {m['predicted_cell']}"
            )

    print("\nPer-cell identity + match residual:")
    for row in range(n_rows):
        parts = []
        for col in range(n_cols):
            ref_idx = assignment[(row, col)]
            parts.append(f"({row},{col})={REFERENCE_NAMES[ref_idx]} [dRGB={residuals[(row, col)]:.1f}]")
        print("  " + "  ".join(parts))

    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    for (row, col), box in zip(grid_coords, boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        ref_idx = assignment[(row, col)]
        label = f"{row},{col}\n{REFERENCE_NAMES[ref_idx]}"
        draw.text((cx - 30, cy - 12), label, fill=(255, 0, 0))
    overlay.save(OUTPUT_PATH)
    print(f"\nSaved annotated grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
