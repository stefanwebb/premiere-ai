"""Locate the ColorChecker with SAM3, then combine the left target panel's
mask(s) with the individual color squares' masks into a single image with
everything else masked out (transparent + black).

Works for either physical page of the Calibrite ColorChecker Passport Video
2 — see pages.py for what each page contains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker import detect as vlm
from premiere_ai.colorchecker.pages import get_page_config

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask > 0
    resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST
    )
    return np.array(resized) > 0


def union_masks(predictor, crop, prompt, score_threshold, crop_width, crop_height, label):
    result = predictor.predict(crop, text_prompt=prompt, score_threshold=score_threshold)
    n = len(result.scores)
    print(f"Found {n} {label}")
    mask = np.zeros((crop_height, crop_width), dtype=bool)
    for i in range(n):
        mask |= resize_mask(result.masks[i], crop_width, crop_height)
        print(f"  {label} {i}: score={result.scores[i]:.2f} box={result.boxes[i].tolist()}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page",
        choices=["classic", "video"],
        default="classic",
        help="Which physical page of the ColorChecker to extract targets from.",
    )
    parser.add_argument(
        "--image", type=Path, default=vlm.IMAGE_PATH, help="Path to the source frame."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to extracted_targets_<page>.png next to this script).",
    )
    args = parser.parse_args()

    page = get_page_config(args.page)
    output_path = args.output or Path(__file__).parent / f"extracted_targets_{args.page}.png"

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
    crop_width, crop_height = crop.size
    print(f"ColorChecker box: ({x1}, {y1}, {x2}, {y2}), score={chart_result.scores[best]:.2f}")

    left_mask = page.build_left_mask(predictor, crop, crop_width, crop_height)
    square_mask = union_masks(
        predictor, crop, page.square_prompt, page.square_score_threshold,
        crop_width, crop_height, "color squares",
    )
    combined_mask = left_mask | square_mask

    rgba = np.dstack([np.array(crop), np.full((crop_height, crop_width), 255, dtype=np.uint8)])
    rgba[~combined_mask] = (0, 0, 0, 0)

    Image.fromarray(rgba, mode="RGBA").save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    sys.exit(main())
