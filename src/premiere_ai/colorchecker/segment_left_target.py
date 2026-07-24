"""Locate the ColorChecker with SAM3, then segment the left target panel —
the white-balance grey card on the "classic" page, or the 3-step grayscale
strip on the "video" page."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", choices=["classic", "video"], default="classic")
    parser.add_argument("--image", type=Path, default=vlm.IMAGE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    page = get_page_config(args.page)
    output_path = args.output or Path(__file__).parent / f"left_target_check_{args.page}.png"

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

    mask = page.build_left_mask(predictor, crop, crop_width, crop_height)

    overlay = np.array(crop).copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    Image.fromarray(overlay).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    sys.exit(main())
