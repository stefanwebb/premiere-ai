"""Locate the ColorChecker with SAM3, then subsegment the individual color
squares within the 24-patch reference grid (Classic or video-production,
depending on --page)."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", choices=["classic", "video"], default="classic")
    parser.add_argument("--image", type=Path, default=vlm.IMAGE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    page = get_page_config(args.page)
    output_path = args.output or Path(__file__).parent / f"colorchecker_squares_check_{args.page}.png"

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

    square_result = predictor.predict(
        crop, text_prompt=page.square_prompt, score_threshold=page.square_score_threshold
    )
    n_squares = len(square_result.scores)
    print(f"Found {n_squares} color squares")

    overlay = np.array(crop).copy()
    rng = np.random.default_rng(0)
    squares = []
    for i in range(n_squares):
        mask = resize_mask(square_result.masks[i], crop_width, crop_height)
        color = rng.integers(50, 255, size=3)
        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)

        sx1, sy1, sx2, sy2 = square_result.boxes[i]
        squares.append(
            {
                "score": float(square_result.scores[i]),
                # box in full-image pixel coordinates
                "box": [
                    round(x1 + sx1),
                    round(y1 + sy1),
                    round(x1 + sx2),
                    round(y1 + sy2),
                ],
            }
        )

    Image.fromarray(overlay).save(output_path)
    print(f"Saved {output_path}")

    for i, sq in enumerate(sorted(squares, key=lambda s: (s["box"][1], s["box"][0]))):
        print(f"  square {i}: score={sq['score']:.2f} box={sq['box']}")


if __name__ == "__main__":
    sys.exit(main())
