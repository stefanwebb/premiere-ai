"""Segment the ColorChecker with SAM3 and confirm the VLM's predicted
center point actually lands inside the resulting segment."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker import detect as vlm

MODEL_ID = "mlx-community/sam3-4bit"
TEXT_PROMPT = "a colorchecker color calibration chart"
SCORE_THRESHOLD = 0.3
OUTPUT_PATH = Path(__file__).parent / "colorchecker_segment_check.png"


def mask_containing_point(masks: np.ndarray, x: int, y: int) -> int | None:
    for i, mask in enumerate(masks):
        if mask[y, x] > 0:
            return i
    return None


def main() -> None:
    image_path = vlm.IMAGE_PATH
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    image_width, image_height = image.size

    # Independently re-derive the VLM's center point (not just reusing a
    # cached answer) so this is a genuine cross-check.
    content = vlm.query_vlm(image_path)
    parsed = vlm.parse_model_response(content)
    vlm_result = vlm.to_pixel_result(parsed, image_width, image_height)

    if not vlm_result.present or vlm_result.center is None:
        print("VLM did not report a ColorChecker center point; nothing to confirm.")
        return

    point_x, point_y = vlm_result.center.x, vlm_result.center.y
    print(f"VLM center point: ({point_x}, {point_y})")

    print(f"Loading {MODEL_ID}...")
    model_path = get_model_path(MODEL_ID)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=SCORE_THRESHOLD)

    result = predictor.predict(image, text_prompt=TEXT_PROMPT)
    print(f"SAM3 found {len(result.scores)} candidate(s) for {TEXT_PROMPT!r}")

    # Masks may come back at a lower resolution than the source image.
    masks_full_res = []
    for mask in result.masks:
        if mask.shape != (image_height, image_width):
            mask = np.array(
                Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                    (image_width, image_height), Image.NEAREST
                )
            )
        masks_full_res.append(mask > 0)

    match_idx = mask_containing_point(masks_full_res, point_x, point_y)

    overlay = np.array(image).copy()
    if match_idx is None:
        print("No SAM3 segment contains the VLM's predicted center point.")
    else:
        score = result.scores[match_idx]
        box = result.boxes[match_idx]
        print(f"VLM point falls inside segment {match_idx} (score={score:.2f}, box={box.tolist()})")
        binary = masks_full_res[match_idx]
        overlay[binary] = (overlay[binary] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    out_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out_img)
    r = 25
    draw.line([point_x - r, point_y, point_x + r, point_y], fill=(255, 0, 0), width=8)
    draw.line([point_x, point_y - r, point_x, point_y + r], fill=(255, 0, 0), width=8)
    draw.ellipse([point_x - r, point_y - r, point_x + r, point_y + r], outline=(255, 0, 0), width=8)

    out_img.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
