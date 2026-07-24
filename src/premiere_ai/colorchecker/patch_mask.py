"""Detect a Calibrite ColorChecker Passport Video 2's video page in an
image and build a boolean mask over one category of its 24-patch grid,
for restricting downstream analysis (e.g. a vectorscope trace) to just
those patches.

Reuses the same SAM3 chart-location + grid-recovery pipeline as
build_lut.py, but only needs the patch bounding boxes (not their measured
colors), so masks are built directly from boxes rather than per-patch
segmentation masks.
"""

import numpy as np
from PIL import Image
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker.pages import get_page_config
from premiere_ai.colorchecker.patch_grid import assign_grid

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20

# Fixed physical layout of the video page's 24-patch grid (see
# build_lut.py — confirmed against the actual chart, not inferred from
# saturation, which misclassifies desaturated skin patches under a color
# cast).
PATCH_CATEGORY_COLUMNS = {"chromatic": 0, "skin": 1}


def load_predictor() -> Sam3Predictor:
    model_path = get_model_path(MODEL_ID)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    return Sam3Predictor(model, processor, score_threshold=COLORCHECKER_SCORE_THRESHOLD)


def detect_patch_mask(
    image: Image.Image, category: str, predictor: Sam3Predictor | None = None
) -> np.ndarray:
    """Returns a full-resolution boolean mask (image.height, image.width)
    that is True over the six patches of `category` ("chromatic" or
    "skin") on the video page's 24-patch grid, False everywhere else
    (including the rest of the chart)."""
    if category not in PATCH_CATEGORY_COLUMNS:
        raise ValueError(
            f"Unknown category {category!r}; choose from {sorted(PATCH_CATEGORY_COLUMNS)}"
        )
    target_col = PATCH_CATEGORY_COLUMNS[category]

    page = get_page_config("video")
    predictor = predictor or load_predictor()
    image_width, image_height = image.size

    chart_result = predictor.predict(image, text_prompt=COLORCHECKER_PROMPT)
    if len(chart_result.scores) == 0:
        raise RuntimeError("No ColorChecker found in the image.")
    best = int(np.argmax(chart_result.scores))
    x1, y1, x2, y2 = [int(v) for v in chart_result.boxes[best]]
    x1, y1 = max(0, x1 - CROP_PADDING), max(0, y1 - CROP_PADDING)
    x2, y2 = min(image_width, x2 + CROP_PADDING), min(image_height, y2 + CROP_PADDING)
    crop = image.crop((x1, y1, x2, y2))
    print(f"ColorChecker box: ({x1}, {y1}, {x2}, {y2}), score={chart_result.scores[best]:.2f}")

    square_result = predictor.predict(
        crop, text_prompt=page.square_prompt, score_threshold=page.square_score_threshold
    )
    all_boxes = [tuple(float(v) for v in box) for box in square_result.boxes]
    boxes = [b for b in all_boxes if 0.6 <= (b[2] - b[0]) / (b[3] - b[1]) <= 1.6]
    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    grid_coords = assign_grid(centroids)

    mask = np.zeros((image_height, image_width), dtype=bool)
    matched = 0
    for (row, col), box in zip(grid_coords, boxes):
        if col != target_col:
            continue
        bx1, by1, bx2, by2 = [round(v) for v in box]
        mask[y1 + by1 : y1 + by2, x1 + bx1 : x1 + bx2] = True
        matched += 1
    print(f"Masked {matched} {category!r} patch(es) (grid column {target_col})")
    if matched == 0:
        raise RuntimeError(
            f"No {category!r} patches (column {target_col}) found in the recovered grid."
        )
    return mask
