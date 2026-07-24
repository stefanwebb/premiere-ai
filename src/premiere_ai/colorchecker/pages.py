"""Shared per-page config for the Calibrite ColorChecker Passport Video 2.

The physical target has two pages, each with a "left" target panel and a
24-patch reference grid on the right:

- "classic": plain white-balance grey card (left) + Classic 24-patch target
- "video": 3-step grayscale strip — white, 40 IRE mid-tone grey, and
  saturated black (left) + video-production 24-patch target (chromatic,
  skin tone, grey balance, and highlight/shadow chips)

Each page exposes a `build_left_mask(predictor, crop, crop_width,
crop_height)` function rather than a single text prompt, because SAM3 can
segment the "classic" grey card directly, but on the "video" page a plain
text prompt run over the whole crop can't isolate the black step (it blends
into the surrounding black bezel). Instead, the white/grey steps are used to
extrapolate a rough black-step ROI, which is cropped tightly and re-queried
with a prompt naming the surrounding frame explicitly — SAM3 can separate
chip from bezel once it isn't competing with the rest of the (mostly black)
card for attention.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image


def _mask_at(result, index: int, crop_width: int, crop_height: int) -> np.ndarray:
    mask = result.masks[index]
    if mask.shape == (crop_height, crop_width):
        return mask > 0
    resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (crop_width, crop_height), Image.NEAREST
    )
    return np.array(resized) > 0


def _classic_left_mask(predictor, crop, crop_width: int, crop_height: int) -> np.ndarray:
    result = predictor.predict(crop, text_prompt="a rectangular grey panel", score_threshold=0.2)
    print(f"Found {len(result.scores)} left target patch(es)")
    mask = np.zeros((crop_height, crop_width), dtype=bool)
    for i in range(len(result.scores)):
        print(f"  left target patch {i}: score={result.scores[i]:.2f} box={result.boxes[i].tolist()}")
        mask |= _mask_at(result, i, crop_width, crop_height)
    return mask


def _best_on_left_half(result, crop_width: int):
    candidates = [i for i, box in enumerate(result.boxes) if box[2] < crop_width * 0.55]
    if not candidates:
        return None
    return max(candidates, key=lambda i: result.scores[i])


GREY_STEP_PROMPTS = ["a grey patch", "a rectangular grey panel"]


def _predict_left_half_with_fallback(predictor, crop, crop_width: int, prompts: list[str], score_threshold: float):
    for prompt in prompts:
        result = predictor.predict(crop, text_prompt=prompt, score_threshold=score_threshold)
        idx = _best_on_left_half(result, crop_width)
        if idx is not None:
            return result, idx, prompt
    return None, None, None


def _resegment_step(predictor, crop, crop_width, crop_height, roi, prompt, pad=20):
    """Re-query a rough ROI on its own, tightly cropped (with a prompt that
    names the surrounding frame), so SAM3 can separate a chip from a
    same-tone bezel/neighbor instead of competing for attention with the
    whole card. Returns a full-size mask, or None if nothing matched."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    sub_x1 = max(0, roi_x1 - pad)
    sub_y1 = max(0, roi_y1 - pad)
    sub_x2 = min(crop_width, roi_x2 + pad)
    sub_y2 = min(crop_height, roi_y2 + pad)
    subcrop = crop.crop((sub_x1, sub_y1, sub_x2, sub_y2))
    sub_w, sub_h = subcrop.size

    result = predictor.predict(subcrop, text_prompt=prompt, score_threshold=0.3)
    if len(result.scores) == 0:
        return None

    idx = int(np.argmax(result.scores))
    print(f"    score={result.scores[idx]:.2f} box={result.boxes[idx].tolist()}")
    sub_mask = _mask_at(result, idx, sub_w, sub_h)
    full_mask = np.zeros((crop_height, crop_width), dtype=bool)
    full_mask[sub_y1:sub_y2, sub_x1:sub_x2] |= sub_mask
    return full_mask


def video_left_step_masks(predictor, crop, crop_width: int, crop_height: int) -> dict[str, np.ndarray]:
    """Segment the video page's 3-step grayscale strip, keeping white/grey/
    black as separate masks (rather than a single union) — needed by
    anything that wants each step's own measured color, e.g. a tone-curve
    fit against their published IRE targets.

    Only the grey step is located by a direct whole-crop text prompt — it's
    the one that's matched reliably across every test image so far. White
    and black are each *extrapolated* from the grey step's own box width
    (assuming three contiguous equal-width columns) and then re-segmented
    within a tight, padded sub-crop, the same technique already proven for
    the black step: under dim/tinted lighting a plain "white patch" prompt
    can measure closer to grey than to white and match the wrong column
    entirely (confirmed on a real captured frame), so a direct whole-crop
    semantic match for "white" isn't trustworthy enough to rely on alone.
    """
    empty = np.zeros((crop_height, crop_width), dtype=bool)
    steps = {"white": empty.copy(), "grey": empty.copy(), "black": empty.copy()}

    grey_result, grey_idx, grey_prompt = _predict_left_half_with_fallback(
        predictor, crop, crop_width, GREY_STEP_PROMPTS, 0.15
    )
    if grey_idx is None:
        print("Could not find the grey step; skipping the left target panel.")
        return steps

    print(f"  grey step: prompt={grey_prompt!r} score={grey_result.scores[grey_idx]:.2f} box={grey_result.boxes[grey_idx].tolist()}")
    steps["grey"] = _mask_at(grey_result, grey_idx, crop_width, crop_height)

    gx1, gy1, gx2, gy2 = [float(v) for v in grey_result.boxes[grey_idx]]
    pitch = gx2 - gx1

    white_roi = (
        max(0, round(gx1 - pitch)), max(0, round(gy1)),
        min(crop_width, round(gx2 - pitch)), min(crop_height, round(gy2)),
    )
    black_roi = (
        max(0, round(gx1 + pitch)), max(0, round(gy1)),
        min(crop_width, round(gx2 + pitch)), min(crop_height, round(gy2)),
    )
    print(f"  white step ROI (extrapolated): box={list(white_roi)}")
    white_mask = _resegment_step(
        predictor, crop, crop_width, crop_height, white_roi,
        "a rectangular white or light target inside a dark frame",
    )
    print(f"  black step ROI (extrapolated): box={list(black_roi)}")
    black_mask = _resegment_step(
        predictor, crop, crop_width, crop_height, black_roi,
        "a rectangular black target inside a black frame",
    )

    if white_mask is None:
        print("  white step: no match in sub-crop, falling back to extrapolated box.")
        white_mask = empty.copy()
        white_mask[white_roi[1]:white_roi[3], white_roi[0]:white_roi[2]] = True
    if black_mask is None:
        print("  black step: no match in sub-crop, falling back to extrapolated box.")
        black_mask = empty.copy()
        black_mask[black_roi[1]:black_roi[3], black_roi[0]:black_roi[2]] = True

    steps["white"] = white_mask
    steps["black"] = black_mask
    return steps


def _video_left_mask(predictor, crop, crop_width: int, crop_height: int) -> np.ndarray:
    steps = video_left_step_masks(predictor, crop, crop_width, crop_height)
    return steps["white"] | steps["grey"] | steps["black"]


@dataclass(frozen=True)
class PageConfig:
    build_left_mask: Callable[..., np.ndarray]
    square_prompt: str
    square_score_threshold: float


PAGES = {
    "classic": PageConfig(
        build_left_mask=_classic_left_mask,
        square_prompt="a color square",
        square_score_threshold=0.3,
    ),
    "video": PageConfig(
        build_left_mask=_video_left_mask,
        square_prompt="a color square",
        square_score_threshold=0.3,
    ),
}


def get_page_config(page: str) -> PageConfig:
    try:
        return PAGES[page]
    except KeyError:
        raise ValueError(f"Unknown page {page!r}; choose from {sorted(PAGES)}") from None
