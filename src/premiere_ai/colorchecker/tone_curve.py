"""Fit a monotonic per-channel (identical R=G=B) tone curve from the video
page's grey-balance patches against their published IRE targets.

Reference values (from the Calibrite ColorChecker Video guide + "what the
patch" writeups — see conversation for sources):
  - left-panel 3-step strip: white ~95% IRE, mid-grey exactly 40% IRE,
    black ~5% IRE (the guide gives ranges for a related 4-chip layout —
    90-100%, 40-50%, 0-10% — the single "40 IRE" figure is stated exactly
    for the mid-tone step, the others use each range's midpoint here)
  - the 24-patch grid's grey-balance column: a ramp evenly spanning
    20-90% IRE across its 6 steps

CAVEATS (this is a *partial* fit, not a finished color pipeline):
  - Only corrects overall tone/exposure response (identical curve on R, G,
    B) — it does NOT correct white balance/color cast or chromatic
    response, since no chromatic reference values exist for this chart yet
    (see conversation — Calibrite doesn't publish them).
  - IRE percent -> 8-bit code value assumes full-range (0-100% -> 0-255),
    not legal/broadcast range (16-235) — verify against your camera's
    actual output range before trusting this for anything but a rough
    exposure/contrast match.
  - The left-panel white/black targets are range midpoints, not exact
    figures like the grey step's confirmed 40% — treat those two anchors
    as the least certain points on the curve.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_erosion
from sklearn.isotonic import IsotonicRegression

from premiere_ai.colorchecker.pages import get_page_config, video_left_step_masks
from premiere_ai.colorchecker.patch_grid import assign_grid, column_profiles, mean_patch_color

COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
CROP_PADDING = 20

# IRE-percent targets. Left-panel steps use each published range's midpoint
# except the grey step, which the spec states exactly as 40 IRE.
LEFT_STEP_TARGETS_IRE = {"white": 95.0, "grey": 40.0, "black": 5.0}
# The grid ramp's 6 steps evenly span 20-90 IRE; direction (which row is
# which end) is resolved from measured luminance, not assumed.
RAMP_IRE_RANGE = (90.0, 20.0)  # (row nearest 90%, row nearest 20%)


def mean_mask_color(image: np.ndarray, mask: np.ndarray, erode_iterations: int = 5) -> tuple[float, float, float] | None:
    eroded = binary_erosion(mask, iterations=erode_iterations)
    if not eroded.any():
        eroded = mask
    if not eroded.any():
        return None
    region = image[eroded]
    return tuple(region.mean(axis=0))


def luma(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fit_tone_curve(predictor, image_path: Path) -> dict:
    """Run the video-page grey-balance anchor extraction + monotonic curve
    fit, returning everything a caller needs to both report on it and apply
    it to other images: {"curve", "measured_min", "measured_max", "anchors"}.
    `curve(x)` expects x already clipped to [measured_min, measured_max].
    """
    page = get_page_config("video")
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    image_width, image_height = image.size

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
    crop_array = np.array(crop).astype(float)
    crop_width, crop_height = crop.size
    print(f"ColorChecker box: ({x1}, {y1}, {x2}, {y2}), score={chart_result.scores[best]:.2f}")

    anchors = []  # list of (measured_luma, target_code, label)

    print("\nLeft-panel 3-step strip:")
    step_masks = video_left_step_masks(predictor, crop, crop_width, crop_height)
    for name, target_ire in LEFT_STEP_TARGETS_IRE.items():
        mean_rgb = mean_mask_color(crop_array, step_masks[name])
        if mean_rgb is None:
            print(f"  {name}: no mask found, skipping")
            continue
        measured_luma = luma(mean_rgb)
        target_code = target_ire / 100 * 255
        print(f"  {name}: measured RGB={tuple(round(v, 1) for v in mean_rgb)} luma={measured_luma:.1f} -> target {target_ire}% IRE = code {target_code:.1f}")
        anchors.append((measured_luma, target_code, f"left:{name}"))

    print("\n24-patch grid, grey-balance ramp column:")
    square_result = predictor.predict(
        crop, text_prompt=page.square_prompt, score_threshold=page.square_score_threshold
    )
    all_boxes = [tuple(float(v) for v in box) for box in square_result.boxes]
    boxes = [b for b in all_boxes if 0.7 <= (b[2] - b[0]) / (b[3] - b[1]) <= 1.4]
    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    grid_coords = assign_grid(centroids)
    grid_colors = {cell: mean_patch_color(np.array(crop), box) for cell, box in zip(grid_coords, boxes)}

    profiles = column_profiles(grid_colors)
    monotonic_profiles = [p for p in profiles if p["monotonic"]]
    if not monotonic_profiles:
        print("  No monotonic column found; skipping the grid ramp.")
    else:
        ramp_col = min(monotonic_profiles, key=lambda p: p["mean_saturation"])["col"]
        n_rows = max(r for r, c in grid_colors) + 1
        line = [grid_colors[(r, ramp_col)] for r in range(n_rows)]
        line_lumas = [luma(rgb) for rgb in line]
        # RAMP_IRE_RANGE[0] (90%) goes with the brighter end, regardless of
        # which row that happens to be.
        bright_first = RAMP_IRE_RANGE if line_lumas[0] >= line_lumas[-1] else RAMP_IRE_RANGE[::-1]
        targets_ire = np.linspace(bright_first[0], bright_first[1], n_rows)
        print(f"  using col {ramp_col} as the grey-balance ramp")
        for row, (rgb, target_ire) in enumerate(zip(line, targets_ire)):
            measured_luma = luma(rgb)
            target_code = target_ire / 100 * 255
            print(f"  row {row}: measured RGB={tuple(round(v, 1) for v in rgb)} luma={measured_luma:.1f} -> target {target_ire:.1f}% IRE = code {target_code:.1f}")
            anchors.append((measured_luma, target_code, f"ramp:row{row}"))

    if len(anchors) < 2:
        print("\nNot enough anchors to fit a curve.")
        return None

    anchors.sort(key=lambda a: a[0])
    measured = np.array([a[0] for a in anchors])
    target = np.array([a[1] for a in anchors])

    if not np.all(np.diff(measured) > 0):
        print("\nWARNING: measured lumas are not strictly increasing across anchors —")
        print("duplicate/near-duplicate measurements will make the fit unstable:")
        for m, t, label in anchors:
            print(f"  {label}: measured_luma={m:.1f} target_code={t:.1f}")

    # The left-panel strip and the grid ramp are physically separate
    # patches (different lighting/angle in a flat product photo can easily
    # make them inconsistent even though both are nominally IRE-calibrated),
    # so the raw targets aren't guaranteed monotonic in measured-luma order.
    # Isotonic regression finds the closest non-decreasing sequence — the
    # minimal, principled adjustment rather than silently trusting a target
    # order that would produce a tone curve with a dip in it.
    adjusted_target = IsotonicRegression(increasing=True).fit_transform(measured, target)
    changed = ~np.isclose(target, adjusted_target)
    if changed.any():
        print("\nIsotonic adjustment (raw targets weren't monotonic in measured-luma order):")
        for (m, t, label), adj, was_changed in zip(anchors, adjusted_target, changed):
            if was_changed:
                print(f"  {label}: measured_luma={m:.1f} raw_target={t:.1f} -> adjusted={adj:.1f}")

    # Extrapolate=False + clamping to the calibrated range (rather than
    # letting PCHIP extrapolate cubics past its endpoints) avoids the
    # overshoot/dip a cubic can produce right at the boundary — outside the
    # measured range there's no data to inform behavior anyway, so holding
    # flat at the nearest known value is the safer default.
    curve = PchipInterpolator(measured, adjusted_target, extrapolate=False)

    return {
        "curve": curve,
        "measured_min": float(measured.min()),
        "measured_max": float(measured.max()),
        "anchors": anchors,
    }


def apply_tone_curve(fit: dict, rgb: np.ndarray) -> np.ndarray:
    """Apply a fitted tone curve identically to every channel of an (..., 3)
    RGB array (0-255 scale), clamping to the calibrated luma range."""
    clamped = np.clip(rgb, fit["measured_min"], fit["measured_max"])
    return np.clip(fit["curve"](clamped), 0, 255)
