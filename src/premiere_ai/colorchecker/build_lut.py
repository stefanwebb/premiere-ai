"""Build a 3D .cube LUT from JUST the video page of the ColorChecker
Passport Video 2, correcting:

  - exposure + white/grey balance: independent per-channel (R, G, B) tone
    curves fit from the neutral patches (left-panel 3-step strip + grid
    columns 2 and 3) against IRE targets — since these patches are
    "spectrally neutral," each channel should independently land on the
    SAME target code; fitting per-channel (not one shared luma curve)
    corrects color cast simultaneously with tone.
  - chromatic colors (col 0): a single global hue-rotation (not a full
    matrix — no Lab/RGB reference exists for these chips) aligning the six
    green/cyan/blue/magenta/red/yellow chips to the Rec.709 vectorscope
    target hues, computed analytically (see vectorscope.py).
  - skin tones (col 1): folded into the SAME global hue rotation, targeting
    the classic "-I axis" skin-tone line (see vectorscope.py for how that
    target is derived, and its caveats — it's a reasoned construction, not
    a confirmed Adobe figure, so verify visually against Premiere's own
    skin-tone-line toggle).
  - highlights/shadows (col 3): folded into the per-channel tone-curve fit
    as extra neutral anchors, at IRE percentages given via
    --highlight-shadow-ire (default is a guess at plausible highlight/
    shadow rolloff test points — Calibrite doesn't publish exact values
    for this column, so override this if you obtain real ones).

This all comes from ONE page / one shot — deliberately not combined with
the classic page's 24-patch color match, which would require both pages to
be photographed together under identical conditions to be physically
meaningful (see CHANGELOG / conversation history for the rejected
combined-page attempt).

CAVEATS: per-channel WB/exposure uses real IRE targets (left-panel + grid
col2). Chromatic hue targets are exact (computed from Rec.709). Skin-tone
target and highlight/shadow IRE values are reasoned/parameterized
estimates, not vendor-confirmed.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker.pages import get_page_config, video_left_step_masks
from premiere_ai.colorchecker.tone_curve import mean_mask_color
from premiere_ai.colorchecker.lut_io import write_lut_3d
from premiere_ai.colorchecker.patch_grid import assign_grid, mean_patch_color, verify_hue_progression
from premiere_ai.colorchecker.vectorscope import (
    PRIMARY_TARGET_HUES,
    angular_diff_deg,
    circular_mean_deg,
    hue_angle_deg,
    rgb_to_ycbcr,
    rotate_hue,
    skin_tone_target_hue,
)

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20
LUT_3D_SIZE = 17

LEFT_STEP_TARGETS_IRE = {"white": 95.0, "grey": 40.0, "black": 5.0}
RAMP_IRE_RANGE = (90.0, 20.0)
DEFAULT_HIGHLIGHT_SHADOW_IRE = (100.0, 97.0, 94.0, 6.0, 3.0, 0.0)
CHROMATIC_ORDER = ["green", "cyan", "blue", "magenta", "red", "yellow"]


def fit_channel_curve(anchors: list[tuple[float, float, str]]) -> dict | None:
    """anchors: (measured_value, target_code, label). Same monotonic-safe
    fit as tone_curve.fit_tone_curve, generalized to one channel at a time."""
    if len(anchors) < 2:
        return None
    sorted_anchors = sorted(anchors, key=lambda a: a[0])
    measured = np.array([a[0] for a in sorted_anchors])
    target = np.array([a[1] for a in sorted_anchors])

    if not np.all(np.diff(measured) > 0):
        # average duplicate/near-duplicate x's target rather than fail
        unique_measured, inverse = np.unique(measured, return_inverse=True)
        target = np.array([target[inverse == i].mean() for i in range(len(unique_measured))])
        measured = unique_measured

    adjusted = IsotonicRegression(increasing=True).fit_transform(measured, target)

    # A curve that interpolates every anchor exactly (PCHIP) is only as
    # smooth as the anchors are consistent — with ~15 closely-spaced,
    # independently-measured neutral patches (several bunched near black:
    # the left-panel black step, 3 highlight/shadow rows, and the ramp's
    # darkest row), small per-anchor measurement noise forces locally steep
    # segments into the curve. A steep segment stretches a narrow input
    # range across a wide output range, which amplifies whatever sensor
    # noise/banding was already in that tonal region — confirmed visually
    # as blotchy shadow noise on a real captured frame. A smooth degree-2
    # (parabolic) least-squares fit trades exact anchor agreement for a
    # bounded, gentle slope everywhere, which is the actual fix for that
    # artifact. Falls back to a line (degree 1) if the parabola isn't
    # monotonic across the full code range.
    coeffs = np.polyfit(measured, adjusted, deg=2)
    domain = np.linspace(0, 255, 256)
    derivative = np.polyder(coeffs)
    if np.any(np.polyval(derivative, domain) <= 0):
        coeffs = np.polyfit(measured, adjusted, deg=1)

    curve = np.poly1d(coeffs)
    return {"curve": curve, "measured_min": float(measured.min()), "measured_max": float(measured.max())}


def apply_channel_curve(fit: dict, values: np.ndarray) -> np.ndarray:
    return np.clip(fit["curve"](values), 0, 255)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Photo of the ColorChecker Passport Video 2's VIDEO page")
    parser.add_argument(
        "--highlight-shadow-ire",
        type=str,
        default=",".join(str(v) for v in DEFAULT_HIGHLIGHT_SHADOW_IRE),
        help="6 comma-separated IRE percentages for the highlight/shadow column, "
        "brightest row first (default is a placeholder guess — override with real "
        "values if you obtain them).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .cube path (default: <image>.cube next to the input image)",
    )
    parser.add_argument("--lut-size", type=int, default=LUT_3D_SIZE)
    args = parser.parse_args()

    output_path = args.output or args.image.with_suffix(".cube")

    hs_targets = [float(v) for v in args.highlight_shadow_ire.split(",")]
    if len(hs_targets) != 6:
        raise ValueError(f"--highlight-shadow-ire needs exactly 6 values, got {len(hs_targets)}")

    page = get_page_config("video")
    print(f"Loading {MODEL_ID}...")
    model_path = get_model_path(MODEL_ID)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=COLORCHECKER_SCORE_THRESHOLD)

    with Image.open(args.image) as img:
        image = img.convert("RGB")
    image_width, image_height = image.size

    chart_result = predictor.predict(image, text_prompt=COLORCHECKER_PROMPT)
    if len(chart_result.scores) == 0:
        print("No ColorChecker found in the image.")
        return
    best = int(np.argmax(chart_result.scores))
    x1, y1, x2, y2 = [int(v) for v in chart_result.boxes[best]]
    x1, y1 = max(0, x1 - CROP_PADDING), max(0, y1 - CROP_PADDING)
    x2, y2 = min(image_width, x2 + CROP_PADDING), min(image_height, y2 + CROP_PADDING)
    crop = image.crop((x1, y1, x2, y2))
    crop_array = np.array(crop).astype(float)
    crop_width, crop_height = crop.size
    print(f"ColorChecker box: ({x1}, {y1}, {x2}, {y2}), score={chart_result.scores[best]:.2f}")

    # ---- Neutral anchors: left-panel 3-step strip ----
    channel_anchors = {"R": [], "G": [], "B": []}

    print("\nLeft-panel 3-step strip:")
    step_masks = video_left_step_masks(predictor, crop, crop_width, crop_height)
    for name, target_ire in LEFT_STEP_TARGETS_IRE.items():
        mean_rgb = mean_mask_color(crop_array, step_masks[name])
        if mean_rgb is None:
            print(f"  {name}: no mask found, skipping")
            continue
        target_code = target_ire / 100 * 255
        print(f"  {name}: measured RGB={tuple(round(v, 1) for v in mean_rgb)} -> target {target_ire}% IRE = code {target_code:.1f}")
        for ch, val in zip("RGB", mean_rgb):
            channel_anchors[ch].append((val, target_code, f"left:{name}:{ch}"))

    # ---- Grid squares + column identification ----
    square_result = predictor.predict(crop, text_prompt=page.square_prompt, score_threshold=page.square_score_threshold)
    all_boxes = [tuple(float(v) for v in box) for box in square_result.boxes]
    boxes = [b for b in all_boxes if 0.6 <= (b[2] - b[0]) / (b[3] - b[1]) <= 1.6]
    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    grid_coords = assign_grid(centroids)
    grid_colors = {cell: mean_patch_color(np.array(crop), box) for cell, box in zip(grid_coords, boxes)}
    n_rows = max(r for r, c in grid_colors) + 1

    # Fixed physical layout (confirmed against the actual chart), not
    # inferred from saturation — a saturation-based heuristic misidentifies
    # columns under adverse lighting (confirmed: under this frame's blue
    # cast, desaturated skin patches got misclassified as the grey ramp).
    chromatic_col, skin_col, ramp_col, hs_col = 0, 1, 2, 3
    print(f"\nColumn roles (fixed layout): chromatic={chromatic_col} skin={skin_col} grey_ramp={ramp_col} highlight_shadow={hs_col}")

    # ---- Grey-balance ramp column -> neutral anchors ----
    ramp_line = [grid_colors[(r, ramp_col)] for r in range(n_rows)]
    ramp_lumas = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in ramp_line]
    ramp_bright_first = RAMP_IRE_RANGE if ramp_lumas[0] >= ramp_lumas[-1] else RAMP_IRE_RANGE[::-1]
    ramp_targets_ire = np.linspace(ramp_bright_first[0], ramp_bright_first[1], n_rows)
    print(f"\nGrey-balance ramp column {ramp_col}:")
    for row, (rgb, target_ire) in enumerate(zip(ramp_line, ramp_targets_ire)):
        target_code = target_ire / 100 * 255
        print(f"  row {row}: measured RGB={rgb} -> target {target_ire:.1f}% IRE = code {target_code:.1f}")
        for ch, val in zip("RGB", rgb):
            channel_anchors[ch].append((val, target_code, f"ramp:row{row}:{ch}"))

    # ---- Highlight/shadow column -> neutral anchors (user-parameterized targets) ----
    hs_line = [grid_colors[(r, hs_col)] for r in range(n_rows)]
    hs_lumas = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in hs_line]
    hs_bright_first = hs_targets if hs_lumas[0] >= hs_lumas[-1] else hs_targets[::-1]
    print(f"\nHighlight/shadow column {hs_col} (targets from --highlight-shadow-ire):")
    for row, (rgb, target_ire) in enumerate(zip(hs_line, hs_bright_first)):
        target_code = target_ire / 100 * 255
        print(f"  row {row}: measured RGB={rgb} -> target {target_ire:.1f}% IRE = code {target_code:.1f}")
        for ch, val in zip("RGB", rgb):
            channel_anchors[ch].append((val, target_code, f"hs:row{row}:{ch}"))

    print("\nFitting per-channel tone/WB curves...")
    channel_fits = {ch: fit_channel_curve(channel_anchors[ch]) for ch in "RGB"}
    for ch in "RGB":
        if channel_fits[ch] is None:
            print(f"  {ch}: not enough anchors, aborting.")
            return

    def apply_wb_exposure(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=float)
        out = np.empty_like(rgb)
        for i, ch in enumerate("RGB"):
            out[..., i] = apply_channel_curve(channel_fits[ch], rgb[..., i])
        return out

    # ---- Chromatic column -> hue-rotation anchors ----
    chromatic_line = [grid_colors[(r, chromatic_col)] for r in range(n_rows)]
    hue_check = verify_hue_progression(chromatic_line)
    names = CHROMATIC_ORDER if hue_check["direction"] != "backward" else list(reversed(CHROMATIC_ORDER))
    if hue_check["direction"] == "inconsistent":
        print("\nWARNING: chromatic column hue steps aren't consistent — defaulting to forward order, verify by eye.")

    print(f"\nChromatic column {chromatic_col} (hue direction: {hue_check['direction']}):")
    rotation_offsets = []
    for row, (rgb, name) in enumerate(zip(chromatic_line, names)):
        corrected = apply_wb_exposure(np.array(rgb)) / 255
        cb, cr = rgb_to_ycbcr(corrected)[1:]
        measured_hue = hue_angle_deg(cb, cr)
        target_hue = PRIMARY_TARGET_HUES[name]
        offset = angular_diff_deg(target_hue, measured_hue)
        print(f"  row {row} ({name}): measured_hue={measured_hue:.1f} target_hue={target_hue:.1f} needed_rotation={offset:+.1f}")
        rotation_offsets.append(offset)

    # ---- Skin-tone column -> hue-rotation anchors (same target for all 6) ----
    skin_target = skin_tone_target_hue()
    skin_line = [grid_colors[(r, skin_col)] for r in range(n_rows)]
    print(f"\nSkin-tone column {skin_col} (target hue={skin_target:.1f}, see vectorscope.py caveats):")
    for row, rgb in enumerate(skin_line):
        corrected = apply_wb_exposure(np.array(rgb)) / 255
        cb, cr = rgb_to_ycbcr(corrected)[1:]
        measured_hue = hue_angle_deg(cb, cr)
        offset = angular_diff_deg(skin_target, measured_hue)
        print(f"  row {row}: measured_hue={measured_hue:.1f} needed_rotation={offset:+.1f}")
        rotation_offsets.append(offset)

    theta = circular_mean_deg(rotation_offsets)
    print(f"\nGlobal hue rotation (circular mean of {len(rotation_offsets)} chromatic+skin offsets): {theta:.1f} degrees")

    print("\nPost-rotation residuals:")
    for row, (rgb, name) in enumerate(zip(chromatic_line, names)):
        corrected = apply_wb_exposure(np.array(rgb)) / 255
        rotated = rotate_hue(corrected, theta)
        cb, cr = rgb_to_ycbcr(rotated)[1:]
        residual = angular_diff_deg(PRIMARY_TARGET_HUES[name], hue_angle_deg(cb, cr))
        print(f"  chromatic row {row} ({name}): residual={residual:+.1f}")
    for row, rgb in enumerate(skin_line):
        corrected = apply_wb_exposure(np.array(rgb)) / 255
        rotated = rotate_hue(corrected, theta)
        cb, cr = rgb_to_ycbcr(rotated)[1:]
        residual = angular_diff_deg(skin_target, hue_angle_deg(cb, cr))
        print(f"  skin row {row}: residual={residual:+.1f}")

    def pipeline(rgb_255: np.ndarray) -> np.ndarray:
        wb = apply_wb_exposure(rgb_255) / 255
        rotated = rotate_hue(wb, theta)
        return rotated * 255

    print(f"\nWriting {args.lut_size}^3 video-only LUT...")
    write_lut_3d(
        output_path,
        lambda rgb: pipeline(rgb * 255) / 255,
        args.lut_size,
        "video-page-only: per-channel WB/exposure + hue rotation",
    )
    print(f"Saved to {output_path}")

    print(
        "\nCaveats: per-channel WB/exposure uses real IRE targets (left-panel + grid col2). "
        "Chromatic hue targets are exact (computed from Rec.709). Skin-tone target and "
        "highlight/shadow IRE values are reasoned/parameterized estimates, not vendor-confirmed "
        "— see module docstring."
    )


if __name__ == "__main__":
    sys.exit(main())
