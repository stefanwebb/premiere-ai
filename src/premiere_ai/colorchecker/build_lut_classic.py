"""Build a 3D .cube LUT from a photo of the CLASSIC (traditional 24-patch)
page of an X-Rite/Calibrite ColorChecker: locate the chart (SAM3), segment
its 24 squares, match each to its published reference sRGB value (Hungarian
assignment over the whole grid, so one ambiguous patch can't silently steal
another's slot), and fit a color-correction matrix mapping measured colors
to those references.

This corrects hue/saturation/white-balance only — no separate exposure/tone
curve — via colour-science's `matrix_colour_correction`: "Cheung 2004"
(default) is a plain 3x3 + offset affine fit; "Finlayson 2015" fits an
exposure-invariant root-polynomial expansion instead, at the cost of extra
fitted terms. Grid orientation is double-checked by finding which single
rigid transform (rotate/flip) is consistent across the most matched
patches — patches that disagree with it are flagged and excluded from the
fit rather than silently degrading it.

Reference values are BabelColor/Danny Pascale's published nominal sRGB
measurements for the Classic chart (see classic_reference.py) — good
enough to verify grid identity/orientation and fit a real correction
against, but not the physical unit's own individually-calibrated values.
"""

import sys
from pathlib import Path

import colour
import numpy as np
from PIL import Image
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model

from premiere_ai.colorchecker.classic_reference import ORIG_SHAPE, REFERENCE_NAMES, REFERENCE_POSITIONS, REFERENCE_RGBS
from premiere_ai.colorchecker.pages import get_page_config
from premiere_ai.colorchecker.lut_io import write_lut_3d
from premiere_ai.colorchecker.patch_grid import assign_grid, best_axis_swap_transform, match_to_reference, mean_patch_color

MODEL_ID = "mlx-community/sam3-4bit"
COLORCHECKER_PROMPT = "a colorchecker color calibration chart"
COLORCHECKER_SCORE_THRESHOLD = 0.3
CROP_PADDING = 20
LUT_3D_SIZE = 17


def measure_classic_patches(predictor, image_path: Path) -> dict[tuple[int, int], tuple[float, float, float]]:
    page = get_page_config("classic")
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    image_width, image_height = image.size

    chart_result = predictor.predict(image, text_prompt=COLORCHECKER_PROMPT)
    if len(chart_result.scores) == 0:
        raise RuntimeError("No ColorChecker found in the image.")

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
    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    grid_coords = assign_grid(centroids)
    return {cell: mean_patch_color(crop_array, box) for cell, box in zip(grid_coords, boxes)}


def correction_kwargs(method: str, terms: int, degree: int) -> dict:
    """Per-method kwargs for colour's colour-correction fitters. Cheung 2004
    is parameterised by `terms` (4 == the [R,G,B,1] affine augmentation, i.e.
    a 3x3 + offset); Finlayson 2015 by polynomial `degree`, and we request
    the root-polynomial expansion — it's exposure-invariant, which a plain
    matrix with an offset is not."""
    if method == "Finlayson 2015":
        return {"degree": degree, "root_polynomial_expansion": True}
    return {"terms": terms}


def fit_color_matrix(measured: np.ndarray, reference: np.ndarray, method: str, kwargs: dict) -> np.ndarray:
    """Fit a colour-correction matrix mapping measured RGB to reference RGB,
    via colour-science. Inputs are 0-255; the fit runs in 0-1 (colour's
    convention, and it keeps any higher-order polynomial terms well-scaled)."""
    return colour.matrix_colour_correction(measured / 255, reference / 255, method=method, **kwargs)


def apply_color_matrix(matrix: np.ndarray, rgb: np.ndarray, method: str, kwargs: dict) -> np.ndarray:
    """Apply a fitted CCM to 0-255 RGB, returning 0-255 RGB."""
    return colour.apply_matrix_colour_correction(rgb / 255, matrix, method=method, **kwargs) * 255


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Photo of the ColorChecker's CLASSIC (24-patch) page")
    parser.add_argument(
        "--matrix-method", choices=["Cheung 2004", "Finlayson 2015"], default="Cheung 2004",
        help="colour-correction fitter. Cheung 2004 with --matrix-terms 4 (default) is the "
             "3x3+offset affine fit; Finlayson 2015 gives an exposure-invariant root-polynomial.",
    )
    parser.add_argument("--matrix-terms", type=int, default=4, help="Cheung 2004 augmentation terms (4 == 3x3 + offset)")
    parser.add_argument("--matrix-degree", type=int, default=2, help="Finlayson 2015 root-polynomial degree")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .cube path (default: <image>.cube next to the input image)",
    )
    parser.add_argument("--lut-size", type=int, default=LUT_3D_SIZE)
    args = parser.parse_args()

    output_path = args.output or args.image.with_suffix(".cube")
    mkwargs = correction_kwargs(args.matrix_method, args.matrix_terms, args.matrix_degree)

    print(f"Loading {MODEL_ID}...")
    model_path = get_model_path(MODEL_ID)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=COLORCHECKER_SCORE_THRESHOLD)

    print("\n--- Measuring classic-page patches ---")
    grid_colors = measure_classic_patches(predictor, args.image)
    assignment, residuals = match_to_reference(grid_colors, REFERENCE_RGBS)
    transform_name, transform_result = best_axis_swap_transform(assignment, REFERENCE_POSITIONS, ORIG_SHAPE)
    print(f"Grid identity check: {transform_result['matches']}/{len(assignment)} patches consistent "
          f"with orientation {transform_name!r}")
    if transform_result["mismatches"]:
        print("WARNING: some patches disagree with the recovered orientation — "
              "excluding them from the color-matrix fit:")
        bad_cells = set()
        for m in transform_result["mismatches"]:
            bad_cells.add(m["cell"])
            print(f"  {m['cell']} -> {REFERENCE_NAMES[m['ref_index']]!r}")
    else:
        bad_cells = set()

    cells = [c for c in assignment if c not in bad_cells]
    measured_rgb = np.array([grid_colors[c] for c in cells], dtype=float)
    reference_rgb = np.array([REFERENCE_RGBS[assignment[c]] for c in cells], dtype=float)

    print("\n--- Fitting color matrix ---")
    matrix = fit_color_matrix(measured_rgb, reference_rgb, args.matrix_method, mkwargs)
    print(f"Fitted CCM ({args.matrix_method}, {mkwargs}), shape {matrix.shape}:")
    for row in np.atleast_2d(matrix):
        print("  " + "  ".join(f"{v:8.4f}" for v in row))

    predicted = apply_color_matrix(matrix, measured_rgb, args.matrix_method, mkwargs)
    raw_rmse = np.sqrt(np.mean((measured_rgb - reference_rgb) ** 2))
    corrected_rmse = np.sqrt(np.mean((predicted - reference_rgb) ** 2))
    print(f"\nPer-patch RGB RMSE: raw={raw_rmse:.2f} -> corrected={corrected_rmse:.2f}")
    for c in cells:
        name = REFERENCE_NAMES[assignment[c]]
        raw = grid_colors[c]
        pred = apply_color_matrix(matrix, np.array(raw, dtype=float), args.matrix_method, mkwargs)
        ref = REFERENCE_RGBS[assignment[c]]
        err = np.linalg.norm(pred - np.array(ref))
        print(f"  {name:15s} measured={tuple(round(v) for v in raw)} -> corrected={tuple(round(v) for v in pred)} "
              f"reference={ref} err={err:.1f}")

    def pipeline(rgb: np.ndarray) -> np.ndarray:
        return apply_color_matrix(matrix, rgb, args.matrix_method, mkwargs)

    print(f"\nWriting {args.lut_size}^3 classic-page color-matrix LUT...")
    write_lut_3d(
        output_path,
        lambda rgb: pipeline(rgb * 255) / 255,
        args.lut_size,
        f"classic-page color matrix ({args.matrix_method})",
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
