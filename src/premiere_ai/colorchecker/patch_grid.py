"""Recover a regular (row, col) grid layout from an unordered set of
detected patch boxes, plus colorimetric self-consistency checks (neutral
ramp monotonicity, and full-grid identity matching against a reference set)
that confirm grid orientation without needing manufacturer Lab data."""

import colorsys

import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_1d(values: list[float], gap_fraction: float = 0.3) -> list[int]:
    """Assign each value a cluster index (0-based, in ascending order of
    cluster position) by sorting and splitting wherever the gap between
    consecutive values exceeds `gap_fraction` of the largest gap.

    A manufactured chart's grid has a clean bimodal gap distribution: small
    jitter within a row/column of patches, and a much larger, roughly
    constant pitch between rows/columns — so thresholding relative to the
    largest observed gap separates the two cleanly. (A median-based
    threshold doesn't work here: with more within-cluster gaps than
    between-cluster ones, the median sits inside the jitter band itself,
    and jitter can exceed a small multiple of the median.)
    """
    order = np.argsort(values)
    sorted_values = np.array(values)[order]
    gaps = np.diff(sorted_values)
    max_gap = np.max(gaps) if len(gaps) else 0.0
    threshold = max_gap * gap_fraction if max_gap > 0 else np.inf

    cluster_of_sorted = np.zeros(len(values), dtype=int)
    current = 0
    for i, gap in enumerate(gaps, start=1):
        if gap > threshold:
            current += 1
        cluster_of_sorted[i] = current

    labels = np.empty(len(values), dtype=int)
    labels[order] = cluster_of_sorted
    return labels.tolist()


def deduplicate_boxes(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression: return the indices to KEEP, dropping any box
    that overlaps a higher-scoring one by more than `iou_threshold`.

    SAM3 can return the same patch twice (observed: one patch detected at
    both 0.85 and 0.43 with boxes overlapping ~99%), which breaks grid
    recovery — 25 "patches" for a 6x4 grid. Score thresholding alone can't
    fix that without also discarding legitimately dim patches, so overlap
    is the right discriminator.
    """
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        ax1, ay1, ax2, ay2 = boxes[i]
        area_i = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        duplicate = False
        for j in keep:
            bx1, by1, bx2, by2 = boxes[j]
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            intersection = inter_w * inter_h
            if intersection <= 0:
                continue
            area_j = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_i + area_j - intersection
            if union > 0 and intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return sorted(keep)


def assign_grid(
    centroids: list[tuple[float, float]], require_complete: bool = True
) -> list[tuple[int, int]]:
    """Map (x, y) centroids to (row, col) grid coordinates, row 0 = top,
    col 0 = left.

    `require_complete=False` tolerates cells with no detected patch, for
    callers that only need particular columns. A missing patch is common
    when measuring a chart that is small in frame (e.g. a chart inside a
    rendered 4K sequence frame rather than a full-resolution photo);
    demanding all 24 there throws away an otherwise perfectly good
    measurement. Duplicate cells are always an error — those mean the
    clustering is genuinely ambiguous, not merely incomplete.
    """
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    cols = cluster_1d(xs)
    rows = cluster_1d(ys)

    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    if require_complete and n_rows * n_cols != len(centroids):
        raise ValueError(
            f"Expected a clean {n_rows}x{n_cols} grid ({n_rows * n_cols} cells) "
            f"but got {len(centroids)} patches — clustering did not recover a "
            f"rectangular grid."
        )

    seen = set()
    for r, c in zip(rows, cols):
        if (r, c) in seen:
            raise ValueError(f"Duplicate grid cell ({r}, {c}) — clustering is ambiguous.")
        seen.add((r, c))

    return list(zip(rows, cols))


def mean_patch_color(image: np.ndarray, box: tuple[float, float, float, float], center_fraction: float = 0.5) -> tuple[int, int, int]:
    """Mean RGB over the central `center_fraction` of a patch's bounding
    box, sidestepping the need for per-patch masks/erosion — sampling well
    inside the box avoids edge bleed from anti-aliasing and the mask's own
    boundary noise."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    margin_x = w * (1 - center_fraction) / 2
    margin_y = h * (1 - center_fraction) / 2
    cx1, cy1 = round(x1 + margin_x), round(y1 + margin_y)
    cx2, cy2 = round(x2 - margin_x), round(y2 - margin_y)
    region = image[cy1:cy2, cx1:cx2]
    mean = region.reshape(-1, 3).mean(axis=0)
    return tuple(round(v) for v in mean)


def verify_neutral_ramp(grid_colors: dict[tuple[int, int], tuple[int, int, int]]) -> dict:
    """Sanity-check the recovered grid orientation without reference data:
    the row (or column) with the lowest average saturation should be the
    neutral grayscale ramp, and its luminance should be monotonic along that
    line. This confirms row/col direction is being read consistently with
    the physical chart, independent of knowing patch identities.
    """
    n_rows = max(r for r, c in grid_colors) + 1
    n_cols = max(c for r, c in grid_colors) + 1

    def saturation(rgb):
        r, g, b = [v / 255 for v in rgb]
        return colorsys.rgb_to_hsv(r, g, b)[1]

    def luminance(rgb):
        r, g, b = [v / 255 for v in rgb]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    row_sat = [np.mean([saturation(grid_colors[(r, c)]) for c in range(n_cols)]) for r in range(n_rows)]
    col_sat = [np.mean([saturation(grid_colors[(r, c)]) for r in range(n_rows)]) for c in range(n_cols)]

    best_row = int(np.argmin(row_sat))
    best_col = int(np.argmin(col_sat))
    row_is_ramp = row_sat[best_row] <= col_sat[best_col]

    if row_is_ramp:
        line = [grid_colors[(best_row, c)] for c in range(n_cols)]
        axis, index = "row", best_row
    else:
        line = [grid_colors[(r, best_col)] for r in range(n_rows)]
        axis, index = "col", best_col

    luminances = [luminance(rgb) for rgb in line]
    increasing = all(a <= b for a, b in zip(luminances, luminances[1:]))
    decreasing = all(a >= b for a, b in zip(luminances, luminances[1:]))

    return {
        "axis": axis,
        "index": index,
        "luminances": [round(v, 3) for v in luminances],
        "monotonic": increasing or decreasing,
        "direction": "increasing" if increasing else ("decreasing" if decreasing else "non-monotonic"),
    }


def _saturation(rgb: tuple[int, int, int]) -> float:
    r, g, b = [v / 255 for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)[1]


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = [v / 255 for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hue_degrees(rgb: tuple[int, int, int]) -> float:
    r, g, b = [v / 255 for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def column_profiles(grid_colors: dict[tuple[int, int], tuple[int, int, int]]) -> list[dict]:
    """Per-column saturation/luminance summary, for charts where more than
    one column may be a neutral-ish ramp (e.g. the video page's "grey
    balance" and "highlights and shadows" columns) — unlike
    `verify_neutral_ramp`, this doesn't assume there's exactly one such
    line, it just reports every column so a human (or a page-specific
    caller who knows how many to expect) can judge them.
    """
    n_rows = max(r for r, c in grid_colors) + 1
    n_cols = max(c for r, c in grid_colors) + 1

    profiles = []
    for col in range(n_cols):
        line = [grid_colors[(r, col)] for r in range(n_rows)]
        luminances = [_luminance(rgb) for rgb in line]
        increasing = all(a <= b for a, b in zip(luminances, luminances[1:]))
        decreasing = all(a >= b for a, b in zip(luminances, luminances[1:]))
        profiles.append(
            {
                "col": col,
                "colors": line,
                "mean_saturation": float(np.mean([_saturation(rgb) for rgb in line])),
                "luminances": [round(v, 3) for v in luminances],
                "monotonic": increasing or decreasing,
                "direction": "increasing" if increasing else ("decreasing" if decreasing else "non-monotonic"),
            }
        )
    return profiles


def verify_hue_progression(
    colors: list[tuple[int, int, int]], expected_step: float = 60.0, tolerance: float = 25.0
) -> dict:
    """Check whether a line of patches steps around the hue circle by a
    consistent amount in a consistent direction (e.g. the video page's
    chromatic column: green -> cyan -> blue -> magenta -> red -> yellow is
    six ~60-degree steps around the wheel). This is a strong, reference-free
    orientation signal for a chromatic ramp, the counterpart to
    `verify_neutral_ramp`/`column_profiles` for neutral ones.
    """

    def circular_distance(a: float, b: float) -> float:
        d = abs(a - b) % 360
        return min(d, 360 - d)

    hues = [_hue_degrees(rgb) for rgb in colors]
    forward_diffs = [(b - a) % 360 for a, b in zip(hues, hues[1:])]
    backward_diffs = [(360 - d) % 360 for d in forward_diffs]

    forward_consistent = all(circular_distance(d, expected_step) <= tolerance for d in forward_diffs)
    backward_consistent = all(circular_distance(d, expected_step) <= tolerance for d in backward_diffs)

    return {
        "hues": [round(h, 1) for h in hues],
        "forward_diffs": [round(d, 1) for d in forward_diffs],
        "forward_consistent": forward_consistent,
        "backward_consistent": backward_consistent,
        "consistent": forward_consistent or backward_consistent,
        "direction": "forward" if forward_consistent else ("backward" if backward_consistent else "inconsistent"),
    }


def match_to_reference(
    grid_colors: dict[tuple[int, int], tuple[int, int, int]],
    reference_rgbs: list[tuple[int, int, int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], float]]:
    """Globally match every grid cell to a reference patch by color alone
    (Hungarian algorithm, minimizing total squared RGB distance) — a 1-1
    assignment across the whole grid, not per-cell nearest-neighbor, so a
    single ambiguous patch can't silently steal another's slot.

    Returns (assignment, residuals): assignment maps each grid cell to a
    reference index (position in `reference_rgbs`); residuals gives that
    match's RGB distance, for flagging poor matches.
    """
    cells = list(grid_colors.keys())
    measured = np.array([grid_colors[c] for c in cells], dtype=float)
    reference = np.array(reference_rgbs, dtype=float)
    cost = np.sum((measured[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)

    assignment = {}
    residuals = {}
    for ri, ci in zip(row_idx, col_idx):
        cell = cells[ri]
        assignment[cell] = int(ci)
        residuals[cell] = float(np.sqrt(cost[ri, ci]))
    return assignment, residuals


def best_axis_swap_transform(
    assignment: dict[tuple[int, int], int],
    ref_positions: list[tuple[int, int]],
    ref_shape: tuple[int, int],
) -> tuple[str, dict]:
    """Given a color-based cell -> reference-index assignment, find which
    single rigid transform (of the 4 that map an (R, C) reference layout
    onto its (C, R) transpose — i.e. a 90-degree-rotated physical card) is
    consistent across the most patches. A correct grid recovery + color
    match should agree with one transform on every patch; patches that
    don't are flagged as outliers worth a manual look (bad color match, or
    a genuine grid-recovery error).
    """
    n_rows, n_cols = ref_shape
    transforms = {
        "rotate-ccw": lambda r, c: (c, r),
        "rotate-ccw+flip-rows": lambda r, c: (c, n_rows - 1 - r),
        "rotate-ccw+flip-cols": lambda r, c: (n_cols - 1 - c, r),
        "rotate-ccw+flip-both": lambda r, c: (n_cols - 1 - c, n_rows - 1 - r),
    }

    results = {}
    for name, transform in transforms.items():
        mismatches = []
        matches = 0
        for cell, ref_idx in assignment.items():
            predicted = transform(*ref_positions[ref_idx])
            if predicted == cell:
                matches += 1
            else:
                mismatches.append({"cell": cell, "ref_index": ref_idx, "predicted_cell": predicted})
        results[name] = {"matches": matches, "mismatches": mismatches}

    best_name = max(results, key=lambda k: results[k]["matches"])
    return best_name, results[best_name]
