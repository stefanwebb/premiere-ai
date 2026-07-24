"""Write fitted corrections out as .cube LUTs, via colour-science.

The .cube format requires the RED index to vary FASTEST, which reads
backwards from the `for b: for g: for r:` nesting that produces it — so
every hand-rolled writer here was one transposed loop away from silently
emitting a channel-swapped LUT that still loads fine in Premiere. colour's
`LUT3D` indexes its table `[i_R, i_G, i_B]` and `write_LUT_ResolveCube`
owns the axis ordering, so a fit only ever has to describe a 0-1 -> 0-1
mapping and never touches file layout.

Equivalence with the previous hand-rolled writer was confirmed by reading
an existing 17^3 output back: `table[i_R, i_G, i_B] == manual[i_B, i_G, i_R]`,
and a colour write/read round-trip differs by 0.0.

One cosmetic difference: colour omits DOMAIN_MIN/DOMAIN_MAX when they are
the 0-1 default (the spec's default), where the old writer always spelled
them out. Same LUT.
"""

from os import PathLike
from typing import Callable, Sequence

import numpy as np
from colour import LUT1D, LUT3D
from colour.io import write_LUT_ResolveCube

DECIMALS = 6

Mapping = Callable[[np.ndarray], np.ndarray]


def write_lut_3d(
    path: str | PathLike,
    mapping: Mapping,
    size: int,
    title: str,
    decimals: int = DECIMALS,
) -> None:
    """Sample `mapping` over a `size`^3 lattice and write it as a 3D .cube.

    `mapping` takes an (N, 3) array of 0-1 RGB and returns the corrected
    0-1 RGB. Fits working in 0-255 should wrap themselves at the call site
    (`lambda rgb: pipeline(rgb * 255) / 255`) rather than have this guess
    at a domain.

    Output is clipped to 0-1, which every fit here wants: these correct a
    display-referred Rec.709 frame, so out-of-range values are fit
    extrapolation artifacts rather than headroom worth carrying.
    """
    lattice = LUT3D.linear_table(size)
    corrected = np.asarray(mapping(lattice.reshape(-1, 3)), dtype=float)
    table = np.clip(corrected, 0.0, 1.0).reshape(lattice.shape)

    write_LUT_ResolveCube(LUT3D(table, title), path, decimals=decimals)


def write_lut_1d(
    path: str | PathLike,
    values: Sequence[float] | np.ndarray,
    title: str,
    decimals: int = DECIMALS,
) -> None:
    """Write a tone curve, sampled evenly over 0-1, as a 1D .cube.

    `values` are the 0-1 outputs; the writer emits them as the same value
    on all three channels, matching a curve fit that is by construction
    identical on R/G/B.
    """
    table = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)

    write_LUT_ResolveCube(LUT1D(table, title), path, decimals=decimals)
