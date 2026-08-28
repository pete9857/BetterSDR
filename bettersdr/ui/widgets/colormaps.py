"""Colour maps for the waterfall.

Defined as control points and interpolated here rather than pulled from
matplotlib, which is not a dependency and would be a heavy one to add for five
gradients. `viridis`, `inferno` and `turbo` are close approximations of the
well-known maps, not bit-exact copies of them.

Pure NumPy on purpose - no Qt import - so the maps can be checked without a
display and reused anywhere a lookup table is wanted.
"""

from __future__ import annotations

import numpy as np

# Each entry is (position 0..1, red, green, blue) with 8-bit components.
_CONTROL_POINTS: dict[str, tuple[tuple[float, int, int, int], ...]] = {
    # The dark-blue-through-white ramp SDR# users will recognise. Black floor
    # matters: it makes an empty band look genuinely empty.
    "classic": (
        (0.00, 0, 0, 0),
        (0.20, 0, 0, 92),
        (0.40, 0, 92, 184),
        (0.55, 0, 194, 194),
        (0.70, 60, 214, 60),
        (0.82, 245, 233, 60),
        (0.92, 235, 70, 40),
        (1.00, 255, 255, 255),
    ),
    "viridis": (
        (0.00, 68, 1, 84),
        (0.25, 59, 82, 139),
        (0.50, 33, 145, 140),
        (0.75, 94, 201, 98),
        (1.00, 253, 231, 37),
    ),
    "inferno": (
        (0.00, 0, 0, 4),
        (0.25, 87, 16, 110),
        (0.50, 188, 55, 84),
        (0.75, 249, 142, 9),
        (1.00, 252, 255, 164),
    ),
    "turbo": (
        (0.00, 48, 18, 59),
        (0.20, 70, 134, 251),
        (0.40, 27, 229, 181),
        (0.60, 164, 252, 60),
        (0.80, 251, 152, 44),
        (1.00, 122, 4, 3),
    ),
    "grayscale": (
        (0.00, 0, 0, 0),
        (1.00, 255, 255, 255),
    ),
}

NAMES = tuple(_CONTROL_POINTS)
DEFAULT_NAME = "classic"


def lookup_table(name: str = DEFAULT_NAME, levels: int = 256) -> np.ndarray:
    """An (levels, 3) uint8 table ramping through the named map."""
    if name not in _CONTROL_POINTS:
        raise ValueError(f"unknown colour map {name!r}; expected one of {NAMES}")
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")

    points = np.array(_CONTROL_POINTS[name], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, levels)
    table = np.empty((levels, 3), dtype=np.float64)
    for channel in range(3):
        table[:, channel] = np.interp(positions, points[:, 0], points[:, channel + 1])
    return np.clip(np.round(table), 0, 255).astype(np.uint8)


def control_points(name: str = DEFAULT_NAME) -> tuple[tuple[float, int, int, int], ...]:
    """The raw stops, for handing to a toolkit that interpolates its own."""
    if name not in _CONTROL_POINTS:
        raise ValueError(f"unknown colour map {name!r}; expected one of {NAMES}")
    return _CONTROL_POINTS[name]


__all__ = ["DEFAULT_NAME", "NAMES", "control_points", "lookup_table"]
