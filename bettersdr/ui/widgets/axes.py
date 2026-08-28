"""Axis items shared by the spectrum, the waterfall and the band ribbon.

The three panes are stacked and must line up to the pixel: a waterfall offset
from the spectrum above it makes every frequency read wrong by a few hundred
kilohertz, which is worse than having no waterfall at all. They line up by
reserving the same axis width in all three, so the only pane that actually
draws labels does not shift itself relative to the others.
"""

from __future__ import annotations

import pyqtgraph as pg

# Wide enough for "-100" at the spectrum's font size. Fixed rather than
# measured so the panes cannot drift apart as the y range changes.
AXIS_WIDTH = 52


class FrequencyAxis(pg.AxisItem):
    """An x axis labelled in MHz, because nobody reads nine digits of hertz."""

    def tickStrings(
        self, values: list[float], scale: float, spacing: float
    ) -> list[str]:
        # Decimals follow the tick spacing, so labels neither repeat nor carry
        # digits that are below the resolution of the view.
        decimals = 0 if spacing >= 1e6 else (3 if spacing >= 1e3 else 6)
        return [f"{value / 1e6:.{decimals}f}" for value in values]


class BlankAxis(pg.AxisItem):
    """Reserves the same width as a real axis but draws nothing."""

    def __init__(self, orientation: str = "left") -> None:
        super().__init__(orientation)
        self.setWidth(AXIS_WIDTH)
        self.setStyle(tickLength=0, showValues=False)
        self.setPen(pg.mkPen(None))

    def tickStrings(
        self, values: list[float], scale: float, spacing: float
    ) -> list[str]:
        return ["" for _ in values]


__all__ = ["AXIS_WIDTH", "BlankAxis", "FrequencyAxis"]
