"""Progressive disclosure levels.

One app, three levels. The DSP engine is identical at every level - only the
visible controls change - because a beginner who outgrows BetterSDR and leaves
for SDR# is the failure case the whole project exists to avoid. Nothing is
removed at a lower level; it is only quiet until asked for.

Promoting a control is therefore a one-word change at its call site, never a
rewrite.
"""

from __future__ import annotations

from enum import IntEnum


class Level(IntEnum):
    """How much of the radio to show."""

    SIMPLE = 0
    STANDARD = 1
    EXPERT = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def description(self) -> str:
        return {
            Level.SIMPLE: "Just listen. Everything is chosen for you.",
            Level.STANDARD: "Tune by hand, pick the mode, adjust the display.",
            Level.EXPERT: "Every control, including the ones you can break.",
        }[self]


__all__ = ["Level"]
