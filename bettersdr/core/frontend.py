"""Front-end (tuner) setup shared by the listener, the GUI and the scanner.

Gain selection lives here rather than in any one caller because all three need
the same answer, and an inconsistent one would be user-visible: the scanner
measuring the band at a different gain from the display means "the app found a
signal I cannot see".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp import convert
from .device import Device


@dataclass(frozen=True)
class GainChoice:
    gain_db: float
    level_dbfs: float
    clipped_fraction: float

    @property
    def overloaded(self) -> bool:
        """True when even the lowest gain could not stop the ADC clipping."""
        return self.clipped_fraction > 5e-4


def choose_gain(
    dev: Device,
    *,
    target_dbfs: float = -12.0,
    max_clip_fraction: float = 5e-4,
    probe_bytes: int = 32_768,
) -> GainChoice:
    """Pick the highest gain setting that does not overload the ADC.

    An overloaded 8-bit front end sounds like distortion, and a beginner has no
    reason to suspect the gain control - they will assume the dongle is faulty.
    The FM band is the common case: a capture at 100 MHz on maximum gain had
    1.64% of samples pinned at the rails.

    So we step down from maximum and take the first setting where the signal
    sits below `target_dbfs` and essentially nothing is clipping. Must be
    called with the device idle - it reads directly, so run it before the
    reader thread starts or from the reader thread itself.
    """
    dev.set_manual_gain(True)
    measured = GainChoice(min(dev.gains_db), 0.0, 1.0)
    for gain in sorted(dev.gains_db, reverse=True):
        dev.gain_db = gain
        dev.reset_buffer()
        dev.read(probe_bytes)  # discard: the tuner is still settling
        raw = dev.read(probe_bytes)
        clipped = float(np.count_nonzero((raw == 0) | (raw == 255))) / raw.size
        level = convert.dbfs(convert.rms(convert.to_complex(raw)))
        measured = GainChoice(gain, level, clipped)
        if clipped <= max_clip_fraction and level <= target_dbfs:
            return measured
    # Everything overloads: the lowest gain is the best available answer.
    return measured


__all__ = ["GainChoice", "choose_gain"]
