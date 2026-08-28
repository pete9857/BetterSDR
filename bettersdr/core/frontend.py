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
from .device import DEFAULT_SAMPLE_RATE, MAX_TUNE_HZ, MIN_TUNE_HZ, Device


@dataclass(frozen=True)
class GainChoice:
    gain_db: float
    level_dbfs: float
    clipped_fraction: float

    @property
    def overloaded(self) -> bool:
        """True when even the lowest gain could not stop the ADC clipping."""
        return self.clipped_fraction > 5e-4


# A probe has to cover a span of *time*, not a count of bytes. 32 KB is 6.8 ms
# at 2.4 MS/s but 137 ms at 240 kS/s, and this reads twice per gain setting
# across up to 29 settings - so a byte count chosen on the FM band turns a
# 0.4 s probe into a four-second freeze of the reader thread on the AM band.
PROBE_SECONDS = 32_768 / 2 / 2_400_000


def probe_bytes_for(sample_rate_hz: float) -> int:
    """Bytes covering `PROBE_SECONDS` of radio, rounded for the USB transfer."""
    raw = int(sample_rate_hz * 2 * PROBE_SECONDS)
    return max(4_096, (raw // 512) * 512)


def choose_gain(
    dev: Device,
    *,
    target_dbfs: float = -12.0,
    max_clip_fraction: float = 5e-4,
    probe_bytes: int | None = None,
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
    if probe_bytes is None:
        probe_bytes = probe_bytes_for(dev.sample_rate)
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


__all__ = ["GainChoice", "choose_gain", "probe_bytes_for"]


# --------------------------------------------------------------------------
# Sample rate
# --------------------------------------------------------------------------
#
# Two constraints intersect, and only a handful of rates satisfy both. The
# RTL2832U accepts 225001-300000 and 900001-3200000 samples per second and
# nothing in between, and `dsp/demod.py` requires a whole multiple of the
# 48 kHz audio rate so no stage ever resamples by an awkward ratio.
SUPPORTED_SAMPLE_RATES = (
    240_000,  # 5 x 48 kHz
    288_000,  # 6
    960_000,  # 20
    1_152_000,  # 24
    1_200_000,  # 25
    1_440_000,  # 30
    1_920_000,  # 40
    2_400_000,  # 50 - the most the RTL2832U sustains without dropping samples
)

# How far above 0 Hz the bottom of the window has to stay, as a fraction of
# the window's own width.
HF_EDGE_FRACTION = 0.1


def safe_sample_rate(
    center_hz: float,
    preferred_hz: int = DEFAULT_SAMPLE_RATE,
) -> int:
    """The widest supported window that stays clear of 0 Hz on the dial.

    On the HF side the V4's SA612 upconverter leaks its local oscillator at
    0 Hz, and the leak is enormous: measured at 710 kHz it sat 65 dB above the
    noise floor, which on an 8-bit ADC is the only thing the front end can see.
    `choose_gain` duly stepped down to the bottom of its range to keep from
    clipping on it, and the wanted station went down with it - KIRO on 710 kHz
    came back 36 dB quieter than it does through a window that never reaches
    zero. Nothing in the audio path can recover that; it has to not happen.

    So a window low enough to contain 0 Hz is not a tuning preference, it is a
    fault, and this is the guard against it wherever a frequency is chosen -
    band defaults, manual tuning, or a sweep. A band that knows it wants
    something narrower still says so itself; this only ever narrows.
    """
    ceiling = center_hz / (0.5 + HF_EDGE_FRACTION)
    allowed = [
        rate
        for rate in SUPPORTED_SAMPLE_RATES
        if rate <= min(preferred_hz, ceiling)
    ]
    # Below about 480 kHz even the narrowest window reaches zero. The dongle
    # tunes down to 500 kHz, so this is reachable, and the narrowest rate is
    # still the least wrong answer available.
    return max(allowed) if allowed else min(SUPPORTED_SAMPLE_RATES)


def safe_center_hz(center_hz: float) -> int:
    """The nearest frequency the dongle can actually be tuned to.

    `Device.center_freq` rejects anything outside 500 kHz - 1.766 GHz, and
    that rejection reaches the hardware from the reader thread, where there is
    no user to show it to. The display already clamps the digit tuner, but
    click-to-tune reads a frequency straight off the spectrum's x-axis, and at
    the bottom of the AM band the window legitimately extends below 500 kHz -
    so a click on the left of the plot asked for a frequency that does not
    exist and took the radio down with it.

    Clamping is the right answer rather than raising: the user pointed at a
    place on a picture, and the nearest reachable frequency is what they meant.
    """
    return int(min(MAX_TUNE_HZ, max(MIN_TUNE_HZ, round(center_hz))))
