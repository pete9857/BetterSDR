"""PPM calibration against a known carrier.

Every RTL-SDR is built around a cheap crystal, and cheap crystals are a few
parts per million out. At 100 MHz a few ppm is a few hundred hertz, which
nobody notices on wideband FM. On single sideband it makes every voice sound
like a duck, and on any narrow digital mode it is the difference between
decoding and not.

The fix is one number, and the honest way to find it is to point the radio at
something whose frequency is known to be exact - a broadcast transmitter, which
is held to a far tighter tolerance than anything in the dongle - and measure
how far off it lands.

Three details make the measurement useful rather than approximate:

* **Interpolate the peak.** A 4096-point FFT of a 240 kHz window has 59 Hz
  bins, and reporting the centre of the largest bin would quantise the answer
  to 0.6 ppm at 100 MHz - the same size as the error being measured. Fitting a
  parabola through the peak and its two neighbours recovers the true position
  to a small fraction of a bin.
* **Search a window, not the whole span.** The strongest thing in a 2.4 MHz
  slice of the FM band is very often the station next door. The caller says
  how far the carrier could plausibly have drifted and nothing outside that is
  considered.
* **Measure several times and check they agree.** This is the one that turns
  the feature from misleading into useful, and it was not obvious until it was
  measured on air. **A wideband FM broadcast station is a terrible reference**
  - its energy is spread across 150 kHz by the modulation and the strongest
  bin wanders with the programme. Measured on this machine at 94.9 MHz, six
  consecutive readings had a standard deviation of **1814 Hz**, which is 19
  ppm of random number. The same six readings on NOAA weather radio at
  162.55 MHz, which transmits a genuinely steady carrier, spread by **11.5 Hz**
  - 0.07 ppm. So the assistant measures in segments and refuses to report an
  answer whose segments disagree, rather than handing back a confident figure
  derived from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Wide enough for any crystal that is not actually broken - 200 ppm at
# 100 MHz is 20 kHz - and narrow enough to exclude the adjacent channel.
DEFAULT_SEARCH_HZ = 20_000.0
DEFAULT_FFT_SIZE = 16_384

# How far the peak has to stand above the noise floor *in the transform* to
# be a carrier rather than the tallest noise bin. Measured on this machine:
# pure noise peaks 4.0-4.8 dB above the median, and a carrier 30 dB below the
# noise in the time domain still reaches 8.7 dB here and measures exactly - so
# 10 dB rejects noise with margin while accepting almost any real transmitter.
MIN_PEAK_DB = 10.0

# How far the segments of one capture may disagree, as a fraction of the
# carrier, before the answer is worthless. One ppm is 95 Hz at 94.9 MHz and
# 163 Hz at 162.55 MHz; the measured figures either side of it are 19 ppm for
# a wideband FM station and 0.07 ppm for a weather-radio carrier, so almost
# anything in between separates them.
MAX_SPREAD_PPM = 1.0

# The capture is split into this many, each measured on its own. Enough for a
# spread to mean something, few enough that each still holds a whole
# transform.
SEGMENTS = 4


@dataclass(frozen=True)
class Calibration:
    """The result of one measurement, including whether to believe it."""

    offset_hz: float
    ppm: int
    exact_ppm: float
    # How far the carrier stands above the noise floor of the transform, which
    # includes ~39 dB of processing gain and is therefore a much larger number
    # than the signal-to-noise ratio of the same capture in the time domain.
    peak_db: float
    # How much the segments of the capture disagreed. Large means the
    # reference is modulated rather than steady, not that the receiver is bad.
    spread_hz: float
    trustworthy: bool
    steady: bool

    @property
    def summary(self) -> str:
        # Weakness first. Bare noise fails both checks, and "there is nothing
        # here to measure" is the more actionable of the two messages; a
        # strong reference that wanders still reaches the second branch.
        if not self.trustworthy:
            return (
                "The carrier was too weak to measure. Tune to a strong local "
                "station and try again."
            )
        if not self.steady:
            return (
                "This station's frequency moves around too much to measure "
                f"against - readings varied by {self.spread_hz:.0f} Hz. Use "
                "something with a steady carrier instead: a weather radio "
                "station, or an AM broadcast station on the medium wave band."
            )
        return (
            f"The carrier landed {self.offset_hz:+.0f} Hz from where it should "
            f"be, which is {self.exact_ppm:+.2f} ppm."
        )


def measure_offset_hz(
    iq: np.ndarray,
    sample_rate: float,
    search_hz: float = DEFAULT_SEARCH_HZ,
    fft_size: int = DEFAULT_FFT_SIZE,
) -> tuple[float, float]:
    """Where the strongest carrier near centre actually is, and its SNR.

    Returns the offset from the middle of the window in hertz - positive when
    the carrier came in above the frequency the radio was tuned to - and how
    far the peak stands above the noise floor of the transform.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    frames = iq.size // fft_size
    if frames == 0:
        raise ValueError(
            f"need at least {fft_size} samples to measure, got {iq.size}"
        )

    window = np.hanning(fft_size).astype(np.float32)
    block = iq[: frames * fft_size].reshape(frames, fft_size) * window
    # Averaged in power across frames: a single frame of a modulated carrier
    # wanders, and it is the carrier's average position we want.
    power = np.mean(np.abs(np.fft.fftshift(np.fft.fft(block, axis=1), axes=1)) ** 2, 0)

    bin_width = sample_rate / fft_size
    centre = fft_size // 2
    reach = max(2, int(round(search_hz / bin_width)))
    low, high = max(1, centre - reach), min(fft_size - 1, centre + reach + 1)

    window_power = power[low:high]
    peak = int(np.argmax(window_power)) + low
    # The noise floor from outside the search window, so the carrier being
    # measured cannot inflate its own reference.
    outside = np.concatenate((power[:low], power[high:]))
    floor = float(np.median(outside)) if outside.size else float(np.median(power))
    peak_db = 10.0 * np.log10(max(float(power[peak]), 1e-30) / max(floor, 1e-30))

    return (peak - centre + _parabolic_offset(power, peak)) * bin_width, peak_db


def _parabolic_offset(power: np.ndarray, peak: int) -> float:
    """Sub-bin position of a peak, from a parabola through three log points.

    In dB rather than in linear power because a windowed peak is close to a
    parabola on a log scale and nothing like one on a linear one.
    """
    if peak <= 0 or peak >= power.size - 1:
        return 0.0
    left, middle, right = (
        10.0 * np.log10(np.maximum(power[peak - 1 : peak + 2], 1e-30))
    )
    denominator = left - 2.0 * middle + right
    if denominator == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def ppm_from_offset(offset_hz: float, carrier_hz: float, current_ppm: int = 0) -> float:
    """The correction to apply, given a measured offset at a known frequency.

    The sign is the part worth spelling out, because getting it backwards
    doubles the error instead of removing it - and it was checked on hardware
    rather than reasoned about, since the direction depends on librtlsdr's
    internals as much as on physics. Forcing +50 ppm while tuned to a steady
    carrier at 162.55 MHz moved the measured offset by **+8246 Hz**, against
    the +8128 Hz that +50 ppm of that frequency comes to. So:

        offset(ppm) = offset(0) + ppm * carrier * 1e-6

    Raising the correction moves the carrier *up* the window. To bring an
    offset back to zero the correction therefore has to come *down* by that
    offset expressed in ppm, which is why this subtracts from whatever is
    already in force rather than replacing it.
    """
    if carrier_hz <= 0:
        raise ValueError("carrier frequency must be positive")
    return float(current_ppm) - (offset_hz / carrier_hz) * 1e6


def calibrate(
    iq: np.ndarray,
    sample_rate: float,
    carrier_hz: float,
    current_ppm: int = 0,
    search_hz: float = DEFAULT_SEARCH_HZ,
    fft_size: int = DEFAULT_FFT_SIZE,
) -> Calibration:
    """Measure a capture and turn it into a ppm figure to apply.

    The capture is measured in segments and the *median* is reported, so one
    segment that caught a modulation excursion cannot drag the answer, and the
    spread between them decides whether the answer is reported at all.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    per_segment = iq.size // SEGMENTS
    segments = (
        [iq[i * per_segment : (i + 1) * per_segment] for i in range(SEGMENTS)]
        if per_segment >= fft_size
        else [iq]
    )
    measured = [
        measure_offset_hz(segment, sample_rate, search_hz, fft_size)
        for segment in segments
    ]
    offsets = np.array([offset for offset, _ in measured])
    peak_db = float(np.median([peak for _, peak in measured]))
    offset_hz = float(np.median(offsets))
    spread_hz = float(np.std(offsets)) if offsets.size > 1 else 0.0

    exact = ppm_from_offset(offset_hz, carrier_hz, current_ppm)
    allowed = MAX_SPREAD_PPM * carrier_hz * 1e-6
    return Calibration(
        offset_hz=offset_hz,
        ppm=int(round(exact)),
        exact_ppm=exact,
        peak_db=peak_db,
        spread_hz=spread_hz,
        trustworthy=bool(peak_db >= MIN_PEAK_DB),
        steady=bool(spread_hz <= allowed),
    )


__all__ = [
    "DEFAULT_FFT_SIZE",
    "DEFAULT_SEARCH_HZ",
    "MAX_SPREAD_PPM",
    "MIN_PEAK_DB",
    "SEGMENTS",
    "Calibration",
    "calibrate",
    "measure_offset_hz",
    "ppm_from_offset",
]
