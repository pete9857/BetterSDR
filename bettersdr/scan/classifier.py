"""Turning a bump in the spectrum into a sentence a beginner can read.

The rules here are rules, not a model, and that is the whole point. The app's
promise is that it can say *why* it thinks something - "constant power, 150 kHz
wide, sits in the 88-108 MHz broadcast band, so: an FM radio station" - and a
classifier that cannot explain itself would break that promise no matter how
accurate it was. Explainability is the product, not a debugging aid.

Two sources of evidence are combined. The band plan says what is *allocated*
at a frequency, which is strong prior knowledge and usually decisive. The shape
features say what the signal actually looks like, which is what catches the
cases where the allocation is empty, shared, or being used by something else.
When the two agree, confidence is high; when they disagree, the app says so
rather than quietly picking one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..dsp import features
from ..dsp.features import HdRadio
from .bandplan import DEFAULT_REGION, Band, find
from .detector import Detection

# Above this share of its power in one bin, a signal has a discrete carrier.
CARRIER_FRACTION = 0.20

# Above this, a signal is as featureless as noise across its own band.
#
# Set from off-air measurement, and higher than synthetic testing suggested.
# A tone-modulated FM signal has a line spectrum and measures about 0.45, so
# 0.6 looked like a comfortable margin; real FM carrying music spreads its
# energy smoothly and measures 0.7 to 0.9, which had the app calling most of
# the local dial digital. Flatness on its own cannot separate OFDM from analog
# FM with programme material on it - both genuinely look like noise - so it is
# used only where nothing is allocated and there is no better evidence. What
# actually identifies a digital signal in a known band is *where* the flat
# energy sits, which is what the HD Radio sideband test does.
DIGITAL_FLATNESS = 0.85

# How far the measured width may differ from the band plan's expectation and
# still count as agreement. Generous on purpose: measured width depends on the
# detection threshold and on how hard the station is being modulated at the
# moment it was swept.
WIDTH_TOLERANCE = (0.3, 3.0)

# Width disagreement is not symmetric, and treating it as though it were makes
# the app call every weak station a guess. A signal that measures *wider* than
# its channel cannot be that channel. A signal that measures *narrower* very
# often is: when it is weak, only the middle of it clears the noise floor, so
# a real 200 kHz broadcast station can measure 9 kHz. Off air in the FM band
# that one distinction is the difference between a list where two thirds of
# the entries are hedged and one where only the odd ones are.
#
# The number is high because it gates on the *peak*, while what actually
# matters is whether the signal's edges cleared the noise. A distant NOAA
# transmitter showed a 26 dB peak with only its 2 kHz carrier above the floor,
# and got hedged, while an identical one at 21 dB did not - the same situation
# described two different ways. Well above any of that, a signal that is still
# narrow really is narrow.
WIDTH_TRUSTED_SNR_DB = 35.0

# Below this, a signal is mostly noise, and noise is flat and has no carrier -
# so the shape features describe the noise rather than the signal. Saying "this
# looks digital" about a station that is merely faint is worse than saying
# nothing, because it is a confident statement about the wrong thing.
SHAPE_TRUSTED_SNR_DB = 15.0

# The label for a signal whose power is all on one steady tone. Named because
# two other modules have to recognise it: it is the reading a short dwell
# produces from a real transmitter caught between two words, so anything with
# better evidence - the monitor, which actually listens to the channel - has to
# be able to tell that this is the reading it is contradicting.
UNMODULATED = "Unmodulated carrier"

CONFIDENCE_BAND_AND_SHAPE = 0.95
CONFIDENCE_BAND_ONLY = 0.60
CONFIDENCE_SHAPE_ONLY = 0.45
CONFIDENCE_UNKNOWN = 0.20


class Strength(IntEnum):
    """Signal strength as a beginner should see it: four bars, not a number."""

    WEAK = 1
    FAIR = 2
    GOOD = 3
    STRONG = 4

    @classmethod
    def from_snr(cls, snr_db: float) -> Strength:
        if snr_db >= 30.0:
            return cls.STRONG
        if snr_db >= 20.0:
            return cls.GOOD
        if snr_db >= 12.0:
            return cls.FAIR
        return cls.WEAK

    @property
    def label(self) -> str:
        return {1: "Weak", 2: "Fair", 3: "Good", 4: "Strong"}[int(self)]


def format_frequency(hz: float) -> str:
    """A frequency the way a radio dial would show it."""
    if hz >= 1e9:
        return f"{hz / 1e9:.4f} GHz".replace("0 GHz", " GHz")
    if hz >= 1e6:
        return f"{hz / 1e6:g} MHz"
    return f"{hz / 1e3:g} kHz"


def format_bandwidth(hz: float) -> str:
    """A width in the units a person would say out loud."""
    if hz >= 1e6:
        return f"{hz / 1e6:.1f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.0f} kHz"
    return f"{hz:.0f} Hz"


@dataclass(frozen=True)
class Shape:
    """What a signal looks like, independent of how strong it is."""

    carrier_fraction: float
    flatness: float

    @property
    def has_carrier(self) -> bool:
        return self.carrier_fraction >= CARRIER_FRACTION

    @property
    def looks_digital(self) -> bool:
        return self.flatness >= DIGITAL_FLATNESS

    @property
    def phrase(self) -> str:
        """The shape as one clause of the explanation sentence.

        Says only what a power spectrum can actually support. "There is a
        carrier" and "the power is spread out" are both directly measured;
        "this is digital" is not, and claiming it from flatness alone was
        wrong about most of a real FM dial.
        """
        if self.has_carrier:
            return "power concentrated on a single carrier"
        return "power spread across the whole channel"


def measure_shape(
    spectrum_db: np.ndarray,
    bin_width_hz: float,
    center_hz: float,
    detection: Detection,
) -> Shape:
    """Measure a detection's shape from the spectrum it was found in.

    The slice covers only the detection's own bins, so the answer describes the
    signal rather than the empty spectrum around it.
    """
    bins = spectrum_db.size
    if bins == 0 or bin_width_hz <= 0:
        return Shape(carrier_fraction=0.0, flatness=0.0)
    origin = bins // 2
    low = int(round(origin + (detection.start_hz - center_hz) / bin_width_hz))
    high = int(round(origin + (detection.end_hz - center_hz) / bin_width_hz))
    window = spectrum_db[max(0, low) : min(bins, max(high, low + 1))]
    return Shape(
        carrier_fraction=features.carrier_fraction(window),
        flatness=features.spectral_flatness(window),
    )


@dataclass(frozen=True)
class Signal:
    """A classified signal: what it is, how to hear it, and why we think so."""

    frequency_hz: float
    measured_hz: float
    bandwidth_hz: float
    peak_dbfs: float
    snr_db: float

    label: str
    icon: str
    description: str
    mode: str
    demod_bandwidth_hz: float
    confidence: float
    reasons: tuple[str, ...]
    band_name: str | None = None
    hd: HdRadio | None = None

    @property
    def strength(self) -> Strength:
        return Strength.from_snr(self.snr_db)

    @property
    def display_frequency(self) -> str:
        return format_frequency(self.frequency_hz)

    @property
    def headline(self) -> str:
        """The one line on the card: what, where, how strong."""
        return f"{self.label} - {self.display_frequency} - {self.strength.label}"

    @property
    def explanation(self) -> str:
        """Why the app thinks this is what it is, in one sentence."""
        return f"{', '.join(self.reasons)} -> {self.label}"

    @property
    def certain(self) -> bool:
        """Whether to present this as a fact rather than as a best guess.

        Requires the band plan and the measured shape to agree. Knowing an
        allocation is not enough on its own: something 30 kHz wide in the FM
        broadcast band is *not* a broadcast station, and saying so confidently
        would also have it demodulated as wideband FM, which is silence.
        """
        return self.confidence >= CONFIDENCE_BAND_AND_SHAPE


def _shape_phrase(shape: Shape, detection: Detection) -> str:
    """The shape clause, or an honest admission that the signal is too faint."""
    if detection.snr_db < SHAPE_TRUSTED_SNR_DB:
        return "only just clear of the noise, so its shape is hard to read"
    return shape.phrase


def _shape_only(
    detection: Detection, shape: Shape
) -> tuple[str, str, str, float, float, str]:
    """Guess from shape alone: label, icon, mode, bandwidth, confidence, reason.

    Reached when nothing is allocated at this frequency. The answers are
    deliberately cautious and generic - "Digital signal" is a useful and
    non-embarrassing thing to say, and a confident wrong guess is not.
    """
    width = detection.bandwidth_hz
    if detection.snr_db < SHAPE_TRUSTED_SNR_DB:
        # No allocation to fall back on and too faint to read the shape, so
        # there is genuinely nothing to say beyond "something is here".
        return ("Unknown signal", "question", "raw", max(width, 12_500.0),
                CONFIDENCE_UNKNOWN, "too faint to make anything of")
    if shape.has_carrier and width <= 1_500:
        return (UNMODULATED, "wave", "cw", 500.0, CONFIDENCE_SHAPE_ONLY,
                "a bare tone with no modulation on it")
    if shape.looks_digital:
        return ("Digital signal", "chip", "raw", max(width, 12_500.0),
                CONFIDENCE_SHAPE_ONLY,
                "power spread dead flat across its whole width with no "
                "carrier, the way a digital transmission looks")
    if shape.has_carrier:
        return ("AM signal", "wave", "am", max(width, 6_000.0),
                CONFIDENCE_SHAPE_ONLY, shape.phrase)
    if width <= 30_000:
        return ("Two-way radio", "walkie", "nfm", 12_500.0, CONFIDENCE_SHAPE_ONLY,
                "narrow and constant, the shape of a handheld radio")
    if width <= 300_000:
        return ("Wideband FM signal", "music", "wfm", 200_000.0,
                CONFIDENCE_SHAPE_ONLY, shape.phrase)
    return ("Unknown signal", "question", "raw", max(width, 12_500.0),
            CONFIDENCE_UNKNOWN, "nothing here matches a shape we recognise")


def classify(
    detection: Detection,
    shape: Shape | None = None,
    hd: HdRadio | None = None,
    region: str = DEFAULT_REGION,
) -> Signal:
    """Decide what a detection is.

    `shape` comes from `measure_shape`; without it the band plan alone decides,
    which is what a caller that has thrown away the spectrum can still do.
    `hd` is the HD Radio verdict for an FM station, if one was measured.
    """
    if shape is None:
        shape = Shape(carrier_fraction=0.0, flatness=0.0)
    band: Band | None = find(detection.center_hz, region)
    width_phrase = f"{format_bandwidth(detection.bandwidth_hz)} wide"

    if band is None:
        label, icon, mode, demod_bw, confidence, reason = _shape_only(detection, shape)
        description = (
            "Nothing is officially allocated at this frequency, so this is a "
            "guess from the shape of the signal alone."
        )
        reasons = (width_phrase, reason, "no allocation at this frequency")
        frequency = detection.center_hz
    else:
        ratio = detection.bandwidth_hz / max(band.bandwidth_hz, 1.0)
        too_wide = ratio > WIDTH_TOLERANCE[1]
        # Narrow only counts against it when the signal is strong enough for
        # the width to have been measurable in the first place.
        too_narrow = ratio < WIDTH_TOLERANCE[0]
        measurable = detection.snr_db >= WIDTH_TRUSTED_SNR_DB
        # Where stations run around the clock, a steady carrier is not an
        # anomaly, it is the shape a station has: an AM broadcaster radiates
        # its carrier continuously and only the sidebands follow the
        # programme. So a narrow measurement there agrees with the band
        # rather than contradicting it.
        carrier_expected = band.continuous and shape.has_carrier
        agrees = not too_wide and not (
            too_narrow and measurable and not carrier_expected
        )

        # A bare carrier a fraction of the channel wide, with all its power in
        # one bin, is not a conversation on that channel. Off air with an
        # indoor aerial the airband filled up with these - stable 2 kHz
        # carriers from switching supplies and the dongle's own clock - and
        # calling them "Aircraft" would have the app confidently reporting
        # eighty aeroplanes in an empty sky. The band still supplies the mode
        # and bandwidth, so Listen does the sensible thing.
        bare_carrier = (
            shape.has_carrier
            and ratio <= WIDTH_TOLERANCE[0]
            and not band.continuous
        )
        if bare_carrier:
            label, icon, mode = UNMODULATED, "wave", band.mode
            demod_bw = band.bandwidth_hz
            description = (
                f"A steady tone with nothing on it, in the part of the dial "
                f"used for {band.name}. Often interference from a nearby "
                f"gadget rather than a real station."
            )
            return Signal(
                frequency_hz=band.snap(detection.center_hz),
                measured_hz=detection.center_hz,
                bandwidth_hz=detection.bandwidth_hz,
                peak_dbfs=detection.peak_dbfs,
                snr_db=detection.snr_db,
                label=label,
                icon=icon,
                description=description,
                mode=mode,
                demod_bandwidth_hz=demod_bw,
                confidence=CONFIDENCE_SHAPE_ONLY,
                reasons=(
                    width_phrase,
                    "all of its power on one steady tone",
                    f"far narrower than the {format_bandwidth(band.bandwidth_hz)} "
                    f"a real signal here would be",
                ),
                band_name=band.name,
                hd=hd,
            )

        label, icon, mode = band.name, band.icon, band.mode
        demod_bw = band.bandwidth_hz
        description = band.description
        confidence = CONFIDENCE_BAND_AND_SHAPE if agrees else CONFIDENCE_BAND_ONLY
        placement = (
            f"sits in the {format_frequency(band.start_hz)} to "
            f"{format_frequency(band.end_hz)} {band.name} band"
        )
        if not agrees:
            reasons = (
                width_phrase,
                f"{'wider' if too_wide else 'narrower'} than the "
                f"{format_bandwidth(band.bandwidth_hz)} expected here",
                placement,
            )
        elif too_narrow and carrier_expected:
            reasons = (
                "a steady carrier, which is how stations here transmit",
                f"only {format_bandwidth(detection.bandwidth_hz)} of it clears "
                f"the noise right now, because the rest comes and goes with "
                f"the programme",
                placement,
            )
        elif too_narrow:
            reasons = (
                f"only the strongest {format_bandwidth(detection.bandwidth_hz)} "
                f"of it clears the noise, which is what a distant station looks "
                f"like",
                placement,
            )
        else:
            reasons = (width_phrase, _shape_phrase(shape, detection), placement)
        # Snap to the channel raster where the band has one. A detection's
        # centroid lands within a few kHz of the transmitter, and 94.9 MHz is a
        # better thing to show, and to tune, than 94.8987 MHz.
        frequency = band.snap(detection.center_hz)

    if hd is not None and hd.present:
        reasons = (*reasons, "with flat digital sidebands either side")

    return Signal(
        frequency_hz=frequency,
        measured_hz=detection.center_hz,
        bandwidth_hz=detection.bandwidth_hz,
        peak_dbfs=detection.peak_dbfs,
        snr_db=detection.snr_db,
        label=label,
        icon=icon,
        description=description,
        mode=mode,
        demod_bandwidth_hz=demod_bw,
        confidence=confidence,
        reasons=reasons,
        band_name=band.name if band else None,
        hd=hd,
    )


__all__ = [
    "CARRIER_FRACTION",
    "DIGITAL_FLATNESS",
    "UNMODULATED",
    "Shape",
    "Signal",
    "Strength",
    "classify",
    "format_bandwidth",
    "format_frequency",
    "measure_shape",
]
