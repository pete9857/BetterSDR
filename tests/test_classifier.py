"""Tests for classification.

The classifier's output is read by a beginner, so these check the sentence as
well as the answer. A label that is right but arrives with no reason attached
fails the thing the app is actually for: the user should be able to see why it
said what it said, and disagree with it if it is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.features import HdRadio
from bettersdr.dsp.psd import Spectrum
from bettersdr.scan.classifier import (
    Shape,
    Strength,
    classify,
    format_bandwidth,
    format_frequency,
    measure_shape,
)
from bettersdr.scan.detector import Detection, detect

from . import synth

RATE = 2_400_000
FRAMES = 4096 * 24

WIDE = Shape(carrier_fraction=0.02, flatness=0.45)
CARRIER = Shape(carrier_fraction=0.35, flatness=0.01)
DIGITAL = Shape(carrier_fraction=0.004, flatness=0.94)


def at(hz: float, width: float, snr: float = 40.0) -> Detection:
    return Detection(
        center_hz=hz,
        bandwidth_hz=width,
        peak_hz=hz,
        peak_dbfs=-80.0 + snr,
        floor_dbfs=-80.0,
    )


# -- Strength and formatting -----------------------------------------------


def test_strength_rises_with_signal_to_noise():
    assert Strength.from_snr(5.0) is Strength.WEAK
    assert Strength.from_snr(15.0) is Strength.FAIR
    assert Strength.from_snr(25.0) is Strength.GOOD
    assert Strength.from_snr(45.0) is Strength.STRONG


def test_every_strength_has_a_word_for_it():
    for strength in Strength:
        assert strength.label


def test_frequencies_are_formatted_the_way_a_dial_reads():
    assert format_frequency(94_900_000) == "94.9 MHz"
    assert format_frequency(1_000_000) == "1 MHz"
    assert format_frequency(530_000) == "530 kHz"


def test_bandwidths_are_formatted_in_units_people_say():
    assert format_bandwidth(200_000) == "200 kHz"
    assert format_bandwidth(2_000_000) == "2.0 MHz"
    assert format_bandwidth(500) == "500 Hz"


# -- Band plan agreement ---------------------------------------------------


def test_a_broadcast_width_signal_in_the_fm_band_is_an_fm_station():
    signal = classify(at(94_900_000, 180_000), WIDE)

    assert signal.label == "FM Radio"
    assert signal.mode == "wfm"
    assert signal.confidence >= 0.9
    assert signal.certain


def test_the_frequency_is_snapped_to_the_channel_raster():
    """94.9 is a station; 94.8987 is a measurement, and looks like a bug."""
    signal = classify(at(94_898_700, 180_000), WIDE)

    assert signal.frequency_hz == pytest.approx(94_900_000)
    assert signal.measured_hz == pytest.approx(94_898_700)
    assert signal.display_frequency == "94.9 MHz"


def test_a_width_the_band_does_not_expect_lowers_confidence():
    """Something 15 kHz wide in the FM band is not a broadcast station."""
    narrow = classify(at(94_900_000, 15_000), WIDE)
    normal = classify(at(94_900_000, 180_000), WIDE)

    assert narrow.confidence < normal.confidence
    assert "narrower than" in narrow.explanation
    assert not narrow.certain


def test_a_bare_carrier_in_a_voice_band_is_not_called_traffic():
    """Measured off air: an indoor aerial fills the airband with these.

    Stable two-kilohertz carriers from switching supplies and the dongle's own
    clock are not aeroplanes, and eighty of them labelled "Aircraft" would be
    the app confidently reporting a busy sky over an empty one.
    """
    signal = classify(at(122_875_000, 2_000), CARRIER)

    assert signal.label == "Unmodulated carrier"
    assert signal.band_name == "Aircraft"
    # The band still says how to listen to it, so Listen does something sane.
    assert signal.mode == "am"
    assert "interference from a nearby gadget" in signal.description


def test_a_weak_but_wide_signal_is_still_the_band_it_sits_in():
    """The other side of that: a distant station is narrow *and* has no carrier."""
    signal = classify(at(122_875_000, 5_000, snr=14.0), WIDE)

    assert signal.label == "Aircraft"
    assert "distant station" in signal.explanation
    assert signal.certain


def test_an_aircraft_channel_is_demodulated_as_am():
    """Getting this wrong is silence, and in Simple mode there is no fixing it."""
    signal = classify(at(118_300_000, 20_000), CARRIER)

    assert signal.label == "Aircraft"
    assert signal.mode == "am"


def test_the_narrower_allocation_still_wins():
    assert classify(at(162_475_000, 16_000), WIDE).label == "Weather Radio"


def test_the_explanation_names_the_band_and_the_width():
    explanation = classify(at(94_900_000, 180_000), WIDE).explanation

    assert "180 kHz wide" in explanation
    assert "FM Radio band" in explanation
    assert explanation.endswith("-> FM Radio")


def test_the_description_is_the_plain_english_one_from_the_band_plan():
    signal = classify(at(94_900_000, 180_000), WIDE)

    assert "car radio" in signal.description
    assert "demodulator" not in signal.description


# -- Nothing allocated here ------------------------------------------------


def test_an_unallocated_flat_signal_is_called_digital():
    signal = classify(at(1_400_000_000, 400_000), DIGITAL)

    assert signal.label == "Digital signal"
    assert signal.band_name is None
    assert "no allocation at this frequency" in signal.explanation


def test_an_unallocated_bare_tone_is_called_a_carrier():
    signal = classify(at(1_400_000_000, 900), CARRIER)

    assert signal.label == "Unmodulated carrier"
    assert signal.mode == "cw"


def test_an_unallocated_narrow_signal_is_called_a_two_way_radio():
    assert classify(at(1_400_000_000, 14_000), WIDE).label == "Two-way radio"


def test_an_unrecognisable_signal_says_so_rather_than_guessing():
    """"Unknown" is a valid answer; a confident wrong one erodes trust."""
    signal = classify(at(1_400_000_000, 900_000), WIDE)

    assert signal.label == "Unknown signal"
    assert not signal.certain


def test_frequency_is_left_alone_where_there_is_no_raster():
    signal = classify(at(1_400_000_123, 400_000), DIGITAL)
    assert signal.frequency_hz == pytest.approx(1_400_000_123)


# -- HD Radio --------------------------------------------------------------


def test_an_hd_station_says_so_in_its_reasons():
    hd = HdRadio(
        present=True,
        lower_snr_db=18.0,
        upper_snr_db=17.0,
        level_dbc=-14.0,
        flatness_db=1.0,
    )
    signal = classify(at(94_900_000, 180_000), WIDE, hd=hd)

    assert "digital sidebands" in signal.explanation
    assert signal.hd is not None and signal.hd.present


def test_an_analog_only_station_does_not_mention_sidebands():
    hd = HdRadio(False, -1.0, 0.5, -40.0, 8.0)
    signal = classify(at(94_900_000, 180_000), WIDE, hd=hd)

    assert "digital sidebands" not in signal.explanation


# -- Shape measurement -----------------------------------------------------


def test_shape_separates_a_carrier_from_a_wideband_signal():
    """The one measurement that tells AM from FM without the band plan."""
    spectrum = Spectrum(fft_size=4096, sample_rate=RATE)
    wide = spectrum.process(
        synth.scene(FRAMES, [synth.fm(FRAMES, 400_000, amplitude=0.5)], 0.005)
    )
    carrier = spectrum.process(
        synth.scene(FRAMES, [synth.am(FRAMES, 400_000, amplitude=0.4)], 0.005)
    )
    bin_width = spectrum.bin_width_hz

    wide_shape = measure_shape(wide, bin_width, 0.0, detect(wide, bin_width)[0])
    am_shape = measure_shape(carrier, bin_width, 0.0, detect(carrier, bin_width)[0])

    assert not wide_shape.has_carrier
    assert am_shape.has_carrier


def test_shape_recognises_a_dense_digital_signal_as_noise_like():
    spectrum = Spectrum(fft_size=4096, sample_rate=RATE)
    db = spectrum.process(
        synth.scene(
            FRAMES,
            [synth.band_noise(FRAMES, 300_000, 500_000, rms=0.2, both_sides=False)],
            0.005,
        )
    )
    detection = detect(db, spectrum.bin_width_hz)[0]
    shape = measure_shape(db, spectrum.bin_width_hz, 0.0, detection)

    assert shape.looks_digital


def test_flatness_never_claims_digital_inside_a_known_band():
    """Measured off air: real FM carrying music is as flat as OFDM.

    Both genuinely look like noise on a power spectrum, so flatness alone
    cannot separate them, and letting it try had the app calling most of the
    local FM dial digital. It is only allowed to decide where nothing is
    allocated and there is no better evidence to go on.
    """
    flat = Shape(carrier_fraction=0.004, flatness=0.94)

    assert "digital" not in flat.phrase
    assert "digital" not in classify(at(94_900_000, 180_000), flat).explanation
    assert classify(at(1_400_000_000, 400_000), flat).label == "Digital signal"


def test_shape_of_an_empty_spectrum_is_harmless():
    shape = measure_shape(np.zeros(0), 100.0, 0.0, at(94.9e6, 200_000))

    assert shape.carrier_fraction == 0.0
    assert not shape.looks_digital


def test_classifying_without_a_shape_still_works():
    """A caller that threw the spectrum away can still use the band plan."""
    assert classify(at(94_900_000, 180_000)).label == "FM Radio"
