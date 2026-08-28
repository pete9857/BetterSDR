"""Tests for the classifier feature extractors.

The HD Radio test is the interesting one: it has to fire on a station that
carries digital sidebands, stay quiet on one that does not, and - the case
that actually bites in a real sweep - stay quiet when a neighbouring station
on the adjacent channel happens to sit where a sideband would be.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp import features
from bettersdr.dsp.psd import Spectrum

from . import synth

RATE = 2_400_000
SAMPLES = 1 << 18  # 64 averaged frames at 4096, enough for a steady floor


def spectrum_of(iq: np.ndarray) -> tuple[np.ndarray, float]:
    spec = Spectrum(fft_size=4096, sample_rate=float(RATE))
    return spec.process(iq), spec.bin_width_hz


# -- HD Radio detection ----------------------------------------------------


def test_detects_hd_sidebands_on_a_hybrid_station():
    iq = synth.hd_radio_fm(SAMPLES, digital_dbc=-15.0)
    result = features.detect_hd_radio(*spectrum_of(iq))

    assert result.present is True
    assert result.lower_snr_db > features.HD_MIN_SNR_DB
    assert result.upper_snr_db > features.HD_MIN_SNR_DB
    # Real stations run -20 to -10 dBc, so the reported level must land in
    # that neighbourhood rather than being an arbitrary number.
    assert -25.0 < result.level_dbc < -5.0


def test_reported_level_tracks_the_digital_power():
    """The absolute figure is a density ratio; what must hold is that it moves.

    `level_dbc` compares mean power per bin between two regions of different
    shape, so it approximates the broadcaster's total-power ratio rather than
    equalling it. Tracking is the property worth pinning down.
    """
    loud = features.detect_hd_radio(
        *spectrum_of(synth.hd_radio_fm(SAMPLES, digital_dbc=-10.0))
    )
    quiet = features.detect_hd_radio(
        *spectrum_of(synth.hd_radio_fm(SAMPLES, digital_dbc=-20.0))
    )
    assert loud.level_dbc - quiet.level_dbc == pytest.approx(10.0, abs=1.0)


def test_analog_only_station_is_not_flagged():
    iq = synth.scene(SAMPLES, [synth.fm(SAMPLES, 0.0)], noise_rms=0.002)
    result = features.detect_hd_radio(*spectrum_of(iq))

    assert result.present is False


def test_one_sided_energy_is_rejected_as_an_adjacent_station():
    """A neighbour 150 kHz up must not read as a digital sideband."""
    iq = synth.hd_radio_fm(SAMPLES, digital_dbc=-15.0, both_sides=False)
    result = features.detect_hd_radio(*spectrum_of(iq))

    assert result.present is False
    assert result.upper_snr_db - result.lower_snr_db > features.HD_MAX_IMBALANCE_DB


def test_distant_station_with_buried_sidebands_is_not_flagged():
    """A far-off HD station is honestly reported as undecidable, not as HD.

    The digital part is 15 dB below an already weak analog signal, so it goes
    under the noise first. Claiming HD here would be a promise the decoder
    could not keep.
    """
    iq = synth.scene(
        SAMPLES,
        [synth.hd_radio_fm(SAMPLES, amplitude=0.02, digital_dbc=-15.0)],
        noise_rms=0.02,
    )
    result = features.detect_hd_radio(*spectrum_of(iq))

    assert result.present is False
    assert result.lower_snr_db < features.HD_MIN_SNR_DB


def test_detection_works_when_the_station_is_offset_from_centre():
    """Offset tuning is normal: the station rarely sits on the DC bin."""
    offset = 400_000.0
    iq = synth.hd_radio_fm(SAMPLES, offset_hz=offset, digital_dbc=-15.0)
    spectrum_db, bin_width = spectrum_of(iq)

    assert features.detect_hd_radio(spectrum_db, bin_width, offset).present is True
    # Looking in the wrong place must not find it.
    assert features.detect_hd_radio(spectrum_db, bin_width, 0.0).present is False


def test_flatness_separates_ofdm_from_a_sloping_skirt():
    flat = synth.hd_radio_fm(SAMPLES, digital_dbc=-15.0)
    # Over-deviated FM splatters into the sideband region with a slope.
    sloped = synth.fm(SAMPLES, 0.0, tone_hz=200.0, deviation_hz=180_000.0)

    assert features.detect_hd_radio(*spectrum_of(flat)).flatness_db < 2.0
    assert features.detect_hd_radio(*spectrum_of(sloped)).flatness_db > 2.0


def test_refuses_a_window_too_narrow_to_decide():
    """Silently answering "no" would make it depend on sweep alignment."""
    spec = Spectrum(fft_size=4096, sample_rate=300_000.0)
    narrow = spec.process(synth.fm(SAMPLES, 0.0, rate=300_000.0))
    with pytest.raises(ValueError, match="needs at least"):
        features.detect_hd_radio(narrow, spec.bin_width_hz)


# -- Explanations ----------------------------------------------------------


def test_simple_summary_avoids_jargon_and_numbers():
    result = features.detect_hd_radio(*spectrum_of(synth.hd_radio_fm(SAMPLES)))
    summary = result.summary

    assert "HD" in summary
    assert not any(char.isdigit() for char in summary)
    for jargon in ("dB", "sideband", "OFDM", "carrier", "IBOC"):
        assert jargon not in summary


def test_detail_explains_the_reasoning_with_numbers():
    result = features.detect_hd_radio(*spectrum_of(synth.hd_radio_fm(SAMPLES)))
    assert "129-198 kHz" in result.detail
    assert "HD Radio" in result.detail


def test_detail_explains_a_negative_too():
    iq = synth.scene(SAMPLES, [synth.fm(SAMPLES, 0.0)], noise_rms=0.002)
    assert "analog only" in features.detect_hd_radio(*spectrum_of(iq)).detail
