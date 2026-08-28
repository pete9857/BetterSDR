"""Tests for the power spectral density stage.

The claim that matters is calibration: a signal of known amplitude at a known
offset must land in the right bin at the right dBFS, independently of FFT size
and window. The scanner will set detection thresholds against these numbers, so
if they drift with display settings the detector drifts with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.psd import (
    WINDOWS,
    PeakHold,
    Spectrum,
    noise_floor_db,
    occupied_bandwidth_hz,
)

from . import synth

RATE = 2_400_000.0


def peak_frequency(spectrum: Spectrum, spectrum_db: np.ndarray) -> float:
    return float(spectrum.frequencies()[np.argmax(spectrum_db)])


# -- Calibration -----------------------------------------------------------


def test_full_scale_tone_reads_zero_dbfs():
    spectrum = Spectrum(4096, RATE)
    tone = synth.carrier(65_536, 300_000.0, amplitude=1.0)
    assert float(np.max(spectrum.process(tone))) == pytest.approx(0.0, abs=0.1)


def test_half_scale_tone_reads_minus_six_db():
    spectrum = Spectrum(4096, RATE)
    tone = synth.carrier(65_536, 300_000.0, amplitude=0.5)
    assert float(np.max(spectrum.process(tone))) == pytest.approx(-6.02, abs=0.1)


@pytest.mark.parametrize("window", WINDOWS)
def test_calibration_holds_for_every_window(window):
    spectrum = Spectrum(4096, RATE, window=window)
    tone = synth.carrier(65_536, 300_000.0, amplitude=1.0)
    assert float(np.max(spectrum.process(tone))) == pytest.approx(0.0, abs=0.2)


@pytest.mark.parametrize("fft_size", [512, 2048, 8192])
def test_calibration_holds_for_every_fft_size(fft_size):
    spectrum = Spectrum(fft_size, RATE)
    tone = synth.carrier(131_072, 300_000.0, amplitude=1.0)
    assert float(np.max(spectrum.process(tone))) == pytest.approx(0.0, abs=0.2)


def test_peak_lands_at_the_signal_frequency():
    spectrum = Spectrum(4096, RATE)
    spectrum_db = spectrum.process(synth.carrier(65_536, -450_000.0))
    assert peak_frequency(spectrum, spectrum_db) == pytest.approx(
        -450_000.0, abs=spectrum.bin_width_hz
    )


def test_frequencies_are_absolute_and_ascending():
    freqs = Spectrum(1024, RATE).frequencies(center_hz=100e6)
    assert np.all(np.diff(freqs) > 0)
    assert freqs[0] == pytest.approx(100e6 - RATE / 2, abs=RATE / 1024)
    assert freqs[512] == pytest.approx(100e6)


# -- Behaviour -------------------------------------------------------------


def test_dc_offset_does_not_create_a_centre_spike():
    """The RTL2832U always has a DC offset; it must not look like a station."""
    signal = synth.noise(65_536, rms=0.01) + np.complex64(0.5 + 0.5j)
    spectrum = Spectrum(4096, RATE, remove_dc=True)
    spectrum_db = spectrum.process(signal)

    centre = spectrum.fft_size // 2
    floor = noise_floor_db(spectrum_db)
    assert spectrum_db[centre] < floor + 10.0


def test_removing_dc_does_not_itself_create_a_centre_spike():
    """The subtler half of the same problem, and the one that shipped.

    Subtracting the plain per-frame mean nulls the *unwindowed* sum, not bin 0.
    A strong signal off centre leaks into that mean, so removing it writes the
    leakage back as a real spike at DC - measured 25 dB above the noise floor
    with one FM station 37 kHz away, which the scanner duly reported as a
    signal. The window-weighted mean nulls the bin itself and leaves its
    neighbours alone.
    """
    signal = synth.scene(
        65_536, [synth.fm(65_536, 37_000, amplitude=0.4, rate=RATE)], noise_rms=0.004
    )
    spectrum = Spectrum(4096, RATE, remove_dc=True)
    spectrum_db = spectrum.process(signal)

    centre = spectrum.fft_size // 2
    floor = noise_floor_db(spectrum_db)
    assert spectrum_db[centre] < floor
    # Only the one bin is touched - a wider notch would eat real signals.
    neighbours = np.concatenate(
        [spectrum_db[centre - 4 : centre - 1], spectrum_db[centre + 2 : centre + 5]]
    )
    assert np.all(neighbours > floor - 5.0)


def test_dc_offset_is_visible_when_removal_is_disabled():
    signal = synth.noise(65_536, rms=0.01) + np.complex64(0.5 + 0.5j)
    spectrum = Spectrum(4096, RATE, remove_dc=False)
    spectrum_db = spectrum.process(signal)

    centre = spectrum.fft_size // 2
    assert spectrum_db[centre] > noise_floor_db(spectrum_db) + 30.0


def test_averaging_reduces_the_scatter_of_the_noise_floor():
    noise = synth.noise(262_144, rms=0.01, seed=7)
    few = Spectrum(4096, RATE).process(noise[:4096])
    many = Spectrum(4096, RATE).process(noise)
    assert float(np.std(many)) < float(np.std(few))


def test_smoothing_damps_frame_to_frame_change():
    spectrum = Spectrum(1024, RATE, smoothing=0.9)
    quiet = synth.noise(4096, rms=0.001, seed=1)
    loud = synth.carrier(4096, 200_000.0, amplitude=1.0)

    spectrum.process(quiet)
    after = float(np.max(spectrum.process(loud)))
    # One loud frame into a heavily smoothed average must not jump to full scale.
    assert after < -5.0


def test_short_input_returns_nothing_rather_than_padding():
    spectrum = Spectrum(4096, RATE)
    assert spectrum.process(np.zeros(100, dtype=np.complex64)).size == 0


def test_rejects_invalid_settings():
    with pytest.raises(ValueError, match="power of two"):
        Spectrum(1000, RATE)
    with pytest.raises(ValueError, match="unknown window"):
        Spectrum(1024, RATE, window="triangle")
    with pytest.raises(ValueError, match="smoothing"):
        Spectrum(1024, RATE, smoothing=1.0)


# -- Peak hold -------------------------------------------------------------


def test_peak_hold_keeps_the_maximum_then_decays():
    hold = PeakHold(decay_db_per_frame=1.0)
    loud = np.full(64, -10.0, dtype=np.float32)
    quiet = np.full(64, -60.0, dtype=np.float32)

    hold.update(loud)
    assert float(hold.update(quiet)[0]) == pytest.approx(-11.0)
    assert float(hold.update(quiet)[0]) == pytest.approx(-12.0)


def test_peak_hold_rises_immediately():
    hold = PeakHold(decay_db_per_frame=1.0)
    hold.update(np.full(64, -60.0, dtype=np.float32))
    assert float(hold.update(np.full(64, -5.0, dtype=np.float32))[0]) == -5.0


def test_peak_hold_restarts_when_fft_size_changes():
    hold = PeakHold()
    hold.update(np.full(64, -10.0, dtype=np.float32))
    assert hold.update(np.full(128, -50.0, dtype=np.float32)).shape == (128,)


# -- Measurements the classifier will rely on ------------------------------


def test_noise_floor_ignores_a_strong_station():
    spectrum = Spectrum(4096, RATE)
    quiet = spectrum.process(synth.noise(65_536, rms=0.01, seed=3))
    with_station = spectrum.process(
        synth.scene(65_536, [synth.carrier(65_536, 400_000.0, 0.9)], noise_rms=0.01)
    )
    assert noise_floor_db(with_station) == pytest.approx(
        noise_floor_db(quiet), abs=1.5
    )


def test_occupied_bandwidth_separates_wide_from_narrow():
    spectrum = Spectrum(4096, RATE)
    wide = spectrum.process(synth.fm(262_144, 0.0, deviation_hz=75_000.0))
    spectrum.reset()
    narrow = spectrum.process(synth.fm(262_144, 0.0, deviation_hz=2_500.0))

    wide_hz = occupied_bandwidth_hz(wide, spectrum.bin_width_hz)
    narrow_hz = occupied_bandwidth_hz(narrow, spectrum.bin_width_hz)

    assert wide_hz > narrow_hz * 5
