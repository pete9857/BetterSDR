"""Tests for the streaming filter primitives.

The property that matters most here is that block-by-block processing gives
exactly the same answer as processing the whole stream at once. Every audible
tick and click in a naive SDR comes from a stage that quietly forgets its
history at a block boundary, and that defect is invisible unless a test
explicitly compares the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.filters import (
    DcBlock,
    Deemphasis,
    Discriminator,
    FirDecimator,
    Squelch,
    power_dbfs,
)

from . import synth

RATE = 2_400_000


def _tone(n: int, freq: float, rate: float) -> np.ndarray:
    t = np.arange(n) / rate
    return np.exp(2j * np.pi * freq * t).astype(np.complex64)


# -- FirDecimator ----------------------------------------------------------


def test_decimator_output_rate_and_length():
    dec = FirDecimator.lowpass(10, 100_000, RATE)
    out = dec.process(np.zeros(48_000, dtype=np.complex64))
    assert dec.output_rate == RATE / 10
    assert out.size == 4_800


def test_decimator_streaming_matches_one_shot():
    signal = synth.scene(96_000, [synth.carrier(96_000, 30_000.0)], noise_rms=0.02)

    one_shot = FirDecimator.lowpass(10, 100_000, RATE).process(signal)

    streamed = FirDecimator.lowpass(10, 100_000, RATE)
    chunks = [streamed.process(block) for block in np.split(signal, 12)]
    joined = np.concatenate(chunks)

    assert joined.size == one_shot.size
    np.testing.assert_allclose(joined, one_shot, atol=1e-6)


def test_decimator_rejects_out_of_band_signal():
    passband = _tone(48_000, 20_000.0, RATE)
    stopband = _tone(48_000, 600_000.0, RATE)
    dec = FirDecimator.lowpass(10, 100_000, RATE)

    kept = power_dbfs(dec.process(passband))
    dec.reset()
    rejected = power_dbfs(dec.process(stopband))

    assert kept > -1.0
    assert rejected < kept - 50.0


def test_decimator_rejects_ragged_block():
    dec = FirDecimator.lowpass(10, 100_000, RATE)
    with pytest.raises(ValueError, match="not a multiple"):
        dec.process(np.zeros(1005, dtype=np.complex64))


def test_decimator_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError, match="outside"):
        FirDecimator.lowpass(10, RATE, RATE)


def test_long_filter_uses_the_fft_path():
    """Guards the branch itself, so a threshold change cannot silently skip it."""
    narrow = FirDecimator.bandpass(1, 300.0, 3_000.0, 48_000, taps_per_phase=512)
    wide = FirDecimator.lowpass(10, 100_000, RATE)
    assert narrow._use_fft is True
    assert wide._use_fft is False


def test_fft_path_streaming_matches_one_shot():
    """The SSB and CW filters take this path, so it must join blocks cleanly."""
    rate = 48_000
    signal = synth.scene(9_600, [synth.carrier(9_600, 1_200.0, rate=rate)], 0.01)

    one_shot = FirDecimator.bandpass(
        1, 300.0, 3_000.0, rate, taps_per_phase=512
    ).process(signal)

    streamed = FirDecimator.bandpass(1, 300.0, 3_000.0, rate, taps_per_phase=512)
    joined = np.concatenate([streamed.process(b) for b in np.split(signal, 10)])

    assert joined.size == one_shot.size
    np.testing.assert_allclose(joined, one_shot, atol=1e-5)


def test_fft_and_direct_paths_agree():
    """Both branches must compute the same filter, not merely a similar one."""
    signal = synth.scene(48_000, [synth.carrier(48_000, 40_000.0)], noise_rms=0.02)

    direct = FirDecimator.lowpass(10, 100_000, RATE)
    assert direct._use_fft is False
    expected = direct.process(signal)

    forced = FirDecimator.lowpass(10, 100_000, RATE)
    forced._use_fft = True
    np.testing.assert_allclose(forced.process(signal), expected, atol=1e-5)


def test_bandpass_keeps_one_side_of_zero_only():
    rate = 48_000
    upper = _tone(48_000, 1_500.0, rate)
    lower = _tone(48_000, -1_500.0, rate)
    dec = FirDecimator.bandpass(1, 300.0, 3_000.0, rate, taps_per_phase=512)

    kept = power_dbfs(dec.process(upper))
    dec.reset()
    rejected = power_dbfs(dec.process(lower))

    assert kept > -3.0
    assert rejected < kept - 40.0


# -- IIR stages ------------------------------------------------------------


def test_deemphasis_passes_dc_unchanged():
    deemph = Deemphasis(240_000, tau_us=75.0)
    out = deemph.process(np.ones(20_000, dtype=np.float32))
    # Settles to unity gain, so overall loudness is not altered.
    assert out[-1] == pytest.approx(1.0, abs=1e-3)


def test_deemphasis_attenuates_treble():
    rate = 240_000
    deemph = Deemphasis(rate, tau_us=75.0)
    n = 240_000
    t = np.arange(n) / rate
    low = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 10_000.0 * t).astype(np.float32)

    low_out = np.std(deemph.process(low))
    deemph.reset()
    high_out = np.std(deemph.process(high))

    # The 75 us corner sits at ~2.1 kHz, so 10 kHz should be well down on 100 Hz.
    assert 20 * np.log10(high_out / low_out) < -10.0


def test_dc_block_removes_offset_but_keeps_tone():
    rate = 48_000
    t = np.arange(rate) / rate
    signal = (2.0 + np.sin(2 * np.pi * 1_000.0 * t)).astype(np.float32)
    out = DcBlock().process(signal)
    # The 7.6 Hz corner means the offset decays over ~1000 samples, so skip a
    # generous ten time constants before measuring what is left of it.
    settled = out[10_000:]
    assert abs(float(np.mean(settled))) < 0.01
    assert float(np.std(settled)) == pytest.approx(0.707, abs=0.05)


def test_biquad_streaming_matches_one_shot():
    signal = np.random.default_rng(3).normal(0, 1, 12_000).astype(np.float32)
    one_shot = Deemphasis(48_000).process(signal)

    streamed = Deemphasis(48_000)
    joined = np.concatenate([streamed.process(b) for b in np.split(signal, 10)])

    np.testing.assert_allclose(joined, one_shot, rtol=1e-5, atol=1e-6)


# -- Discriminator ---------------------------------------------------------


def test_discriminator_recovers_constant_frequency():
    rate = 240_000
    offset = 30_000.0
    out = Discriminator().process(_tone(24_000, offset, rate))
    # Phase step per sample is 2*pi*f/rate; drop the first sample, which has
    # no predecessor to difference against.
    expected = 2 * np.pi * offset / rate
    assert float(np.mean(out[1:])) == pytest.approx(expected, rel=1e-4)


def test_discriminator_streaming_matches_one_shot():
    signal = synth.fm(48_000, 0.0, tone_hz=2_000.0, rate=240_000)
    one_shot = Discriminator().process(signal)

    streamed = Discriminator()
    joined = np.concatenate([streamed.process(b) for b in np.split(signal, 8)])

    np.testing.assert_allclose(joined, one_shot, atol=1e-6)


# -- Squelch ---------------------------------------------------------------


def test_squelch_opens_on_signal_and_closes_on_noise():
    squelch = Squelch(48_000, threshold_dbfs=-40.0, hysteresis_db=3.0)
    assert squelch.update(-20.0) is True
    assert squelch.update(-41.0) is False
    # Hysteresis: -39 is above the threshold but not above threshold+3.
    assert squelch.update(-39.0) is False
    assert squelch.update(-30.0) is True


def test_squelch_ramps_rather_than_switching():
    squelch = Squelch(48_000, threshold_dbfs=-40.0, attack_ms=5.0)
    squelch.update(-10.0)
    out = squelch.process(np.ones(48, dtype=np.float32))
    # 48 samples is well inside a 5 ms (240 sample) attack, so the gain should
    # still be climbing rather than having jumped straight to unity.
    assert 0.0 < out[0] < out[-1] < 1.0
    assert np.all(np.diff(out) > 0)


def test_squelch_silences_when_closed():
    squelch = Squelch(48_000, threshold_dbfs=-40.0)
    squelch.update(-80.0)
    out = squelch.process(np.ones(4_800, dtype=np.float32))
    assert float(np.max(np.abs(out))) == 0.0


# -- Level metering --------------------------------------------------------


def test_power_dbfs_matches_known_amplitude():
    half_scale = 0.5 * np.ones(1_000, dtype=np.complex64)
    assert power_dbfs(half_scale) == pytest.approx(-6.02, abs=0.05)


def test_power_dbfs_floors_on_silence():
    assert power_dbfs(np.zeros(0, dtype=np.complex64)) == -120.0
