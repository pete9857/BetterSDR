"""Tests for the noise blanker and spectral noise reduction.

The trap in a spectral-subtraction stage is that it is easy to write one which
reduces noise by damaging everything equally. So the claims tested here are
always a pair: how much of the noise went, *and* how much of the signal
survived. A stage that fails the second half is worse than no stage at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.denoise import NoiseBlanker, SpectralNoiseReduction

AUDIO_RATE = 48_000
IF_RATE = 240_000


def _stream(stage, samples: np.ndarray, block: int = 4096) -> np.ndarray:
    return np.concatenate(
        [stage.process(samples[i : i + block]) for i in range(0, samples.size, block)]
    )


def _rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -200.0
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.abs(samples) ** 2))), 1e-12)))


# -- NoiseBlanker ----------------------------------------------------------


def _tone(n: int, freq: float, rate: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def test_blanker_removes_impulses_and_leaves_the_signal_alone():
    clean = _tone(240_000, 1_000.0, IF_RATE)
    spiked = clean.copy()
    spiked[::4_800] += 5.0

    blanked = _stream(NoiseBlanker(IF_RATE), spiked)

    assert float(np.max(np.abs(spiked))) > 5.0
    assert float(np.max(np.abs(blanked))) < 2.0
    # Away from every impulse the samples must be untouched, not merely close.
    quiet = slice(1_000, 4_000)
    np.testing.assert_array_equal(blanked[quiet], clean[quiet])


def test_blanker_widening_carries_across_a_block_boundary():
    """An impulse straddling a boundary must be suppressed on both sides."""
    clean = _tone(8_192, 1_000.0, IF_RATE)
    spiked = clean.copy()
    spiked[4_093:4_099] += 5.0  # three samples either side of the split

    blanker = NoiseBlanker(IF_RATE, width=6)
    first = blanker.process(spiked[:4_096])
    second = blanker.process(spiked[4_096:])

    assert float(np.max(np.abs(first[4_093:]))) < 1.0
    assert float(np.max(np.abs(second[:3]))) < 1.0
    assert blanker.blanked == 6


def test_blanker_suppresses_an_impulse_shoulder_the_threshold_missed():
    """The point of widening: what follows a hit is held down as well.

    A shoulder at twice the surrounding level never crosses a threshold set
    at four times it, but it is still part of the same click.
    """
    clean = _tone(8_192, 1_000.0, IF_RATE, amplitude=0.3)
    spiked = clean.copy()
    spiked[2_000] *= 20.0  # the impulse proper
    spiked[2_001:2_004] *= 2.0  # its shoulder, under the threshold

    narrow = NoiseBlanker(IF_RATE, threshold=4.0, width=1).process(spiked)
    wide = NoiseBlanker(IF_RATE, threshold=4.0, width=6).process(spiked)

    assert float(np.max(np.abs(narrow[2_001:2_004]))) > 0.5
    assert float(np.max(np.abs(wide[2_001:2_004]))) < 0.4


def test_blanker_leaves_a_clean_stream_alone():
    clean = _tone(48_000, 1_000.0, IF_RATE)
    blanker = NoiseBlanker(IF_RATE)
    out = _stream(blanker, clean)
    np.testing.assert_allclose(out[2_000:], clean[2_000:], atol=1e-6)


# -- SpectralNoiseReduction ------------------------------------------------


def test_unity_settings_reconstruct_the_input_exactly():
    """With nothing subtracted, overlap-add must be transparent.

    This is the property that breaks first when the windowing or the hop
    bookkeeping is wrong, and every other claim about the stage rests on it.
    """
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.2, 48_000).astype(np.float32)

    stage = SpectralNoiseReduction(AUDIO_RATE, reduction_db=0.0, over_subtraction=0.0)
    out = _stream(stage, audio, block=700)

    settled = stage.settling_samples
    np.testing.assert_allclose(
        out[settled:], audio[settled : out.size], atol=1e-6
    )


def test_hiss_is_cut_without_damaging_the_speech():
    """The pair of claims that matter, on a signal that starts and stops.

    A permanently steady tone is the pathological case for any blind noise
    estimator - it looks exactly like a noise floor - so the test signal is
    gated, which is what real speech does.
    """
    rng = np.random.default_rng(1)
    t = np.arange(AUDIO_RATE * 6) / AUDIO_RATE
    warble = 0.25 * np.sin(2 * np.pi * (900 + 120 * np.sin(2 * np.pi * 3 * t)) * t)
    gate = (np.sin(2 * np.pi * 0.7 * t) > 0.1).astype(np.float32)
    voice = (warble * gate).astype(np.float32)
    noisy = (voice + rng.normal(0, 0.04, voice.size)).astype(np.float32)

    stage = SpectralNoiseReduction(AUDIO_RATE, reduction_db=15.0)
    out = _stream(stage, noisy)

    speaking = gate[: out.size] > 0.5
    silent = ~speaking
    silent[: AUDIO_RATE] = False  # let the estimator find the floor first

    speech_change = _rms_db(out[speaking]) - _rms_db(noisy[: out.size][speaking])
    hiss_change = _rms_db(out[silent]) - _rms_db(noisy[: out.size][silent])

    assert hiss_change < -3.0
    assert speech_change > -1.0


def test_complex_input_improves_signal_to_noise_at_the_if():
    rng = np.random.default_rng(4)
    t = np.arange(IF_RATE * 3) / IF_RATE
    gate = (np.sin(2 * np.pi * 1.5 * t) > 0.0).astype(np.float32)
    signal = (0.05 * gate * np.exp(2j * np.pi * 3_000.0 * t)).astype(np.complex64)
    noise = (rng.normal(0, 0.02, t.size) + 1j * rng.normal(0, 0.02, t.size))
    noisy = (signal + noise).astype(np.complex64)

    stage = SpectralNoiseReduction(
        IF_RATE, fft_size=256, complex_input=True, reduction_db=12.0
    )
    out = _stream(stage, noisy, block=32_768)

    # The last second only, once the estimator has seen at least one gap.
    window = slice(out.size - IF_RATE, out.size)
    on = gate[window] > 0.5
    before = _rms_db(noisy[window][on]) - _rms_db(noisy[window][~on])
    after = _rms_db(out[window][on]) - _rms_db(out[window][~on])
    assert after - before > 5.0


def test_time_constants_do_not_move_with_the_sample_rate():
    """Stated in real time, so the same setting means the same thing at 48 kHz
    and at 2.4 MS/s. Expressed per frame it would be fifty times faster on the
    wide window and the estimator would simply follow the signal."""
    slow = SpectralNoiseReduction(48_000, rise_db_per_second=8.0)
    fast = SpectralNoiseReduction(2_400_000, rise_db_per_second=8.0)
    ratio = slow.rise_db_per_frame / fast.rise_db_per_frame
    assert ratio == pytest.approx(2_400_000 / 48_000, rel=1e-9)


def test_rejects_a_non_power_of_two_transform():
    with pytest.raises(ValueError, match="power of two"):
        SpectralNoiseReduction(AUDIO_RATE, fft_size=500)
