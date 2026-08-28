"""Tests for IQ correction and the two processing chains.

Quadrature imbalance is measured the way it is heard: as *image rejection*,
the difference between a signal and the mirror image of itself the receiver
invents on the other side of centre. A receiver with 25 dB of rejection puts a
ghost of every strong station on top of whatever is 2 x offset away from it,
and the scanner would report that ghost as a signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.chain import AudioChain, FrontEnd
from bettersdr.dsp.correct import (
    DcRemover,
    FrequencyShifter,
    IqBalance,
    swap_iq,
)

RATE = 2_400_000
OFFSET = 300_000.0


def _carrier(n: int, offset_hz: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(n) / RATE
    return (amplitude * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def _unbalance(iq: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    """Plant a known gain and phase error between the two channels."""
    gain = 10 ** (gain_db / 20)
    phase = np.radians(phase_deg)
    q = gain * (np.sin(phase) * iq.real + np.cos(phase) * iq.imag)
    return (iq.real + 1j * q).astype(np.complex64)


def _image_rejection_db(iq: np.ndarray, offset_hz: float = OFFSET) -> float:
    size = 65_536
    window = np.hanning(size)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq[-size:] * window)))
    freqs = np.fft.fftshift(np.fft.fftfreq(size, 1 / RATE))
    wanted = int(np.argmin(np.abs(freqs - offset_hz)))
    ghost = int(np.argmin(np.abs(freqs + offset_hz)))
    return float(20 * np.log10(spectrum[wanted] / max(spectrum[ghost], 1e-20)))


def _stream(stage, iq: np.ndarray, block: int = 32_768) -> np.ndarray:
    return np.concatenate(
        [stage.process(iq[i : i + block]) for i in range(0, iq.size, block)]
    )


# -- DC removal ------------------------------------------------------------


def test_dc_removal_takes_out_a_planted_offset():
    signal = _carrier(RATE // 4, OFFSET) + np.complex64(0.05 + 0.02j)
    cleaned = _stream(DcRemover(RATE), signal)
    assert abs(complex(np.mean(signal))) > 0.04
    assert abs(complex(np.mean(cleaned[-262_144:]))) < 1e-4


def test_dc_removal_time_constant_does_not_move_with_the_block_size():
    """Specified in seconds, so a 27 ms block and a 273 ms block agree.

    Every other rate-dependent constant in the app got this wrong once; this
    pins it down for the one place where the correction is per block.
    """
    signal = (_carrier(RATE // 2, OFFSET) + np.complex64(0.05 + 0.02j)).astype(
        np.complex64
    )
    big = _stream(DcRemover(RATE), signal, block=131_072)
    small = _stream(DcRemover(RATE), signal, block=16_384)
    assert abs(complex(np.mean(big[-262_144:]))) < 1e-4
    assert abs(complex(np.mean(small[-262_144:]))) < 1e-4


# -- IQ imbalance ----------------------------------------------------------


def test_imbalance_correction_restores_image_rejection():
    clean = _carrier(RATE // 2, OFFSET)
    spoiled = _unbalance(clean, gain_db=1.5, phase_deg=4.0)

    corrected = _stream(IqBalance(RATE), spoiled)

    assert _image_rejection_db(spoiled) < 25.0
    assert _image_rejection_db(corrected) > 60.0


def test_imbalance_estimate_reports_the_error_that_was_planted():
    clean = _carrier(RATE // 2, OFFSET)
    spoiled = _unbalance(clean, gain_db=1.5, phase_deg=4.0)

    balance = IqBalance(RATE)
    _stream(balance, spoiled)

    assert balance.estimate.phase_deg == pytest.approx(4.0, abs=0.2)
    assert balance.estimate.gain_db == pytest.approx(1.5, abs=0.1)


def test_imbalance_leaves_a_clean_signal_alone():
    clean = _carrier(RATE // 4, OFFSET)
    corrected = _stream(IqBalance(RATE), clean)
    assert _image_rejection_db(corrected) > 100.0


# -- swap and shift --------------------------------------------------------


def test_swap_iq_mirrors_the_spectrum():
    clean = _carrier(RATE // 8, OFFSET)
    assert _image_rejection_db(clean) > 100.0
    assert _image_rejection_db(swap_iq(clean)) < -100.0


def test_shifter_moves_a_carrier_to_the_centre():
    clean = _carrier(RATE // 4, OFFSET)
    moved = _stream(FrequencyShifter(RATE, -OFFSET), clean)

    size = 65_536
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(moved[-size:] * np.hanning(size))))
    freqs = np.fft.fftshift(np.fft.fftfreq(size, 1 / RATE))
    assert abs(float(freqs[int(np.argmax(spectrum))])) < RATE / size


def test_shifter_keeps_its_phase_across_blocks():
    """A shifter restarting its oscillator each block buzzes at the block rate.

    Comparing against a one-shot run is the direct way to see it: a phase
    discontinuity would show up as a large difference at every boundary.
    """
    clean = _carrier(131_072, OFFSET)
    one_shot = FrequencyShifter(RATE, -137_000.0).process(clean)
    streamed = _stream(FrequencyShifter(RATE, -137_000.0), clean, block=4_096)
    np.testing.assert_allclose(streamed, one_shot, atol=1e-5)


# -- the chains ------------------------------------------------------------


def test_front_end_is_a_no_op_when_nothing_is_enabled():
    """The default install must not pay for the whole feature set."""
    front = FrontEnd(RATE)
    block = _carrier(4_096, OFFSET)
    assert not front.active
    assert front.process(block) is block


def test_front_end_applies_what_is_switched_on():
    front = FrontEnd(RATE)
    front.dc_removal = True
    front.iq_balance = True
    spoiled = _unbalance(_carrier(RATE // 4, OFFSET), 1.5, 4.0) + np.complex64(0.05)

    corrected = _stream(front, spoiled)

    assert front.active
    assert _image_rejection_db(corrected) > 60.0
    assert abs(complex(np.mean(corrected[-262_144:]))) < 1e-3


def test_front_end_offset_puts_the_wanted_signal_back_at_zero():
    """Offset tuning: the tuner moves up, so the software mixes back down."""
    front = FrontEnd(RATE)
    front.set_offset_hz(OFFSET)
    # The tuner is parked OFFSET high, so the wanted signal sits that far low.
    received = _carrier(RATE // 8, -OFFSET)

    moved = _stream(front, received)

    size = 65_536
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(moved[-size:] * np.hanning(size))))
    freqs = np.fft.fftshift(np.fft.fftfreq(size, 1 / RATE))
    assert abs(float(freqs[int(np.argmax(spectrum))])) < RATE / size


def test_audio_chain_applies_volume_and_limits():
    chain = AudioChain(48_000)
    chain.volume = 0.25
    loud = np.full(1_024, 8.0, dtype=np.float32)
    quiet = np.full(1_024, 0.4, dtype=np.float32)

    assert float(np.max(chain.process(loud))) == pytest.approx(1.0)
    assert float(np.max(chain.process(quiet))) == pytest.approx(0.1)


def test_audio_chain_mute_is_exact_silence():
    chain = AudioChain(48_000)
    chain.mute = True
    out = chain.process(np.full(1_024, 0.5, dtype=np.float32))
    assert not np.any(out)


def test_audio_chain_agc_runs_before_volume():
    """Volume must scale the AGC's output, not be swallowed by it.

    An AGC placed after the volume control would drive both settings to the
    same loudness and the slider would appear to do nothing.
    """
    t = np.arange(48_000 * 3) / 48_000
    audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    levels = []
    for volume in (0.2, 0.8):
        chain = AudioChain(48_000)
        chain.agc_enabled = True
        chain.volume = volume
        out = np.concatenate(
            [chain.process(audio[i : i + 1024]) for i in range(0, audio.size, 1024)]
        )
        levels.append(float(np.max(np.abs(out[-48_000:]))))

    assert levels[1] / levels[0] == pytest.approx(4.0, rel=0.1)


def test_audio_filter_removes_rumble_and_hiss_but_keeps_speech():
    """SDR#'s "filter audio": most of what makes a weak signal tiring to
    listen to lies outside the range that carries any of the voice."""
    rate = 48_000
    t = np.arange(rate) / rate
    for freq, expect_kept in ((80.0, False), (1_200.0, True), (9_000.0, False)):
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        chain = AudioChain(rate)
        chain.volume = 1.0
        chain.set_audio_filter(True)
        out = np.concatenate(
            [chain.process(tone[i : i + 1024]) for i in range(0, tone.size, 1024)]
        )
        kept = float(np.max(np.abs(out[-rate // 2 :]))) / 0.5
        assert (kept > 0.7) is expect_kept, f"{freq} Hz kept {kept:.3f}"


def test_audio_filter_is_off_by_default():
    chain = AudioChain(48_000)
    assert not chain.filter_audio
    assert chain.audio_filter_range == (300.0, 3_000.0)


# -- the audio chain in stereo ---------------------------------------------


def test_the_chain_leaves_a_stereo_block_in_stereo():
    chain = AudioChain(48_000)
    chain.volume = 0.5
    block = np.stack(
        [np.full(1_024, 0.4, np.float32), np.full(1_024, -0.2, np.float32)], axis=1
    )
    out = chain.process(block)
    assert out.shape == block.shape
    assert chain.keeps_stereo
    np.testing.assert_allclose(out[:, 0], 0.2, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], -0.1, atol=1e-6)


def test_one_agc_gain_covers_both_ears():
    """Two independent gain riders are a stereo image that wanders about.

    The left channel here is four times the right throughout, so any gain that
    is not shared shows up immediately as that ratio changing.
    """
    t = np.arange(48_000 * 3) / 48_000
    left = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    block = np.stack([left, left / 4.0], axis=1)

    chain = AudioChain(48_000)
    chain.agc_enabled = True
    chain.volume = 1.0
    out = np.concatenate(
        [chain.process(block[i : i + 1024]) for i in range(0, block.shape[0], 1024)]
    )
    tail = out[-48_000:]
    ratio = np.max(np.abs(tail[:, 0])) / np.max(np.abs(tail[:, 1]))
    assert ratio == pytest.approx(4.0, rel=0.02)


def test_the_audio_filter_runs_down_each_ear_and_not_across_them():
    """`lfilter` defaults to the trailing axis, which for a (frames, 2) block
    means filtering the left channel against the right - a mistake that comes
    out as a quiet click rather than as anything recognisable."""
    rate = 48_000
    t = np.arange(rate) / rate
    left = (0.5 * np.sin(2 * np.pi * 1_200.0 * t)).astype(np.float32)
    block = np.stack([left, np.zeros_like(left)], axis=1)

    chain = AudioChain(rate)
    chain.volume = 1.0
    chain.set_audio_filter(True)
    out = np.concatenate(
        [chain.process(block[i : i + 1024]) for i in range(0, block.shape[0], 1024)]
    )
    tail = out[-rate // 2 :]
    assert np.max(np.abs(tail[:, 0])) > 0.35
    assert np.max(np.abs(tail[:, 1])) < 1e-6


def test_audio_noise_reduction_mixes_down_and_says_so():
    """Spectral subtraction builds one noise estimate per channel, and two
    independent estimates pull a stereo image apart. It is a tool for a weak
    mono signal, so the chain mixes down rather than pretending otherwise -
    and `keeps_stereo` is how the indicator knows to go out."""
    chain = AudioChain(48_000)
    chain.set_noise_reduction(True)
    assert not chain.keeps_stereo

    block = np.stack(
        [np.full(4_096, 0.3, np.float32), np.full(4_096, 0.1, np.float32)], axis=1
    )
    out = chain.process(block)
    assert out.ndim == 1
