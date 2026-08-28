"""Tests for FM stereo, against a synthetic multiplex.

The transmitter below is the standard read forwards - 0.45 of the deviation to
the sum, 0.45 to the difference on a suppressed 38 kHz subcarrier, 0.10 to the
pilot - so a round trip through it exercises every step of the receiver
reading it backwards.

Separation is the measurement that matters, and it is the one that fails
quietly: a receiver whose subcarrier phase is 90 degrees out recovers no
difference channel at all and plays a perfectly clean mono broadcast, while
one whose two channels are half a millisecond apart sounds fine on speech and
hollow on music. Neither shows up in a test that only asks whether audio came
out, so every test here asks how *different* the two channels are.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import firwin, lfilter

from bettersdr.dsp import demod
from bettersdr.dsp.stereo import LOCK_DB, MIN_MPX_RATE_HZ, StereoDecoder

MPX_RATE = 240_000.0
SUBCARRIER_HZ = 38_000.0
PILOT_HZ = 19_000.0


def multiplex(
    left: np.ndarray,
    right: np.ndarray,
    rate: float = MPX_RATE,
    pilot: float = 0.10,
    noise: float = 0.0,
    seed: int = 5,
) -> np.ndarray:
    """The composite baseband a stereo transmitter frequency-modulates."""
    t = np.arange(left.size) / rate
    share = 0.5 * (1.0 - pilot)
    out = (
        share * (left + right)
        + share * (left - right) * np.sin(2 * np.pi * SUBCARRIER_HZ * t)
        + pilot * np.sin(2 * np.pi * PILOT_HZ * t)
    )
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, left.size)
    return out.astype(np.float32)


def tone(hz: float, seconds: float, rate: float = MPX_RATE, amp: float = 0.8):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def run(decoder: StereoDecoder, mpx: np.ndarray, block: int = 6_144):
    """Stream a multiplex through the decoder, returning the sum and difference."""
    sums, sides = [], []
    for start in range(0, mpx.size, block):
        total, side = decoder.process(mpx[start : start + block])
        sums.append(total)
        sides.append(np.zeros_like(total) if side is None else side)
    return np.concatenate(sums), np.concatenate(sides)


def baseband(signal: np.ndarray, rate: float = MPX_RATE) -> np.ndarray:
    """Everything below 15 kHz, which is what the audio stage would keep."""
    return lfilter(firwin(401, 15_000.0, fs=rate), 1.0, signal)


def level_db(signal: np.ndarray, hz: float, rate: float = MPX_RATE) -> float:
    windowed = signal * np.hanning(signal.size)
    bin_index = int(round(hz * signal.size / rate))
    return 20.0 * np.log10(max(abs(np.fft.rfft(windowed)[bin_index]), 1e-20))


def separation_db(left: np.ndarray, right: np.ndarray, hz: float, rate=MPX_RATE):
    """How far the silent channel sits below the one carrying the tone."""
    return level_db(left, hz, rate) - level_db(right, hz, rate)


# -- the decoder -----------------------------------------------------------


def test_a_tone_in_one_channel_stays_in_that_channel():
    """The headline measurement. Anything under about 20 dB is not stereo."""
    silence = np.zeros(int(MPX_RATE), dtype=np.float32)
    decoder = StereoDecoder(MPX_RATE)
    total, side = run(decoder, multiplex(tone(1_000.0, 1.0), silence))

    mid, difference = baseband(total)[-100_000:], baseband(side)[-100_000:]
    assert decoder.locked
    assert separation_db(mid + difference, mid - difference, 1_000.0) > 25.0


def test_the_channels_are_not_swapped():
    """A sign error in the subcarrier measures as perfect separation.

    It is the one failure this whole file could otherwise miss: every level is
    exactly where it should be, and the broadcast plays with its ears crossed.
    """
    silence = np.zeros(int(MPX_RATE), dtype=np.float32)
    total, side = run(StereoDecoder(MPX_RATE), multiplex(silence, tone(1_000.0, 1.0)))
    mid, difference = baseband(total)[-100_000:], baseband(side)[-100_000:]
    # The tone was put in the right channel, so the left is the quiet one.
    assert separation_db(mid - difference, mid + difference, 1_000.0) > 25.0


def test_the_sum_is_delayed_to_meet_its_own_pilot():
    """The delay is the whole reason this stage returns both halves.

    A caller that took the difference and used its own undelayed sum would
    have the two channels a group delay apart - which is inaudible on speech
    and hollows out anything with a cymbal in it.
    """
    decoder = StereoDecoder(MPX_RATE)
    mpx = multiplex(tone(1_000.0, 0.5), np.zeros(int(MPX_RATE * 0.5), np.float32))
    total, _ = run(decoder, mpx)
    assert decoder.delay > 0
    np.testing.assert_allclose(
        total[decoder.delay : 20_000], mpx[: 20_000 - decoder.delay], atol=1e-5
    )


def test_a_mono_station_is_not_reported_as_stereo():
    """There is noise at 19 kHz on every station; a pilot is more than that."""
    rate = MPX_RATE
    t = np.arange(int(rate)) / rate
    mono = (0.8 * np.sin(2 * np.pi * 1_000.0 * t)).astype(np.float32)
    mono = mono + np.random.default_rng(1).normal(0.0, 0.02, mono.size)

    decoder = StereoDecoder(rate)
    total, side = run(decoder, mono.astype(np.float32))

    assert not decoder.locked
    assert decoder.pilot_db < LOCK_DB
    assert not np.any(side)


def test_a_pilot_is_found_under_noise():
    left, right = tone(1_000.0, 1.0), tone(1_000.0, 1.0, amp=0.0)
    decoder = StereoDecoder(MPX_RATE)
    total, side = run(decoder, multiplex(left, right, noise=0.02))
    mid, difference = baseband(total)[-100_000:], baseband(side)[-100_000:]

    assert decoder.locked
    assert separation_db(mid + difference, mid - difference, 1_000.0) > 15.0


def test_block_size_does_not_change_what_is_decoded():
    """Both bandpasses and the delay line carry history across the boundary.

    The stream is chopped by USB transfers, not by anything stereo cares
    about. A filter that forgot its history each block would tick once per
    block - inaudible in a single-buffer test, obvious on air - and a delay
    line that restarted would put the two channels a whole block apart.

    The sum has to match exactly, because a delay line has nothing in it to
    settle. The difference is allowed a fraction of a percent: the pilot
    amplitude it is scaled by is smoothed over a fifth of a second, and a
    one-pole settling in ten steps does not land in precisely the same place
    as one settling in a single step. That shows up as a scale error of about
    0.4%, which is 48 dB of separation on its own and inaudible under the
    30-40 dB a real transmitter manages.
    """
    mpx = multiplex(tone(1_000.0, 1.0), np.zeros(int(MPX_RATE), np.float32))
    sums, channels = [], []
    for block in (1_024, 6_144, 48_000):
        total, side = run(StereoDecoder(MPX_RATE), mpx, block=block)
        mid, difference = baseband(total)[-100_000:], baseband(side)[-100_000:]
        assert separation_db(mid + difference, mid - difference, 1_000.0) > 25.0
        sums.append(total)
        channels.append(mid + difference)
    for other in sums[1:]:
        np.testing.assert_array_equal(sums[0], other)
    for other in channels[1:]:
        np.testing.assert_allclose(channels[0], other, atol=1e-2)


def test_reset_forgets_the_previous_station():
    decoder = StereoDecoder(MPX_RATE)
    run(decoder, multiplex(tone(1_000.0, 0.5), tone(1_000.0, 0.5, amp=0.0)))
    assert decoder.locked
    decoder.reset()
    assert not decoder.locked


def test_a_narrow_multiplex_is_refused_rather_than_decoded_badly():
    with pytest.raises(ValueError):
        StereoDecoder(MIN_MPX_RATE_HZ - 1.0)


def test_switching_the_decoder_off_leaves_the_sum_alone():
    """Off still means delayed: the caller must get one consistent timeline."""
    decoder = StereoDecoder(MPX_RATE)
    decoder.enabled = False
    mpx = multiplex(tone(1_000.0, 0.25), np.zeros(int(MPX_RATE * 0.25), np.float32))
    total, side = run(decoder, mpx)
    assert not np.any(side)
    assert total.size == mpx.size


# -- through the real demodulator ------------------------------------------


def test_a_broadcast_is_split_into_two_channels_by_the_demodulator():
    """The path the app actually uses: modulated RF in, two ears out."""
    rate = 2_400_000.0
    seconds = 0.5
    left = tone(1_000.0, seconds, rate=rate)
    right = np.zeros(left.size, dtype=np.float32)
    composite = multiplex(left, right, rate=rate)
    phase = np.cumsum(2.0 * np.pi * 75_000.0 * composite / rate)
    iq = np.exp(1j * phase).astype(np.complex64)

    demodulator = demod.WfmDemodulator(rate)
    demodulator.stereo = StereoDecoder(demodulator.if_rate)
    blocks = [
        demodulator.process(iq[start : start + 32_768])
        for start in range(0, iq.size, 32_768)
    ]
    audio = np.concatenate([block for block in blocks if block.ndim > 1])

    assert audio.ndim == 2 and audio.shape[1] == 2
    # De-emphasis has been applied to both channels by this point, so the two
    # levels are comparable but neither is at the level it went in at.
    tail = audio[-20_000:]
    assert separation_db(tail[:, 0], tail[:, 1], 1_000.0, rate=48_000.0) > 20.0


def test_a_mono_broadcast_still_comes_out_of_the_demodulator_as_mono():
    """No pilot means a one-dimensional block, which is what every stage
    downstream of the demodulator was written against."""
    rate = 2_400_000.0
    t = np.arange(int(rate * 0.3)) / rate
    composite = (0.8 * np.sin(2 * np.pi * 1_000.0 * t)).astype(np.float32)
    phase = np.cumsum(2.0 * np.pi * 75_000.0 * composite / rate)
    iq = np.exp(1j * phase).astype(np.complex64)

    demodulator = demod.WfmDemodulator(rate)
    demodulator.stereo = StereoDecoder(demodulator.if_rate)
    for start in range(0, iq.size, 32_768):
        assert demodulator.process(iq[start : start + 32_768]).ndim == 1
