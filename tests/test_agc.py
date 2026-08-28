"""Tests for the audio AGC.

The two claims worth pinning down are the ones a listener would notice going
wrong: that the output level stops depending on the input level, and that the
gain does not lunge upward the moment somebody stops talking.
"""

from __future__ import annotations

import numpy as np

from bettersdr.dsp.agc import Agc

RATE = 48_000


def _tone(seconds: float, amplitude: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stream(agc: Agc, audio: np.ndarray, block: int = 1024) -> np.ndarray:
    return np.concatenate(
        [agc.process(audio[i : i + block]) for i in range(0, audio.size, block)]
    )


def _peak_dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-12)))


def test_streaming_matches_one_shot():
    audio = np.concatenate([_tone(1.0, 0.02), _tone(1.0, 0.5), _tone(1.0, 0.02)])

    one_shot = Agc(RATE).process(audio)
    streamed = _stream(Agc(RATE), audio, block=997)

    assert streamed.size == one_shot.size
    np.testing.assert_array_equal(streamed, one_shot)


def test_loud_and_quiet_end_up_at_a_similar_level():
    """The whole point: two inputs 34 dB apart come out close together."""
    quiet = _stream(Agc(RATE), _tone(3.0, 0.01))
    loud = _stream(Agc(RATE), _tone(3.0, 0.5))

    settled = slice(-RATE, None)
    difference = abs(_peak_dbfs(quiet[settled]) - _peak_dbfs(loud[settled]))
    assert difference < 4.0


def test_output_lands_near_the_target():
    agc = Agc(RATE, target_dbfs=-12.0)
    out = _stream(agc, _tone(3.0, 0.3))
    assert -16.0 < _peak_dbfs(out[-RATE:]) < -8.0


def test_noise_below_the_threshold_is_not_amplified_to_full_scale():
    """A silent channel must not be turned into a roar.

    The gain is allowed to reach the ceiling the threshold implies and no
    further, so quiet input stays quiet in proportion.
    """
    agc = Agc(RATE, target_dbfs=-12.0, threshold_dbfs=-40.0)
    hiss = (
        np.random.default_rng(0).normal(0, 10 ** (-60 / 20), RATE * 3).astype(np.float32)
    )
    out = _stream(agc, hiss)
    assert _peak_dbfs(out[-RATE:]) < -12.0
    assert agc.gain_db <= agc.ceiling_db + 1e-6


def test_hang_holds_the_gain_through_a_gap():
    speech = _tone(1.0, 0.3)
    gap = np.zeros(int(RATE * 0.15), dtype=np.float32)

    with_hang = Agc(RATE, use_hang=True, hang_ms=250.0)
    _stream(with_hang, np.concatenate([speech, gap]))
    without = Agc(RATE, use_hang=False)
    _stream(without, np.concatenate([speech, gap]))

    assert with_hang.gain_db < without.gain_db


def test_slope_lets_the_output_follow_the_input():
    """Slope 10 dB per decade is no action; slope 0 is a flat output."""
    flat_quiet = _stream(Agc(RATE, slope_db=0.0), _tone(3.0, 0.02))
    flat_loud = _stream(Agc(RATE, slope_db=0.0), _tone(3.0, 0.4))
    sloped_quiet = _stream(Agc(RATE, slope_db=10.0), _tone(3.0, 0.02))
    sloped_loud = _stream(Agc(RATE, slope_db=10.0), _tone(3.0, 0.4))

    settled = slice(-RATE, None)
    flat_span = _peak_dbfs(flat_loud[settled]) - _peak_dbfs(flat_quiet[settled])
    sloped_span = _peak_dbfs(sloped_loud[settled]) - _peak_dbfs(sloped_quiet[settled])
    assert flat_span < 4.0
    assert sloped_span > 15.0


def test_gain_is_ramped_rather_than_stepped():
    """A gain switched once per control step would tick 750 times a second.

    Recovering the applied gain as output over input shows the ramp directly:
    with the interpolation in place no two adjacent samples differ by more
    than a fraction of a dB, and without it the gain would jump by whatever
    one control step decided.
    """
    audio = np.concatenate([_tone(0.3, 0.02), _tone(0.7, 0.6)])
    out = _stream(Agc(RATE, attack_ms=5.0), audio)

    # Only where the tone is well away from a zero crossing, so the ratio is
    # the gain rather than a division by nearly nothing, and only between
    # samples that were actually adjacent.
    source = audio[: out.size]
    index = np.flatnonzero(np.abs(source) > 0.5 * np.abs(source).max())
    gain_db = 20 * np.log10(np.abs(out[index]) / np.abs(source[index]))
    adjacent = np.diff(index) == 1
    assert float(np.max(np.abs(np.diff(gain_db)[adjacent]))) < 0.2


def test_reset_returns_to_unity():
    agc = Agc(RATE)
    _stream(agc, _tone(1.0, 0.5))
    agc.reset()
    assert agc.gain_db == 0.0
    assert agc.process(np.zeros(0, dtype=np.float32)).size == 0
