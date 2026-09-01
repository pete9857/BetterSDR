"""The audio chain's two halves, and why they are two.

`process` is what reaches the sound card. `body` is everything up to and
including the AGC, with volume and mute still to come - the last point at
which the audio is a property of the broadcast rather than of the session.
`audio/repro.py` records that one, because a folder of files recorded at
whatever the volume slider happened to be at is unusable.
"""

from __future__ import annotations

import numpy as np

from bettersdr.dsp.chain import AudioChain


def block(value: float = 0.4, frames: int = 1_024) -> np.ndarray:
    return np.full(frames, value, dtype=np.float32)


def test_process_is_the_body_and_then_the_output():
    chain = AudioChain()
    chain.volume = 0.3
    audio = block()
    assert np.allclose(chain.process(audio), chain.output(chain.body(audio)))


def test_the_volume_control_is_not_in_the_body():
    chain = AudioChain()
    chain.volume = 0.1
    audio = block()
    assert np.allclose(chain.body(audio), audio)
    assert np.allclose(chain.process(audio), audio * 0.1)


def test_muting_the_speakers_does_not_mute_the_body():
    """Otherwise an unattended session left muted is hours of zeros."""
    chain = AudioChain()
    chain.mute = True
    audio = block()
    assert np.allclose(chain.process(audio), 0.0)
    assert np.allclose(chain.body(audio), audio)


def test_the_limiter_belongs_to_the_output():
    chain = AudioChain()
    chain.volume = 1.0
    loud = block(4.0)
    assert np.allclose(chain.body(loud), loud)
    assert chain.process(loud).max() <= 1.0


def test_an_empty_block_survives_both_halves():
    chain = AudioChain()
    empty = np.zeros(0, dtype=np.float32)
    assert chain.body(empty).size == 0
    assert chain.output(empty).size == 0
    assert chain.process(empty).size == 0


def test_stereo_keeps_its_shape_through_both_halves():
    chain = AudioChain()
    chain.volume = 0.5
    stereo = np.stack([block(0.4), block(0.2)], axis=1)
    assert chain.body(stereo).shape == stereo.shape
    assert np.allclose(chain.process(stereo), stereo * 0.5)
