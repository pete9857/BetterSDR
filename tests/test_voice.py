"""What a channel sounded like: voice, music, a tone, data or static.

The point of these tests is the *separation*, not any one verdict. Every
threshold in `scan/voice.py` was set from the numbers this material produces,
and a change that moves one of them is a change that will misreport somebody's
radio traffic - so each case is asserted from both sides where it can be.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp import demod
from bettersdr.scan import voice
from tests import synth_audio as sa

RATE = 2_400_000


def _fm(audio, deviation_hz=2_500.0, rate=RATE, snr_db=30.0, seed=0):
    """Put 48 kHz audio on an FM carrier, so the whole path can be driven."""
    up = int(rate // sa.RATE)
    block = np.repeat(np.asarray(audio, dtype=np.float64), up)
    smooth = np.hanning(up * 2 + 1)
    block = np.convolve(block, smooth / smooth.sum(), mode="same")
    phase = 2 * np.pi * deviation_hz * np.cumsum(block) / rate
    rng = np.random.default_rng(seed)
    amplitude = 10 ** (snr_db / 20.0)
    noise = (rng.normal(0, 1, block.size) + 1j * rng.normal(0, 1, block.size)) / np.sqrt(
        2
    )
    return ((amplitude * np.exp(1j * phase) + noise) / (amplitude + 1) * 0.5).astype(
        np.complex64
    )


def _through_nfm(iq, rate=RATE):
    """Demodulate in blocks, exactly as the engine does. Never one-shot."""
    receiver = demod.create("nfm", float(rate), volume=1.0)
    receiver.clip = False
    step = 1 << 16
    return np.concatenate(
        [receiver.process(iq[i : i + step]) for i in range(0, iq.size, step)]
    )


# -- the five things it can hear --------------------------------------------


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: sa.speech(0.8, pitch_hz=120, seed=1), voice.VOICE),
        (lambda: sa.speech(0.8, pitch_hz=210, seed=2), voice.VOICE),
        (lambda: sa.speech(0.8, syllables_per_second=2.5, seed=3), voice.VOICE),
        (lambda: sa.speech(0.8, cutoff_hz=8_000, seed=4), voice.VOICE),
        (lambda: sa.music(0.8, seed=1), voice.MUSIC),
        (lambda: sa.music(0.8, seed=5), voice.MUSIC),
        (lambda: sa.tone(0.8, hz=1_000), voice.TONE),
        (lambda: sa.tone(0.8, hz=440), voice.TONE),
        (lambda: sa.data(0.8, baud=512, seed=3), voice.DATA),
        (lambda: sa.data(0.8, baud=1_200, seed=1), voice.DATA),
        (lambda: sa.data(0.8, baud=2_400, seed=2), voice.DATA),
        (lambda: sa.static(0.8, seed=1), voice.NOISE),
        (lambda: sa.static(0.8, cutoff_hz=15_000, seed=2), voice.NOISE),
        (lambda: sa.silence(0.8), voice.SILENCE),
    ],
)
def test_each_kind_is_recognised(make, expected):
    assert voice.classify(make()).kind == expected


def test_a_pager_is_not_mistaken_for_a_person():
    """The one confusion that would actually matter on air.

    A pager and a conversation sit on identical-looking channels - narrow,
    constant, in the same allocation - so the power spectrum has nothing to
    say about which is which. If this ever passes for voice the monitor stops
    on every paging transmitter in the county.
    """
    for baud in (512.0, 1_200.0, 2_400.0):
        verdict = voice.classify(sa.data(1.0, baud=baud, seed=7))
        assert verdict.kind == voice.DATA
        assert not verdict.is_voice
        assert not verdict.carries_audio


def test_static_is_not_mistaken_for_data():
    """An open squelch is the most common thing a scanner ever hears."""
    verdict = voice.classify(sa.static(1.0, seed=11))
    assert verdict.kind == voice.NOISE
    assert not verdict.carries_audio


# -- what the monitor asks it -----------------------------------------------


def test_only_voice_and_music_are_worth_stopping_for():
    assert voice.classify(sa.speech(0.8, seed=1)).carries_audio
    assert voice.classify(sa.music(0.8, seed=1)).carries_audio
    assert not voice.classify(sa.data(0.8, seed=1)).carries_audio
    assert not voice.classify(sa.static(0.8, seed=1)).carries_audio
    assert not voice.classify(sa.tone(0.8)).carries_audio
    assert not voice.classify(sa.silence(0.8)).carries_audio


def test_music_is_not_reported_as_somebody_talking():
    """`is_voice` drives the badge and the revisit policy, so it is stricter."""
    verdict = voice.classify(sa.music(0.8, seed=1))
    assert verdict.carries_audio
    assert not verdict.is_voice


# -- the features the thresholds are set from -------------------------------


def test_flatness_separates_static_from_a_bit_stream():
    """Measured over the occupied band, never over the whole 24 kHz.

    Over the full range everything that has been through an audio filter
    measures zero, because the empty bins above the cutoff drive a geometric
    mean to nothing. That reads as a working feature and is not one.
    """
    noise = voice.measure(sa.static(0.8, seed=1)).flatness
    stream = voice.measure(sa.data(0.8, baud=1_200, seed=1)).flatness
    assert noise > voice.NOISE_FLATNESS > stream
    assert noise > 0.7
    assert stream < 0.3


def test_a_bit_stream_finds_a_pitch_but_never_the_same_one_twice():
    """The measurement that actually separates a pager from a person.

    A random bit stream repeats strongly inside most frames, so its voiced
    fraction looks like speech; what it cannot do is find the same lag twice.
    """
    stream = voice.measure(sa.data(0.8, baud=1_200, seed=1))
    talk = voice.measure(sa.speech(0.8, seed=1))
    assert stream.voiced > 0.3
    assert stream.pitch_spread_hz > voice.PITCH_WANDER_HZ
    assert not stream.has_pitch
    assert talk.has_pitch


def test_speech_swings_and_nothing_else_does():
    talk = voice.measure(sa.speech(0.8, seed=1)).modulation_db
    for quiet in (sa.music(0.8, seed=1), sa.data(0.8, seed=1), sa.static(0.8, seed=1)):
        assert voice.measure(quiet).modulation_db < voice.VOICE_MODULATION_DB < talk


def test_the_syllable_rhythm_is_where_speech_puts_it():
    assert voice.measure(sa.speech(0.8, seed=1)).syllabic >= voice.VOICE_SYLLABIC


# -- honesty ----------------------------------------------------------------


def test_too_short_a_clip_is_refused_rather_than_guessed_at():
    """Below `MIN_SECONDS` the syllable band does not exist to be measured."""
    verdict = voice.classify(sa.speech(0.2, seed=1))
    assert verdict.kind == voice.UNCLEAR
    assert not verdict.certain


def test_a_short_clip_is_answered_but_never_called_certain():
    verdict = voice.classify(sa.speech(0.45, seed=1))
    assert verdict.kind == voice.VOICE
    assert not verdict.certain


def test_a_full_clip_is_certain():
    assert voice.classify(sa.speech(0.8, seed=1)).certain


def test_empty_audio_does_not_raise():
    verdict = voice.classify(np.zeros(0, dtype=np.float32))
    assert verdict.kind == voice.UNCLEAR


def test_stereo_is_mixed_down_rather_than_picked_from():
    """A hard-panned broadcast must not read as silence on one side."""
    mono = sa.speech(0.8, seed=1)
    stereo = np.stack([mono, np.zeros_like(mono)], axis=1)
    assert voice.classify(stereo).kind == voice.VOICE


def test_every_verdict_says_why():
    for make in (sa.speech, sa.music, sa.tone, sa.data, sa.static, sa.silence):
        verdict = voice.classify(make(0.8))
        assert verdict.reasons
        assert all(reason for reason in verdict.reasons)
        assert verdict.label


# -- through the receiver the app actually uses -----------------------------


@pytest.mark.parametrize("snr_db", [30.0, 12.0])
@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: sa.speech(1.0, seed=1), voice.VOICE),
        (lambda: sa.data(1.0, baud=1_200, seed=1), voice.DATA),
        (lambda: sa.tone(1.0, hz=1_000), voice.TONE),
        (lambda: sa.static(1.0, seed=1), voice.NOISE),
    ],
)
def test_the_verdict_survives_the_whole_demodulator(make, expected, snr_db):
    """Block by block through the real NFM path at 2.4 MS/s, not one-shot.

    Everything above analyses audio somebody handed over. This is the only
    test that asks whether the thing the engine will actually pass in - a
    channel filter, a discriminator, a 4 kHz audio filter and a decimation
    chain with history across blocks - still produces an answer.
    """
    audio = _through_nfm(_fm(make(), snr_db=snr_db, seed=3))
    assert voice.classify(audio, float(demod.AUDIO_RATE)).kind == expected
