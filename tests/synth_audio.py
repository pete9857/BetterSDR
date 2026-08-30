"""Synthetic audio for the voice detector, at the point a demodulator hands it over.

None of this is a recording. Each generator makes the *shape* the detector
actually keys on - a syllable rhythm, a wandering pitch, a bit stream's
constant loudness - so a threshold that moves shows up as a test that fails
rather than as a station that stops being recognised months later.

Everything comes out mono float32 at 48 kHz, band-limited the way the
demodulator that produced it would have been: 4 kHz for narrow FM, 15 kHz for
broadcast.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

RATE = 48_000


def _lowpass(audio: np.ndarray, cutoff_hz: float, rate: float = RATE) -> np.ndarray:
    sos = butter(6, cutoff_hz / (rate / 2.0), btype="low", output="sos")
    return sosfilt(sos, audio)


def _bandpass(
    audio: np.ndarray, low_hz: float, high_hz: float, rate: float = RATE
) -> np.ndarray:
    sos = butter(
        4, [low_hz / (rate / 2.0), high_hz / (rate / 2.0)], btype="band", output="sos"
    )
    return sosfilt(sos, audio)


def _normalise(audio: np.ndarray, dbfs: float = -12.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio.astype(np.float32)
    return (audio / peak * (10.0 ** (dbfs / 20.0))).astype(np.float32)


def _resonator(
    audio: np.ndarray, hz: float, q: float, rate: float = RATE
) -> np.ndarray:
    """One formant. A narrow resonance is what turns a buzz into a vowel."""
    width = hz / q
    nyquist = rate / 2.0
    low = max(hz - width / 2, 20.0) / nyquist
    high = min(hz + width / 2, nyquist - 100.0) / nyquist
    b, a = butter(2, [low, high], btype="band")
    return lfilter(b, a, audio)


def speech(
    seconds: float = 1.0,
    rate: float = RATE,
    pitch_hz: float = 130.0,
    syllables_per_second: float = 4.0,
    cutoff_hz: float = 3_400.0,
    seed: int = 0,
) -> np.ndarray:
    """A voice: a wandering pitch, three formants, and syllables with gaps.

    The gaps are the point. Everything else on the air can be pitched or can
    sit in the voice band; only speech stops four times a second.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate

    # Pitch wanders, the way nobody holds a note while speaking.
    wander = 1.0 + 0.12 * np.sin(2 * np.pi * 1.3 * t) + 0.05 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(pitch_hz * wander) / rate
    # A pulse train, which is what a glottis actually produces.
    glottal = np.zeros(n)
    for harmonic in range(1, 40):
        if harmonic * pitch_hz > cutoff_hz:
            break
        glottal += np.sin(harmonic * phase) / harmonic

    voiced = (
        1.0 * _resonator(glottal, 700.0, 8.0, rate)
        + 0.6 * _resonator(glottal, 1220.0, 9.0, rate)
        + 0.3 * _resonator(glottal, 2600.0, 10.0, rate)
    )
    # Unvoiced fricatives, which real speech is about a quarter of.
    hiss = _bandpass(rng.normal(0.0, 1.0, n), 2_500.0, cutoff_hz, rate)

    # The syllable envelope: on for most of a beat, off between words.
    beat = rate / syllables_per_second
    index = np.arange(n) / beat
    shape = 0.5 - 0.5 * np.cos(2 * np.pi * np.clip(index % 1.0, 0.0, 1.0))
    words = rng.random(int(np.ceil(index[-1])) + 2) > 0.25
    gate = words[index.astype(int)] * shape
    fricative = rng.random(words.size) > 0.7
    mix = np.where(fricative[index.astype(int)], 0.25, 1.0)

    audio = gate * (mix * voiced + (1.0 - mix) * 2.0 * hiss)
    audio = _bandpass(audio, 300.0, cutoff_hz, rate)
    # A real channel is never silent between words.
    audio = audio + 0.004 * rng.normal(0.0, 1.0, n)
    return _normalise(audio)


def music(
    seconds: float = 1.0,
    rate: float = RATE,
    cutoff_hz: float = 12_000.0,
    seed: int = 0,
) -> np.ndarray:
    """Sustained harmony: pitched throughout, no pauses, bright on top."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    audio = np.zeros(n)
    # A chord every half second, held rather than articulated.
    beat = 0.5
    for start in np.arange(0.0, seconds, beat):
        root = float(rng.choice([146.8, 164.8, 196.0, 220.0, 246.9]))
        window = (t >= start) & (t < start + beat)
        span = t[window] - start
        # A slow attack and no decay to silence: the note is still sounding
        # when the next one starts, which is what makes music unlike speech.
        envelope = 1.0 - np.exp(-span * 20.0)
        for degree in (1.0, 1.25, 1.5, 2.0):
            for harmonic in range(1, 12):
                hz = root * degree * harmonic
                if hz > cutoff_hz:
                    break
                audio[window] += (
                    envelope * np.sin(2 * np.pi * hz * (t[window])) / (harmonic**1.2)
                )
    # Cymbals and consonant noise: the top octave a voice never reaches.
    audio += 0.15 * _bandpass(rng.normal(0.0, 1.0, n), 6_000.0, cutoff_hz, rate)
    return _normalise(audio)


def tone(
    seconds: float = 1.0, rate: float = RATE, hz: float = 1_000.0, seed: int = 0
) -> np.ndarray:
    """One note, forever. A dead carrier with a test tone on it, or a repeater."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    return _normalise(np.sin(2 * np.pi * hz * t) + 0.01 * rng.normal(0.0, 1.0, n))


def data(
    seconds: float = 1.0,
    rate: float = RATE,
    baud: float = 1_200.0,
    cutoff_hz: float = 3_400.0,
    seed: int = 0,
) -> np.ndarray:
    """A bit stream off an FM discriminator: constant, unpitched, wideband.

    This is what POCSAG looks like at the point the audio path sees it, and
    it is the case the detector most has to get right - a pager transmission
    and somebody talking sit on identical-looking channels.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    samples_per_bit = rate / baud
    bits = rng.integers(0, 2, int(np.ceil(n / samples_per_bit)) + 1) * 2.0 - 1.0
    index = (np.arange(n) / samples_per_bit).astype(int)
    audio = bits[index]
    audio = _lowpass(audio, cutoff_hz, rate)
    return _normalise(audio + 0.01 * rng.normal(0.0, 1.0, n))


def static(
    seconds: float = 1.0, rate: float = RATE, cutoff_hz: float = 4_000.0, seed: int = 0
) -> np.ndarray:
    """An open squelch on an empty channel: flat, unpitched, unstructured."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    return _normalise(_lowpass(rng.normal(0.0, 1.0, n), cutoff_hz, rate))


def silence(seconds: float = 1.0, rate: float = RATE, seed: int = 0) -> np.ndarray:
    """An unmodulated carrier: the discriminator has nothing to report."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    return (0.0004 * rng.normal(0.0, 1.0, n)).astype(np.float32)


__all__ = ["RATE", "data", "music", "silence", "speech", "static", "tone"]
