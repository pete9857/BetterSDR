"""Telling somebody talking apart from a tone, a data burst and plain static.

A power spectrum cannot answer this. The sweep measures a channel for fifty
milliseconds and learns how wide it is and how far above the noise it stands,
and none of that separates a conversation from a pager transmission - both are
narrow, both are constant, and `dsp/features.py` already records that spectral
flatness cannot even separate analog broadcast FM from a digital signal. So
this module works on the *audio*: the monitor parks on a busy channel for a
moment, demodulates it exactly as the listening screen would, and asks what
came out.

Five things can come out, and each is recognised by something a person would
also notice:

* **Somebody talking.** Speech starts and stops. Its loudness swings six to
  twelve decibels several times a second - syllables - and underneath that it
  has a pitch, which wanders. Nothing else on the air does both.
* **Music.** Also pitched, also in the voice band, but it does not pause the
  way a sentence does and it reaches much further up the audio range.
* **A tone.** One frequency, one loudness, forever.
* **Data.** Constant loudness like a tone, but spread across the channel
  rather than sitting on one note - a pager, a telemetry burst, a digital
  voice mode this app cannot decode.
* **Static.** Flat, unpitched, unstructured. What an empty channel sounds
  like when the squelch is open.

The rules are rules rather than a model, for the same reason
`scan/classifier.py`'s are: the app has to be able to say *why*, and "the
loudness swings 9 dB about four times a second and there is a pitch around
140 Hz" is a sentence a beginner can check against their own ears. Where the
evidence does not support any of the five, the answer is "not sure", which is
a perfectly good thing to say and much cheaper than being confidently wrong
about somebody's radio traffic.

No Qt and no device in here: audio in, a verdict out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The frame the spectrum and the pitch are measured on. 2048 samples is 43 ms
# at 48 kHz, which is two whole cycles of the lowest pitch a human voice
# reaches - a shorter frame cannot see a 70 Hz fundamental at all, and pitch
# is half the evidence for speech.
FRAME = 2048
# A quarter of a frame. The frame energies this produces are the envelope, at
# 94 per second, which has to be fast enough to show a syllable rhythm.
HOP = 512

# Below this there is nothing to analyse. An FM discriminator handed an
# unmodulated carrier puts out almost nothing, so this is also how a dead
# carrier is recognised - which is why it is "silent", not "too quiet".
SILENCE_DBFS = -55.0

# How much audio is needed before any of this means anything. The syllable
# rhythm is measured between 2 and 8 Hz, and a clip shorter than this cannot
# resolve that band at all - the answer would be arithmetic on two numbers.
MIN_SECONDS = 0.35
# Below this the answer is given but never presented as certain.
TRUSTED_SECONDS = 0.6

# Where a human voice lives. The 300-3400 Hz telephone band is not a
# convention picked out of the air: it is what every two-way radio in the
# world filters its microphone to.
VOICE_LOW_HZ = 300.0
VOICE_HIGH_HZ = 3400.0
# Below the voice band. A bit stream's slow wander lands here and speech does
# not, because the transmitter high-passed it away.
SUB_VOICE_HZ = 250.0

# The envelope band a syllable rhythm occupies, against the whole range the
# envelope is measured over.
SYLLABLE_LOW_HZ = 2.0
SYLLABLE_HIGH_HZ = 8.0
ENVELOPE_LOW_HZ = 0.5
ENVELOPE_HIGH_HZ = 20.0

# The pitch range of a human voice, from a deep male fundamental to a high
# female one.
PITCH_LOW_HZ = 70.0
PITCH_HIGH_HZ = 350.0
# How strongly a frame has to repeat itself at some lag in that range before
# it counts as pitched. Measured on the normalised autocorrelation, so 1.0 is
# a perfect repeat and 0 is noise.
PITCHED = 0.30

# -- the thresholds the rules turn on ---------------------------------------
# All measured against the synthetic material in `tests/test_voice.py`, which
# is the only place any of them can be re-derived.

# One bin holding this much of the audio's power is a tone, not a signal.
TONE_PEAK = 0.35
# ...and a tone does not move. A voice's pitch wanders by tens of hertz
# within a single word.
TONE_PITCH_SPREAD_HZ = 8.0

# How far a real pitch may wander between frames and still be one pitch,
# measured as the spread of the middle half of the readings. This is the
# threshold that separates a pager from a person, and it is not the obvious
# one: a random bit stream *does* find a strong repeat in every frame, so its
# voiced fraction looks like speech. What it cannot do is find the same one
# twice - measured at 114 to 142 Hz of spread against 4 to 18 Hz for speech
# and 0 to 13 Hz for music.
PITCH_WANDER_HZ = 60.0
# How many frames have to be pitched before "it has a pitch" is a fair thing
# to say about the whole clip.
PITCHED_SHARE = 0.35

# Loudness that swings this much, in decibels, is starting and stopping.
VOICE_MODULATION_DB = 4.0
# ...and speech does it at a syllable rate, which is what separates it from a
# signal that merely fades. This is the share of the envelope's movement that
# falls between 2 and 8 Hz.
VOICE_SYLLABIC = 0.30
# Most of the sound has to be in the voice band for it to be a voice.
VOICE_BAND_SHARE = 0.55

# Above this the audio is as featureless as noise across its whole range.
# Measured across the band the audio actually occupies rather than across the
# whole 24 kHz: everything here has been through an audio filter that stops at
# 4 or 15 kHz, and the empty bins above it drive a geometric mean to zero
# whatever is underneath. Over its own band, static measures 0.87 to 0.97, a
# bit stream 0.13 to 0.25, speech and music 0.01 to 0.07 and a tone 0.000.
NOISE_FLATNESS = 0.42
# A channel this steady is not somebody talking, whatever else it is.
STEADY_MODULATION_DB = 3.0

# Music holds its pitch through almost every frame, where speech can only be
# pitched between the pauses that make it speech.
MUSIC_PITCHED_SHARE = 0.50

VOICE = "voice"
MUSIC = "music"
TONE = "tone"
DATA = "data"
NOISE = "noise"
SILENCE = "silence"
UNCLEAR = "unclear"

# What each verdict is called on screen, and the glyph that goes with it.
# Worded as what a listener would hear, never as what the DSP measured.
LABELS: dict[str, tuple[str, str]] = {
    VOICE: ("Voice", "walkie"),
    MUSIC: ("Music", "music"),
    TONE: ("Steady tone", "wave"),
    DATA: ("Data", "chip"),
    NOISE: ("Static", "question"),
    SILENCE: ("Silent", "wave"),
    UNCLEAR: ("Not sure", "question"),
}

CONFIDENCE_CLEAR = 0.85
CONFIDENCE_LIKELY = 0.65
# The ceiling on anything measured over less than `TRUSTED_SECONDS`. Below
# `TRUSTED`, deliberately: a clip too short to hold two syllables can still be
# right and must never be presented as a fact.
CONFIDENCE_SHORT = 0.55
CONFIDENCE_GUESS = 0.35
TRUSTED = 0.65


@dataclass(frozen=True)
class Features:
    """Everything measured about a clip, before anything decides what it is.

    Kept on the verdict rather than thrown away, because the Expert view
    shows them and because a threshold nobody can see the other side of is a
    threshold nobody can re-derive.
    """

    seconds: float
    level_dbfs: float
    # Share of the audio's power in one FFT bin. A tone is one bin.
    peak_fraction: float
    # Geometric over arithmetic mean of the spectrum: 1.0 is featureless.
    flatness: float
    # Share of the power between 300 and 3400 Hz.
    voice_band: float
    # Share below 250 Hz, where a bit stream's wander lives and speech does
    # not, because the transmitter filtered it out.
    sub_voice: float
    # Where 99% of the audio power is below. Also the band the flatness above
    # is measured across - see `NOISE_FLATNESS`.
    bandwidth_hz: float
    # Standard deviation of the frame loudness, in dB. Speech pauses.
    modulation_db: float
    # Share of the envelope's movement in the 2-8 Hz syllable band.
    syllabic: float
    # Share of frames that repeat themselves at a voice-like lag.
    voiced: float
    pitch_hz: float
    pitch_spread_hz: float

    @property
    def has_pitch(self) -> bool:
        """Whether one pitch runs through this, rather than a new one a frame.

        Both halves are needed and the second is the one that does the work.
        A random bit stream finds a strong repeat in most of its frames - its
        voiced fraction reads like speech - but it finds a *different* one
        each time, so the spread of the readings is the measurement that
        actually separates a pager from a person.
        """
        return (
            self.voiced >= PITCHED_SHARE
            and self.pitch_spread_hz <= PITCH_WANDER_HZ
        )


@dataclass(frozen=True)
class Verdict:
    """What the channel sounded like, and why."""

    kind: str
    confidence: float
    reasons: tuple[str, ...]
    features: Features

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, LABELS[UNCLEAR])[0]

    @property
    def icon(self) -> str:
        return LABELS.get(self.kind, LABELS[UNCLEAR])[1]

    @property
    def certain(self) -> bool:
        """Whether to say this as a fact rather than as a best guess."""
        return self.confidence >= TRUSTED

    @property
    def is_voice(self) -> bool:
        """Whether somebody was talking. Music is not somebody talking."""
        return self.kind == VOICE

    @property
    def carries_audio(self) -> bool:
        """Whether there is something on this channel worth hearing.

        The question the monitor asks before deciding to stop on a channel:
        static, silence and a bare tone are not worth interrupting a sweep
        for, and data is not worth listening *to* however interesting it is
        to know it is there.
        """
        return self.kind in (VOICE, MUSIC)

    @property
    def explanation(self) -> str:
        return f"{', '.join(self.reasons)} -> {self.label}"


def _mono(audio: np.ndarray) -> np.ndarray:
    """One channel, DC removed. Stereo is mixed down, never picked from.

    Taking the left channel alone would read a hard-panned broadcast as
    silence, and the difference channel of a fringe FM station is mostly
    noise - so the sum is both the safer answer and the one a listener hears.
    """
    block = np.asarray(audio, dtype=np.float64)
    if block.ndim > 1:
        block = block.mean(axis=1)
    block = np.ascontiguousarray(block)
    if block.size == 0:
        return block
    return block - block.mean()


def _frames(audio: np.ndarray) -> np.ndarray:
    """Overlapping analysis frames as a 2-D view, without copying the audio."""
    if audio.size < FRAME:
        return np.zeros((0, FRAME), dtype=np.float64)
    count = 1 + (audio.size - FRAME) // HOP
    return np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, FRAME),
        strides=(audio.strides[0] * HOP, audio.strides[0]),
        writeable=False,
    )


def _pitch(power: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame pitch, from the autocorrelation the spectrum already paid for.

    The autocorrelation of a frame is the inverse transform of its own power
    spectrum, so this is one more transform rather than a search: no lag loop,
    no Python over samples. The bias correction matters - a plain
    autocorrelation falls away towards long lags simply because fewer samples
    overlap there, which pushes every estimate towards the top of the range.
    """
    acf = np.fft.irfft(power, n=FRAME, axis=1)
    overlap = FRAME - np.arange(FRAME)
    acf = acf / np.maximum(overlap, 1)
    zero = np.maximum(acf[:, :1], 1e-30)
    acf = acf / zero
    low = max(1, int(round(rate / PITCH_HIGH_HZ)))
    high = min(FRAME - 1, int(round(rate / PITCH_LOW_HZ)))
    if high <= low:
        count = acf.shape[0]
        return np.zeros(count), np.zeros(count)
    window = acf[:, low : high + 1]
    best = np.argmax(window, axis=1)
    return window[np.arange(window.shape[0]), best], rate / (low + best)


def _envelope(energy: np.ndarray, rate: float) -> tuple[float, float]:
    """How much the loudness moves, and how much of that is a syllable rhythm.

    The rhythm matters as much as the depth. Static swings a decibel or two at
    random, and a station fading swings a lot but slowly; speech does it three
    to six times a second, which is what the 2-8 Hz share measures. Below
    `MIN_SECONDS` there are not enough envelope samples for that band to exist
    and the share comes back as zero rather than as a number nobody can trust.
    """
    if energy.size < 4:
        return 0.0, 0.0
    level = 10.0 * np.log10(np.maximum(energy, 1e-30))
    depth = float(np.std(level))
    frame_rate = rate / HOP
    centred = level - level.mean()
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    bins = np.fft.rfftfreq(centred.size, d=1.0 / frame_rate)
    whole = (bins >= ENVELOPE_LOW_HZ) & (bins <= ENVELOPE_HIGH_HZ)
    total = float(spectrum[whole].sum())
    if total <= 0.0:
        return depth, 0.0
    band = (bins >= SYLLABLE_LOW_HZ) & (bins <= SYLLABLE_HIGH_HZ)
    return depth, float(spectrum[band].sum() / total)


def measure(audio: np.ndarray, rate: float = 48_000.0) -> Features:
    """Everything the rules below need, measured once."""
    block = _mono(audio)
    seconds = block.size / float(rate)
    rms = float(np.sqrt(np.mean(block**2))) if block.size else 0.0
    level = 20.0 * np.log10(max(rms, 1e-9))

    frames = _frames(block)
    if frames.shape[0] == 0:
        return Features(
            seconds=seconds,
            level_dbfs=level,
            peak_fraction=0.0,
            flatness=0.0,
            voice_band=0.0,
            sub_voice=0.0,
            bandwidth_hz=0.0,
            modulation_db=0.0,
            syllabic=0.0,
            voiced=0.0,
            pitch_hz=0.0,
            pitch_spread_hz=0.0,
        )

    window = np.hanning(FRAME)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    energy = power.sum(axis=1)
    mean_power = power.mean(axis=0)
    bins = np.fft.rfftfreq(FRAME, d=1.0 / rate)

    # Everything below is measured above 100 Hz. What sits underneath is the
    # de-emphasis and the discriminator's own drift, which belongs to the
    # receiver rather than to whatever is transmitting.
    usable = bins >= 100.0
    spectrum = mean_power[usable]
    frequencies = bins[usable]
    total = float(spectrum.sum())
    if total <= 0.0:
        total = 1e-30

    peak_fraction = float(spectrum.max() / total)
    voice_band = float(
        spectrum[(frequencies >= VOICE_LOW_HZ) & (frequencies <= VOICE_HIGH_HZ)].sum()
        / total
    )
    sub_voice = float(spectrum[frequencies < SUB_VOICE_HZ].sum() / total)
    cumulative = np.cumsum(spectrum) / total
    bandwidth = float(
        frequencies[
            min(int(np.searchsorted(cumulative, 0.99)), frequencies.size - 1)
        ]
    )

    # Flatness across the band the audio actually occupies, never across the
    # whole 24 kHz. Everything reaching here has been through an audio filter
    # that stops at 4 kHz or 15 kHz, and a geometric mean including the empty
    # bins above it comes out at zero for static and speech alike - which is
    # to say, it stops measuring anything at all.
    occupied = (frequencies >= 100.0) & (frequencies <= max(bandwidth, 500.0))
    inside = spectrum[occupied]
    if inside.size == 0:
        inside = spectrum
    logs = np.log(np.maximum(inside, 1e-30))
    flatness = float(np.exp(logs.mean()) / max(inside.mean(), 1e-30))

    modulation_db, syllabic = _envelope(energy, rate)
    strength, pitches = _pitch(power, rate)
    # Only the frames that were loud enough to have a pitch at all. A pause
    # between two words is noise, and noise finds a lag like anything else.
    loud = energy > 0.1 * float(np.max(energy)) if energy.size else energy > 0
    pitched = (strength >= PITCHED) & loud
    voiced = float(pitched.mean()) if pitched.size else 0.0
    if pitched.any():
        found = pitches[pitched]
        pitch_hz = float(np.median(found))
        # The spread of the middle of the distribution rather than of all of
        # it: one frame that locked onto a harmonic should not make a steady
        # tone look like a voice.
        pitch_spread = float(np.percentile(found, 75) - np.percentile(found, 25))
    else:
        pitch_hz, pitch_spread = 0.0, 0.0

    return Features(
        seconds=seconds,
        level_dbfs=level,
        peak_fraction=peak_fraction,
        flatness=flatness,
        voice_band=voice_band,
        sub_voice=sub_voice,
        bandwidth_hz=bandwidth,
        modulation_db=modulation_db,
        syllabic=syllabic,
        voiced=voiced,
        pitch_hz=pitch_hz,
        pitch_spread_hz=pitch_spread,
    )


def _rate_phrase(features: Features) -> str:
    """The syllable rhythm as somebody would describe it out loud."""
    return f"its loudness swings {features.modulation_db:.0f} dB at a syllable rate"


def classify(audio: np.ndarray, rate: float = 48_000.0) -> Verdict:
    """Decide what a demodulated channel sounded like.

    The order of the tests is the order of confidence, not the order of
    interest. Silence and a bare tone are unmistakable and are settled first;
    speech is asked about before data because the two look alike from a
    distance and speech has positive evidence - a rhythm and a pitch - where
    data is mostly the absence of things.
    """
    features = measure(audio, rate)
    short = features.seconds < TRUSTED_SECONDS
    unusable = features.seconds < MIN_SECONDS

    def verdict(kind: str, confidence: float, *reasons: str) -> Verdict:
        if unusable:
            return Verdict(
                kind=UNCLEAR,
                confidence=CONFIDENCE_GUESS,
                reasons=(
                    f"only {features.seconds:.1f} s of sound to go on, which "
                    f"is not enough to hear a rhythm in",
                ),
                features=features,
            )
        # A short clip can still be right; it just cannot be certain, and the
        # badge on the card is driven off exactly this.
        return Verdict(
            kind=kind,
            confidence=min(confidence, CONFIDENCE_SHORT) if short else confidence,
            reasons=tuple(reasons),
            features=features,
        )

    if features.level_dbfs < SILENCE_DBFS:
        return verdict(
            SILENCE,
            CONFIDENCE_CLEAR,
            f"nothing coming out of it at all ({features.level_dbfs:.0f} dBFS)",
            "a carrier with nothing on it sounds like this",
        )

    if (
        features.peak_fraction >= TONE_PEAK
        and features.pitch_spread_hz <= TONE_PITCH_SPREAD_HZ
        and features.modulation_db < STEADY_MODULATION_DB
    ):
        return verdict(
            TONE,
            CONFIDENCE_CLEAR,
            f"all of the sound on one note near {features.pitch_hz:.0f} Hz"
            if features.pitch_hz
            else "all of the sound on a single note",
            "which never changes and never pauses",
        )

    talking = (
        features.modulation_db >= VOICE_MODULATION_DB
        and features.syllabic >= VOICE_SYLLABIC
        and features.voice_band >= VOICE_BAND_SHARE
    )
    if talking and features.has_pitch:
        return verdict(
            VOICE,
            CONFIDENCE_CLEAR,
            _rate_phrase(features),
            f"with a pitch around {features.pitch_hz:.0f} Hz that wanders "
            f"the way a voice does",
            f"and {features.voice_band * 100:.0f}% of it in the 300-3400 Hz "
            f"band a microphone is filtered to",
        )
    if talking:
        return verdict(
            VOICE,
            CONFIDENCE_LIKELY,
            _rate_phrase(features),
            "in the voice band, though no clear pitch came through - which is "
            "what a weak or clipped signal does to speech",
        )

    if (
        features.voiced >= MUSIC_PITCHED_SHARE
        and features.pitch_spread_hz <= PITCH_WANDER_HZ
        and features.modulation_db < VOICE_MODULATION_DB
        and features.peak_fraction < TONE_PEAK
    ):
        return verdict(
            MUSIC,
            CONFIDENCE_LIKELY,
            f"pitched in {features.voiced * 100:.0f}% of it, with none of the "
            f"pauses a sentence has",
            "and more than one note sounding at once",
        )

    if (
        features.flatness >= NOISE_FLATNESS
        and not features.has_pitch
        and features.modulation_db < STEADY_MODULATION_DB
    ):
        return verdict(
            NOISE,
            CONFIDENCE_CLEAR,
            "featureless right across the channel",
            "with no pitch and no rhythm - the sound of an open squelch",
        )

    if (
        features.modulation_db < STEADY_MODULATION_DB
        and not features.has_pitch
        and features.peak_fraction < TONE_PEAK
    ):
        return verdict(
            DATA,
            CONFIDENCE_LIKELY,
            "steady from end to end, with no pauses and never twice on the "
            "same note",
            "which is what a bit stream sounds like",
        )

    return verdict(
        UNCLEAR,
        CONFIDENCE_GUESS,
        f"loudness swinging {features.modulation_db:.0f} dB, "
        f"{features.voice_band * 100:.0f}% of it in the voice band",
        "which is not enough like any of the things this can recognise",
    )


__all__ = [
    "DATA",
    "MUSIC",
    "NOISE",
    "SILENCE",
    "TONE",
    "UNCLEAR",
    "VOICE",
    "Features",
    "Verdict",
    "classify",
    "measure",
]
