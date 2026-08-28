"""The optional processing either side of the demodulator.

Phase 3 adds a lot of switches - IQ correction, a noise blanker, noise
reduction, AGC - and they all share two awkward properties: they are stateful,
and they are almost always off. Left loose in the engine they would be a dozen
attributes and a dozen `if enabled` branches interleaved with the code that
actually pumps samples.

So they live here in two chains, one either side of the demodulator, and each
chain answers exactly one question for the engine: *given this block, what
comes out*. When nothing is enabled that answer is the block itself, returned
without touching it, so the cost of the whole feature set on a default install
is one boolean test per block.

The split is not arbitrary. The front end works on raw IQ because that is
where a receiver's own defects live and where an impulse is still an impulse
rather than a smear. The audio chain works at 48 kHz because that is where
loudness is a thing a listener has an opinion about. Noise reduction is in
neither: it sits inside the demodulator, at the IF, where the samples are ten
to fifty times cheaper - see `Demodulator._front`.
"""

from __future__ import annotations

import numpy as np

from .agc import Agc
from .correct import DcRemover, FrequencyShifter, ImbalanceEstimate, IqBalance, swap_iq
from .denoise import NoiseBlanker, SpectralNoiseReduction
from .filters import BandPass

# Far enough off centre to move the DC spike outside any channel a user is
# likely to be listening to, and well inside the narrowest window the app
# supports so it never lands outside the trusted part of the passband.
DEFAULT_OFFSET_FRACTION = 0.15


class FrontEnd:
    """IQ correction, offset tuning and impulse blanking, in that order.

    The order is not interchangeable. DC removal comes first because the
    imbalance estimator averages products of I and Q, and a DC offset biases
    both. The offset shift comes after both corrections, because those
    corrections describe the hardware and the hardware does not know we have
    moved the dial. The blanker comes last so it sees the signal the
    demodulator will see.
    """

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = float(sample_rate)
        self.dc_removal = False
        self.iq_balance = False
        self.swap_iq = False
        self.noise_blanker = False
        self.offset_hz = 0.0
        self._build()

    def _build(self) -> None:
        self._dc = DcRemover(self.sample_rate)
        self._balance = IqBalance(self.sample_rate)
        self._blanker = NoiseBlanker(self.sample_rate)
        self._shifter = FrequencyShifter(self.sample_rate, self.offset_hz)

    # -- configuration -----------------------------------------------------

    def set_sample_rate(self, sample_rate: float) -> None:
        if float(sample_rate) == self.sample_rate:
            return
        self.sample_rate = float(sample_rate)
        self._build()

    def set_offset_hz(self, offset_hz: float) -> None:
        """Where the tuner is parked relative to the frequency being heard.

        The tuner is parked `offset_hz` above the frequency being listened
        to, so the wanted signal arrives that far *below* the middle of the
        window. Mixing up by the same amount is what puts it back at zero -
        and it leaves the DC spike sitting harmlessly at `+offset_hz`, which
        is the entire point of the exercise.
        """
        self.offset_hz = float(offset_hz)
        self._shifter.offset_hz = self.offset_hz

    def set_blanker(self, enabled: bool, threshold: float | None = None) -> None:
        self.noise_blanker = bool(enabled)
        if threshold is not None:
            self._blanker.threshold = float(threshold)

    @property
    def blanker_threshold(self) -> float:
        return self._blanker.threshold

    @property
    def blanked_samples(self) -> int:
        return self._blanker.blanked

    @property
    def imbalance(self) -> ImbalanceEstimate:
        return self._balance.estimate

    @property
    def dc_offset(self) -> complex:
        return self._dc.offset

    @property
    def active(self) -> bool:
        return (
            self.dc_removal
            or self.iq_balance
            or self.swap_iq
            or self.noise_blanker
            or self.offset_hz != 0.0
        )

    def reset(self) -> None:
        self._dc.reset()
        self._balance.reset()
        self._blanker.reset()
        self._shifter.reset()

    # -- streaming ---------------------------------------------------------

    def process(self, iq: np.ndarray) -> np.ndarray:
        if not self.active:
            return iq
        if self.swap_iq:
            iq = swap_iq(iq)
        if self.dc_removal:
            iq = self._dc.process(iq)
        if self.iq_balance:
            iq = self._balance.process(iq)
        if self.offset_hz != 0.0:
            iq = self._shifter.process(iq)
        if self.noise_blanker:
            iq = self._blanker.process(iq)
        return iq


class AudioChain:
    """Noise reduction, AGC, volume and the limiter that ends the path.

    Volume lives here rather than in the demodulator because the AGC has to
    come first: a gain rider working behind a fixed attenuator would spend its
    range undoing it, and one working behind a limiter cannot recover what the
    limiter already flattened. So the engine hands the demodulator unity gain
    and this applies the user's volume last, immediately before the clip.

    That also makes muting exact rather than approximate - the output is
    zeros, not a very small number - which matters because the sound card is
    the one place in the app where "nearly" is audible.
    """

    def __init__(self, sample_rate: float = 48_000.0) -> None:
        self.sample_rate = float(sample_rate)
        self.volume = 0.5
        self.mute = False
        self.agc_enabled = False
        self.noise_reduction = False
        self.filter_audio = False
        self._agc = Agc(self.sample_rate)
        self._nr: SpectralNoiseReduction | None = None
        self._band = BandPass(self.sample_rate)

    # -- configuration -----------------------------------------------------

    @property
    def agc(self) -> Agc:
        return self._agc

    def configure_agc(self, **settings: float | bool) -> None:
        """Rebuild the AGC around new parameters, keeping the ones not given.

        Rebuilt rather than mutated because the ramp coefficients are derived
        from the time constants at construction, and a half-updated AGC would
        apply an attack from one setting with a decay from another.
        """
        current = {
            "target_dbfs": self._agc.target_dbfs,
            "threshold_dbfs": self._agc.threshold_dbfs,
            "slope_db": self._agc.slope_db,
            "attack_ms": self._agc.attack_ms,
            "decay_ms": self._agc.decay_ms,
            "use_hang": self._agc.use_hang,
            "hang_ms": self._agc.hang_ms,
            "max_gain_db": self._agc.max_gain_db,
        }
        current.update(settings)
        self._agc = Agc(self.sample_rate, **current)

    def set_noise_reduction(
        self, enabled: bool, reduction_db: float | None = None
    ) -> None:
        self.noise_reduction = bool(enabled)
        if self._nr is None or reduction_db is not None:
            self._nr = SpectralNoiseReduction(
                self.sample_rate,
                reduction_db=12.0 if reduction_db is None else float(reduction_db),
            )

    @property
    def reduction_db(self) -> float:
        return 12.0 if self._nr is None else self._nr.reduction_db

    def set_audio_filter(
        self, enabled: bool, low_hz: float = 300.0, high_hz: float = 3_000.0
    ) -> None:
        self.filter_audio = bool(enabled)
        if (low_hz, high_hz) != (self._band.low_hz, self._band.high_hz):
            self._band = BandPass(self.sample_rate, low_hz, high_hz)

    @property
    def audio_filter_range(self) -> tuple[float, float]:
        return self._band.low_hz, self._band.high_hz

    @property
    def keeps_stereo(self) -> bool:
        """Whether a stereo block survives this chain as stereo."""
        return not self.noise_reduction

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """Average the channels of a `(frames, channels)` block."""
        return audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)

    def reset(self) -> None:
        self._agc.reset()
        self._band.reset()
        if self._nr is not None:
            self._nr.reset()

    # -- streaming ---------------------------------------------------------

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        if self.noise_reduction and self._nr is not None:
            # Spectral subtraction works one channel at a time, and running it
            # twice would build two independent noise estimates and two gain
            # masks - which is exactly how a stereo image is pulled apart. It
            # is a tool for a weak, hissy, mono signal, so the honest answer
            # is to mix down and say so: `stereo` goes false, and the app's
            # indicator reports what is actually being heard.
            audio = self.to_mono(audio)
            audio = self._nr.process(audio)
        if self.filter_audio:
            # Before the AGC, so the gain rider is measuring the audio the
            # listener will actually hear rather than hiss that is about to be
            # thrown away - otherwise turning the filter on makes everything
            # quieter, which is the opposite of what it is for.
            audio = self._band.process(audio)
        if self.agc_enabled:
            audio = self._agc.process(audio)
        if audio.size == 0:
            return audio
        if self.mute:
            return np.zeros_like(audio)
        return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)


__all__ = ["DEFAULT_OFFSET_FRACTION", "AudioChain", "FrontEnd"]
