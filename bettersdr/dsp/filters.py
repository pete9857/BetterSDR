"""Filtering, decimation and gating primitives shared by the demodulators.

Everything here is *stateful and block-oriented*. The app feeds a continuous
stream in chunks, so each stage remembers enough history to make its output
identical to processing the whole stream at once. Without that, every block
boundary becomes a discontinuity and the audio ticks once per block - a fault
that is invisible in a test on a single buffer and glaringly obvious the
moment you listen to a real station.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter, oaconvolve, upfirdn

# Taps per polyphase branch. 24 puts the stopband edge close enough to the new
# Nyquist for 8-bit data, where the ADC noise floor sits far above the leakage
# anyway. Raising it sharpens the transition at linear CPU cost.
DEFAULT_TAPS_PER_PHASE = 24

# Above this many taps per output sample, an FFT convolution beats the direct
# polyphase form. Measured on the SSB and CW chains, where it is worth ~10x.
_FFT_TAPS_PER_PHASE = 64


def _pad_to_phase(taps: np.ndarray, factor: int) -> np.ndarray:
    """Zero-pad so that `len(taps) - 1` is a multiple of the decimation factor.

    The overlap-save prefix is `len(taps) - 1` samples long, and the output
    phase only stays put across blocks if that prefix is a whole number of
    output samples. Trailing zeros add delay and change nothing else.
    """
    remainder = (taps.size - 1) % factor
    if remainder == 0:
        return taps
    return np.concatenate((taps, np.zeros(factor - remainder, dtype=taps.dtype)))


class FirDecimator:
    """Anti-aliased decimation that survives block boundaries.

    Overlap-save: every call is prefixed with the tail of the previous one, so
    the filter always sees its full history. Only the kept output samples are
    computed - `upfirdn` skips the rest - so cost scales with the output rate,
    not the input rate.
    """

    def __init__(
        self,
        taps: np.ndarray,
        factor: int,
        sample_rate: float,
        dtype: type = np.complex64,
    ) -> None:
        if factor < 1:
            raise ValueError(f"decimation factor must be >= 1, got {factor}")
        self.factor = int(factor)
        self.sample_rate = float(sample_rate)
        self.output_rate = self.sample_rate / self.factor
        # `firwin` hands back float64. Left alone, that promotes every complex64
        # sample to complex128 inside `upfirdn` and roughly triples the cost of
        # the filter for no accuracy that 8-bit input could possibly use.
        taps = np.asarray(taps)
        single = np.complex64 if np.iscomplexobj(taps) else np.float32
        self.taps = _pad_to_phase(taps.astype(single), self.factor)
        self.dtype = np.complex64 if np.iscomplexobj(self.taps) else dtype
        self._overlap = self.taps.size - 1
        self._tail = np.zeros(self._overlap, dtype=self.dtype)
        # Polyphase decimation costs one multiply per tap per *output* sample,
        # so it stays cheap while the factor is large. The narrow SSB and CW
        # filters are the opposite case - hundreds of taps at a factor of one -
        # and there an FFT convolution is an order of magnitude faster.
        self._use_fft = self.taps.size / self.factor >= _FFT_TAPS_PER_PHASE

    @classmethod
    def lowpass(
        cls,
        factor: int,
        cutoff_hz: float,
        sample_rate: float,
        taps_per_phase: int = DEFAULT_TAPS_PER_PHASE,
        dtype: type = np.complex64,
    ) -> FirDecimator:
        """A decimator whose anti-alias filter cuts off at `cutoff_hz`."""
        nyquist = sample_rate / 2.0
        if not 0 < cutoff_hz < nyquist:
            raise ValueError(f"cutoff {cutoff_hz} Hz outside (0, {nyquist})")
        # An even taps-per-phase keeps the tap count odd, which gives the
        # filter an exact integer-sample delay and no half-sample skew.
        per_phase = int(taps_per_phase) + (int(taps_per_phase) % 2)
        taps = firwin(factor * per_phase + 1, cutoff_hz, fs=sample_rate)
        return cls(taps, factor, sample_rate, dtype)

    @classmethod
    def bandpass(
        cls,
        factor: int,
        low_hz: float,
        high_hz: float,
        sample_rate: float,
        taps_per_phase: int = DEFAULT_TAPS_PER_PHASE,
    ) -> FirDecimator:
        """A complex decimator passing only [low_hz, high_hz].

        The band may sit entirely on one side of zero, which is what
        single-sideband needs: the negative frequencies carry the other
        sideband and must be discarded, not folded in.
        """
        centre = 0.5 * (low_hz + high_hz)
        half_width = 0.5 * (high_hz - low_hz)
        if half_width <= 0:
            raise ValueError(f"empty passband [{low_hz}, {high_hz}]")
        per_phase = int(taps_per_phase) + (int(taps_per_phase) % 2)
        numtaps = factor * per_phase + 1
        base = firwin(numtaps, half_width, fs=sample_rate)
        # Shifting a real lowpass up to `centre` turns it into a one-sided
        # complex bandpass.
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        taps = base * np.exp(2j * np.pi * centre * n / sample_rate)
        return cls(taps, factor, sample_rate, np.complex64)

    @property
    def block_multiple(self) -> int:
        return self.factor

    def reset(self) -> None:
        self._tail[:] = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=self.dtype)
        if samples.size % self.factor:
            raise ValueError(
                f"input length {samples.size} is not a multiple of {self.factor}"
            )
        if samples.size == 0:
            return np.zeros(0, dtype=self.dtype)

        buf = np.concatenate((self._tail, samples))
        if self._overlap:
            self._tail = buf[buf.size - self._overlap :].copy()

        if self._use_fft:
            # "valid" drops exactly the outputs that depend on samples before
            # the prefix, which is the same set overlap-save discards below.
            filtered = oaconvolve(buf, self.taps, mode="valid")
            return filtered[:: self.factor].astype(self.dtype)

        decimated = upfirdn(self.taps, buf, 1, self.factor)
        # Drop the outputs that depend on samples from before the prefix.
        start = self._overlap // self.factor
        return decimated[start : start + samples.size // self.factor].astype(self.dtype)


class BiquadState:
    """An IIR section run block-by-block, carrying `lfilter` state across."""

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        self._b = np.asarray(b, dtype=np.float64)
        self._a = np.asarray(a, dtype=np.float64)
        self._zi = np.zeros(max(self._a.size, self._b.size) - 1)

    def reset(self) -> None:
        self._zi[:] = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32)
        out, self._zi = lfilter(self._b, self._a, samples, zi=self._zi)
        return out.astype(np.float32)


class Deemphasis(BiquadState):
    """Undo the treble boost broadcasters apply before transmitting.

    FM broadcast pre-emphasises high frequencies to improve their noise
    performance; skipping the matching cut is the classic reason a home-made FM
    receiver sounds thin and hissy. 75 microseconds is the US standard, 50 is
    used through most of the rest of the world.
    """

    def __init__(self, sample_rate: float, tau_us: float = 75.0) -> None:
        tau = tau_us * 1e-6
        alpha = 1.0 - np.exp(-1.0 / (sample_rate * tau))
        # DC gain is alpha / (1 - (1 - alpha)) = 1, so loudness is unchanged.
        super().__init__(np.array([alpha]), np.array([1.0, alpha - 1.0]))


class DcBlock(BiquadState):
    """One-pole highpass removing the DC term AM demodulation leaves behind."""

    def __init__(self, pole: float = 0.999) -> None:
        super().__init__(np.array([1.0, -1.0]), np.array([1.0, -pole]))


class Discriminator:
    """Instantaneous frequency as the phase step between adjacent samples.

    `angle(x[n] * conj(x[n-1]))` is the whole of FM demodulation. The previous
    block's last sample is carried over so the first output of each block is a
    real phase step rather than a jump away from zero.
    """

    def __init__(self) -> None:
        self._last = np.complex64(0)

    def reset(self) -> None:
        self._last = np.complex64(0)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        previous = np.empty(samples.size, dtype=np.complex64)
        previous[0] = self._last
        previous[1:] = samples[:-1]
        self._last = samples[-1]
        return np.angle(samples * np.conj(previous)).astype(np.float32)


class Squelch:
    """Mute the audio when the channel holds nothing but noise.

    Two details separate a squelch that sounds deliberate from one that sounds
    broken. Hysteresis stops a signal hovering at the threshold from chattering
    open and shut, and ramping the gain rather than switching it avoids the
    click a hard gate puts at every transition.
    """

    def __init__(
        self,
        sample_rate: float,
        threshold_dbfs: float = -45.0,
        hysteresis_db: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 40.0,
    ) -> None:
        self.threshold_dbfs = float(threshold_dbfs)
        self.hysteresis_db = float(hysteresis_db)
        self._attack = max(1.0, sample_rate * attack_ms / 1000.0)
        self._release = max(1.0, sample_rate * release_ms / 1000.0)
        self._open = False
        self._gain = 0.0

    @property
    def is_open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False
        self._gain = 0.0

    def update(self, level_dbfs: float) -> bool:
        """Re-evaluate the gate against this block's channel power."""
        if self._open:
            self._open = level_dbfs > self.threshold_dbfs
        else:
            self._open = level_dbfs > self.threshold_dbfs + self.hysteresis_db
        return self._open

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        target = 1.0 if self._open else 0.0
        if self._gain == target:
            return audio if target == 1.0 else np.zeros_like(audio)
        step = 1.0 / (self._attack if target > self._gain else self._release)
        ramp = self._gain + np.sign(target - self._gain) * step * np.arange(
            1, audio.size + 1, dtype=np.float32
        )
        np.clip(ramp, min(self._gain, target), max(self._gain, target), out=ramp)
        self._gain = float(ramp[-1])
        return (audio * ramp).astype(np.float32)


def power_dbfs(samples: np.ndarray) -> float:
    """Mean power of a block, in dB relative to full scale."""
    if samples.size == 0:
        return -120.0
    mean_square = float(np.mean(np.abs(samples) ** 2))
    return 10.0 * np.log10(max(mean_square, 1e-12))


__all__ = [
    "BiquadState",
    "DcBlock",
    "Deemphasis",
    "Discriminator",
    "FirDecimator",
    "Squelch",
    "power_dbfs",
]
