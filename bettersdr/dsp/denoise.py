"""Noise blanking and noise reduction.

Two completely different problems that get filed under the same heading:

* A **noise blanker** deals with *impulses* - ignition noise, a thermostat, a
  plasma TV, an electric fence. In the time domain these are enormous and very
  short, so they are trivial to spot before any filtering has smeared them
  out, which is why the blanker runs on the raw IQ rather than on the audio.
* **Noise reduction** deals with the *steady* hiss underneath everything. That
  is invisible in the time domain and obvious in the frequency domain, so it
  is done with a short-time Fourier transform: estimate how much of each bin
  is noise, then turn those bins down.

Both are streaming stages in the sense the rest of `dsp/` means it: they carry
whatever state is needed for a block-by-block run to produce the same answer
as one pass over the whole stream.

The one honest warning about spectral subtraction is that it trades hiss for
"musical noise" - isolated surviving bins that warble. `reduction_db` is
capped rather than unlimited for that reason: a gain floor around 10-15 dB
sounds like a quieter radio, and 40 dB sounds like a broken one.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window, lfilter

# Cheap enough at 2.4 MS/s (a 512-point FFT every 256 samples is ~5% of one
# core when the frames are batched) and long enough at 48 kHz to resolve the
# 90 Hz that separates a voice formant from the hiss around it.
DEFAULT_FFT_SIZE = 512


class NoiseBlanker:
    """Clip impulses on the IF before they reach the channel filter.

    The threshold is relative to a running average of the signal's own
    magnitude, not an absolute level, so it means the same thing on a strong
    FM station and on a bare noise floor - and it does not need re-tuning
    every time the gain changes.

    Detection and suppression use *different* levels, which is the detail that
    makes the widening worth having. A sample is called an impulse when it
    exceeds several times the average, and every sample in the window that
    follows is then held down to the average itself. Suppressing to the same
    level that detected would leave the impulse at several times the signal
    around it - still a click - and would make the window pointless, since by
    definition no other sample in it crossed the threshold.

    Widening is causal, because an impulse has shoulders on the way out and
    because the decimation filter that follows smears whatever is left over
    its whole impulse response. It carries across block boundaries, so one
    landing on the last sample of a block still suppresses into the next.
    """

    def __init__(
        self,
        sample_rate: float,
        threshold: float = 4.0,
        width: int = 6,
        average_ms: float = 1.0,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.threshold = float(threshold)
        self.width = max(1, int(width))
        self.blanked = 0
        # A one-pole average of |x|, slow enough to ignore the impulses it is
        # meant to find but fast enough to follow a real fade.
        self._alpha = 1.0 - np.exp(
            -1.0 / max(1.0, self.sample_rate * average_ms / 1000.0)
        )
        self._b = np.array([self._alpha])
        self._a = np.array([1.0, self._alpha - 1.0])
        self._zi = np.zeros(1)
        self._carry = np.zeros(self.width - 1, dtype=np.float32)
        self._seeded = False

    def reset(self) -> None:
        self._zi[:] = 0.0
        self._carry[:] = 0.0
        self._seeded = False
        self.blanked = 0

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size == 0:
            return iq

        magnitude = np.abs(iq)
        if not self._seeded:
            # Starting the average at zero would make every sample of the
            # first millisecond look like an impulse - measured at 77 samples
            # blanked before the filter had climbed to the signal - so the
            # stream would open with the blanker chewing on real signal. The
            # first block's own mean is the best available guess.
            self._zi[:] = (1.0 - self._alpha) * float(np.mean(magnitude))
            self._seeded = True
        average, self._zi = lfilter(self._b, self._a, magnitude, zi=self._zi)
        average = np.maximum(average, 1e-9).astype(np.float32)

        hit = (magnitude > average * self.threshold).astype(np.float32)
        # The full convolution is exactly the causal widening plus the tail
        # that belongs to the next block, so both fall out of one call. The
        # tail is taken before the previous block's is added in, so a carry
        # can never be counted twice.
        spread = np.convolve(hit, np.ones(self.width, dtype=np.float32))
        widened = spread[: iq.size].copy()
        if self._carry.size:
            widened[: self._carry.size] += self._carry
            self._carry = spread[iq.size :].copy()
        blanking = widened > 0.0

        # Scale the offender down to the level around it rather than zeroing
        # it: the phase is still the phase of whatever was underneath the
        # impulse, and a hole punched in the stream is itself an impulse.
        scale = np.ones(iq.size, dtype=np.float32)
        np.divide(average, np.maximum(magnitude, 1e-12), out=scale, where=blanking)
        np.minimum(scale, 1.0, out=scale)
        self.blanked += int(np.count_nonzero(scale < 1.0))
        return (iq * scale).astype(np.complex64)


class SpectralNoiseReduction:
    """Spectral subtraction over a 50%-overlap STFT.

    The noise estimate is a per-bin minimum tracker: it follows the smoothed
    power straight down and is only allowed to climb by `rise_db_per_frame`,
    so it settles on the quietest thing each bin has been recently. That is
    what a noise floor *is*, and unlike a "press this while the channel is
    idle" calibration it needs nothing from the user and keeps working when
    conditions change.

    Analysis and synthesis both use a root-Hann window, whose product is a
    Hann window, which sums to exactly one at half-window hops. So with the
    gains all at one the output is the input, delayed - a property worth
    testing, because it is the thing that breaks first when the overlap-add
    bookkeeping is wrong.
    """

    def __init__(
        self,
        sample_rate: float,
        fft_size: int = DEFAULT_FFT_SIZE,
        reduction_db: float = 12.0,
        over_subtraction: float = 1.5,
        smoothing_ms: float = 20.0,
        rise_db_per_second: float = 8.0,
        smooth_bins: int = 3,
        complex_input: bool = False,
    ) -> None:
        if fft_size < 32 or fft_size & (fft_size - 1):
            raise ValueError(f"fft_size must be a power of two >= 32, got {fft_size}")
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.hop = self.fft_size // 2
        self.reduction_db = float(reduction_db)
        self.over_subtraction = float(over_subtraction)
        self.smooth_bins = max(1, int(smooth_bins))
        self.complex_input = bool(complex_input)

        # Both time constants are stated in real time and converted here.
        # Stated per frame they would mean something different at every
        # sample rate and FFT size the app supports - 187 frames a second at
        # 48 kHz against 9,375 at 2.4 MS/s - and a noise tracker allowed to
        # climb fifty times faster simply follows the signal and reduces
        # nothing.
        frame_rate = self.sample_rate / self.hop
        frames_per_tau = max(1e-6, frame_rate * smoothing_ms / 1000.0)
        self.smoothing = float(np.clip(np.exp(-1.0 / frames_per_tau), 0.0, 0.99))
        self.rise_db_per_frame = float(rise_db_per_second) / frame_rate

        self.dtype = np.complex64 if self.complex_input else np.float32
        self._window = np.sqrt(
            get_window("hann", self.fft_size, fftbins=True)
        ).astype(np.float32)
        self._pending = np.zeros(0, dtype=self.dtype)
        self._tail = np.zeros(self.hop, dtype=self.dtype)
        self._power: np.ndarray | None = None
        self._noise: np.ndarray | None = None

    @property
    def settling_samples(self) -> int:
        """Output samples that are a window fade-in rather than real audio.

        The overlap-add is phase-aligned - output sample `n` is input sample
        `n`, not a delayed copy - but the first half window has only one frame
        contributing to it and so comes out quiet. After that the sum of the
        overlapping windows is exactly one.
        """
        return self.hop

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=self.dtype)
        self._tail = np.zeros(self.hop, dtype=self.dtype)
        self._power = None
        self._noise = None

    # -- the estimate ------------------------------------------------------

    def _track_noise(self, power: np.ndarray) -> np.ndarray:
        """Smooth the power over time, then follow its minimum per bin.

        Both recursions are done without a Python loop over frames. The
        smoothing is a one-pole IIR, which `lfilter` runs along the frame
        axis. The minimum tracker looks recursive but is not: allowing the
        estimate to rise by a fixed number of dB per frame means

            log n[k] = min over j <= k of (log p[j] + (k - j) * log rise)

        and pulling the `k` out of the minimum turns it into a running
        minimum of `log p[j] - j * log rise`, which is one `accumulate` call.
        Done in the log domain rather than on the powers themselves because
        `rise ** -j` underflows within a few hundred frames.
        """
        alpha = 1.0 - self.smoothing
        if self._power is None:
            self._power = power[0].astype(np.float64)
        # `lfilter`'s transposed-form state for a one-pole is the pole times
        # the previous output, so it is seeded rather than assigned, and the
        # carried value is read back off the output instead of out of the
        # state - the state is scaled and the output is not.
        zi = (self._power * self.smoothing)[None, :]
        smoothed, _ = lfilter([alpha], [1.0, -self.smoothing], power, axis=0, zi=zi)
        self._power = smoothed[-1]

        log_rise = self.rise_db_per_frame * np.log(10.0) / 10.0
        log_power = np.log(np.maximum(smoothed, 1e-30))
        frames = np.arange(power.shape[0], dtype=np.float64)[:, None]
        if self._noise is None:
            self._noise = log_power[0].copy()
        # The carried estimate is the value at frame -1, so it enters the
        # running minimum already advanced by one step of rise.
        seed = (self._noise + log_rise)[None, :]
        shifted = np.concatenate((seed, log_power - frames * log_rise))
        running = np.minimum.accumulate(shifted, axis=0)[1:]
        log_noise = running + frames * log_rise
        self._noise = log_noise[-1]
        return np.exp(log_noise), smoothed

    # -- streaming ---------------------------------------------------------

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=self.dtype)
        buffered = (
            np.concatenate((self._pending, samples))
            if self._pending.size
            else samples
        )
        frames = (buffered.size - self.fft_size) // self.hop + 1
        if frames < 1:
            self._pending = buffered.copy()
            return np.zeros(0, dtype=self.dtype)

        index = np.arange(self.fft_size)[None, :] + self.hop * np.arange(frames)[
            :, None
        ]
        block = buffered[index] * self._window
        self._pending = buffered[frames * self.hop :].copy()

        transform = np.fft.fft if self.complex_input else np.fft.rfft
        inverse = np.fft.ifft if self.complex_input else np.fft.irfft
        spectra = transform(block, axis=1)

        power = np.abs(spectra) ** 2
        noise, smoothed = self._track_noise(power)
        floor = 10.0 ** (-abs(self.reduction_db) / 10.0)
        keep = (smoothed - self.over_subtraction * noise) / np.maximum(
            smoothed, 1e-30
        )
        gain = np.sqrt(np.clip(keep, floor, 1.0))
        if self.smooth_bins > 1:
            # Smearing the gain across neighbouring bins is what keeps the
            # surviving bins from standing alone and warbling.
            gain = uniform_filter1d(gain, self.smooth_bins, axis=1, mode="nearest")

        restored = inverse(spectra * gain, n=self.fft_size, axis=1) * self._window
        return self._overlap_add(restored.astype(self.dtype))

    def _overlap_add(self, frames: np.ndarray) -> np.ndarray:
        """Sum the half-overlapping frames back into a continuous stream.

        At exactly 50% overlap every output hop is one frame's first half plus
        the previous frame's second half, so the whole overlap-add is two
        array additions rather than a loop with an index.
        """
        first, second = frames[:, : self.hop], frames[:, self.hop :]
        out = first.copy()
        out[1:] += second[:-1]
        out[0] += self._tail
        self._tail = second[-1].copy()
        return out.reshape(-1)


__all__ = ["DEFAULT_FFT_SIZE", "NoiseBlanker", "SpectralNoiseReduction"]
