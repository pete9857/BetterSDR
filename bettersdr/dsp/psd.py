"""Power spectral density: the numbers behind the spectrum and waterfall.

This module is shared by the display and, later, the scanner. That is
deliberate - if the detector measured the band differently from the picture the
user is looking at, "the app found a signal I cannot see" becomes possible, and
explainability is the whole point of the product.

Levels are dBFS: a full-scale complex sinusoid reads 0 dB regardless of FFT
size or window, so the numbers mean the same thing whatever the display
settings are.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

# The windows SDR# offers, under the names users will recognise.
WINDOWS = ("hann", "blackmanharris", "hamming", "flattop", "boxcar")

DEFAULT_FFT_SIZE = 4096


class Spectrum:
    """Welch-averaged PSD with optional exponential smoothing across calls.

    Averaging several FFT frames per block trades frequency resolution for a
    steadier noise floor, which matters twice over: it stops the display
    shimmering, and it is what makes a detection threshold meaningful rather
    than a coin toss on which noise peak happened to be tallest.
    """

    def __init__(
        self,
        fft_size: int = DEFAULT_FFT_SIZE,
        sample_rate: float = 2_400_000.0,
        window: str = "hann",
        smoothing: float = 0.0,
        remove_dc: bool = True,
    ) -> None:
        if fft_size < 16 or fft_size & (fft_size - 1):
            raise ValueError(f"fft_size must be a power of two >= 16, got {fft_size}")
        if window not in WINDOWS:
            raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.fft_size = int(fft_size)
        self.sample_rate = float(sample_rate)
        self.window_name = window
        self.smoothing = float(smoothing)
        self.remove_dc = bool(remove_dc)

        self._window = get_window(window, self.fft_size).astype(np.float32)
        # Coherent gain: dividing by it puts a full-scale tone at 0 dBFS
        # whatever window is selected.
        self._normalisation = float(np.sum(self._window)) ** 2
        self._average: np.ndarray | None = None

    @property
    def bin_width_hz(self) -> float:
        return self.sample_rate / self.fft_size

    def reset(self) -> None:
        self._average = None

    def frequencies(self, center_hz: float = 0.0) -> np.ndarray:
        """Absolute frequency of each bin, ordered low to high."""
        return center_hz + np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate)
        )

    def power(self, iq: np.ndarray) -> np.ndarray:
        """Linear power per bin, fftshifted. Returns an empty array if short."""
        frames = iq.size // self.fft_size
        if frames == 0:
            return np.zeros(0, dtype=np.float32)

        block = np.asarray(iq[: frames * self.fft_size], dtype=np.complex64)
        block = block.reshape(frames, self.fft_size)
        if self.remove_dc:
            # The RTL2832U leaves a DC offset that shows up as a permanent
            # spike at centre. Removing the mean per frame kills it at source,
            # which is better than patching the bin afterwards and pretending.
            block = block - block.mean(axis=1, keepdims=True)

        spectra = np.fft.fftshift(np.fft.fft(block * self._window, axis=1), axes=1)
        power = np.mean(np.abs(spectra) ** 2, axis=0) / self._normalisation

        if self.smoothing > 0.0 and self._average is not None:
            power = self.smoothing * self._average + (1.0 - self.smoothing) * power
        self._average = power
        return power.astype(np.float32)

    def process(self, iq: np.ndarray) -> np.ndarray:
        """dBFS per bin, fftshifted so index 0 is the lowest frequency."""
        power = self.power(iq)
        if power.size == 0:
            return power
        return (10.0 * np.log10(np.maximum(power, 1e-20))).astype(np.float32)


class PeakHold:
    """A max-hold trace that decays, so old peaks fade instead of sticking."""

    def __init__(self, decay_db_per_frame: float = 0.5) -> None:
        self.decay_db_per_frame = float(decay_db_per_frame)
        self._peak: np.ndarray | None = None

    @property
    def trace(self) -> np.ndarray | None:
        return self._peak

    def reset(self) -> None:
        self._peak = None

    def update(self, spectrum_db: np.ndarray) -> np.ndarray:
        if self._peak is None or self._peak.shape != spectrum_db.shape:
            self._peak = spectrum_db.copy()
            return self._peak
        self._peak = np.maximum(self._peak - self.decay_db_per_frame, spectrum_db)
        return self._peak


def noise_floor_db(spectrum_db: np.ndarray, percentile: float = 30.0) -> float:
    """Estimate the noise floor as a low percentile of the bins.

    A percentile rather than the mean, because the mean of a band containing a
    strong station is pulled up by that station and would hide everything
    weaker sitting next to it.
    """
    if spectrum_db.size == 0:
        return -120.0
    return float(np.percentile(spectrum_db, percentile))


def occupied_bandwidth_hz(
    spectrum_db: np.ndarray, bin_width_hz: float, fraction: float = 0.99
) -> float:
    """Width of the band holding `fraction` of the total power.

    This is one of the features the classifier will use to tell a 200 kHz FM
    station from a 12.5 kHz two-way radio channel.
    """
    if spectrum_db.size == 0:
        return 0.0
    power = 10.0 ** (spectrum_db / 10.0)
    cumulative = np.cumsum(power)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    margin = (1.0 - fraction) / 2.0
    low = int(np.searchsorted(cumulative, total * margin))
    high = int(np.searchsorted(cumulative, total * (1.0 - margin)))
    return max(1, high - low) * bin_width_hz


__all__ = [
    "DEFAULT_FFT_SIZE",
    "WINDOWS",
    "PeakHold",
    "Spectrum",
    "noise_floor_db",
    "occupied_bandwidth_hz",
]
