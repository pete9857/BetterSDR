"""Synthetic IQ generation for tests.

Almost every DSP and scanning claim in BetterSDR can be checked without a
dongle attached: plant a signal with known parameters, run it through the
pipeline, and assert the pipeline recovers those parameters. That keeps the
test suite fast and runnable on a machine with no hardware.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RATE = 2_400_000


def _time(n: int, rate: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / rate


def noise(n: int, rms: float = 0.01, seed: int = 0) -> np.ndarray:
    """Complex additive white Gaussian noise at a given RMS magnitude."""
    rng = np.random.default_rng(seed)
    scale = rms / np.sqrt(2.0)
    return (rng.normal(0, scale, n) + 1j * rng.normal(0, scale, n)).astype(np.complex64)


def carrier(
    n: int, offset_hz: float, amplitude: float = 0.5, rate: float = DEFAULT_RATE
) -> np.ndarray:
    """An unmodulated tone at `offset_hz` from centre."""
    phase = 2 * np.pi * offset_hz * _time(n, rate)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def am(
    n: int,
    offset_hz: float,
    tone_hz: float = 1000.0,
    depth: float = 0.8,
    amplitude: float = 0.5,
    rate: float = DEFAULT_RATE,
) -> np.ndarray:
    """Amplitude modulation: a strong carrier plus symmetric sidebands."""
    t = _time(n, rate)
    envelope = 1.0 + depth * np.sin(2 * np.pi * tone_hz * t)
    phase = 2 * np.pi * offset_hz * t
    return (amplitude * envelope * np.exp(1j * phase)).astype(np.complex64)


def fm(
    n: int,
    offset_hz: float,
    tone_hz: float = 1000.0,
    deviation_hz: float = 75_000.0,
    amplitude: float = 0.5,
    rate: float = DEFAULT_RATE,
) -> np.ndarray:
    """Frequency modulation: constant envelope, phase carries the audio.

    `deviation_hz` defaults to the 75 kHz used by FM broadcast; pass 2500 for
    the narrowband FM used on ham and business radio.
    """
    t = _time(n, rate)
    # Phase is the integral of instantaneous frequency. For a sine tone that
    # integral is a negative cosine, so we can write it in closed form rather
    # than accumulating and picking up drift.
    modulation = -(deviation_hz / tone_hz) * np.cos(2 * np.pi * tone_hz * t)
    phase = 2 * np.pi * offset_hz * t + modulation
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def scene(
    n: int,
    signals: list[np.ndarray],
    noise_rms: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Sum several signals into one capture over a noise floor."""
    out = noise(n, rms=noise_rms, seed=seed)
    for signal in signals:
        out = out + signal
    return out.astype(np.complex64)


def band_noise(
    n: int,
    lo_hz: float,
    hi_hz: float,
    rms: float = 0.05,
    rate: float = DEFAULT_RATE,
    seed: int = 1,
    both_sides: bool = True,
) -> np.ndarray:
    """Flat noise confined to a frequency band, built in the frequency domain.

    This stands in for OFDM. A dense multi-carrier signal is indistinguishable
    from band-limited noise on a PSD, which is exactly the property the HD
    Radio sideband test keys on, so it is the right synthetic for that test.
    """
    rng = np.random.default_rng(seed)
    spectrum = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    freqs = np.fft.fftfreq(n, 1.0 / rate)
    if both_sides:
        keep = (np.abs(freqs) >= lo_hz) & (np.abs(freqs) <= hi_hz)
    else:
        keep = (freqs >= lo_hz) & (freqs <= hi_hz)
    band = np.fft.ifft(spectrum * keep)
    scale = rms / max(float(np.sqrt(np.mean(np.abs(band) ** 2))), 1e-20)
    return (band * scale).astype(np.complex64)


def hd_radio_fm(
    n: int,
    offset_hz: float = 0.0,
    tone_hz: float = 1000.0,
    amplitude: float = 0.5,
    digital_dbc: float = -15.0,
    rate: float = DEFAULT_RATE,
    seed: int = 1,
    both_sides: bool = True,
) -> np.ndarray:
    """An FM station carrying hybrid IBOC digital sidebands.

    Analog FM as usual, plus flat energy 129-198 kHz either side of the
    carrier at `digital_dbc` relative to it - the shape measured off air on
    real stations, where the sidebands ran about 14 dB down.
    """
    analog = fm(n, offset_hz, tone_hz=tone_hz, amplitude=amplitude, rate=rate)
    # Match power spectral density rather than total power: the detector
    # compares mean power per bin, and the two regions differ in width.
    core_width = 2 * 60_000.0
    sideband_width = 2 * (198_000.0 - 129_000.0)
    density_ratio = np.sqrt(sideband_width / core_width)
    digital_rms = amplitude * (10.0 ** (digital_dbc / 20.0)) * density_ratio
    sidebands = band_noise(
        n, 129_000.0, 198_000.0, rms=digital_rms, rate=rate, seed=seed,
        both_sides=both_sides,
    )
    if offset_hz:
        sidebands = sidebands * np.exp(
            2j * np.pi * offset_hz * _time(n, rate)
        ).astype(np.complex64)
    return (analog + sidebands).astype(np.complex64)


# -- A whole band that answers retuning ------------------------------------
#
# The sweeper's job is to step a 2.4 MHz window across a range and end up with
# a correct picture of the whole thing, so testing it needs air that responds
# to being tuned rather than one fixed capture. `Air` holds transmitters at
# absolute frequencies and renders only what falls inside the current window,
# which is what puts the step planning, the tile ownership and the duplicate
# merging under test at all.


def wfm_station(amplitude: float = 0.5, tone_hz: float = 1000.0):
    """A broadcast FM transmitter, about 180 kHz wide."""

    def make(n: int, offset_hz: float, rate: float) -> np.ndarray:
        return fm(
            n, offset_hz, tone_hz=tone_hz, deviation_hz=75_000.0,
            amplitude=amplitude, rate=rate,
        )

    return make


def nfm_station(amplitude: float = 0.4, tone_hz: float = 1000.0):
    """A two-way radio, about 15 kHz wide."""

    def make(n: int, offset_hz: float, rate: float) -> np.ndarray:
        return fm(
            n, offset_hz, tone_hz=tone_hz, deviation_hz=2_500.0,
            amplitude=amplitude, rate=rate,
        )

    return make


def am_station(amplitude: float = 0.4, tone_hz: float = 3000.0):
    """An AM transmitter: a hard carrier with sidebands either side."""

    def make(n: int, offset_hz: float, rate: float) -> np.ndarray:
        return am(n, offset_hz, tone_hz=tone_hz, amplitude=amplitude, rate=rate)

    return make


def digital_station(amplitude: float = 0.3, width_hz: float = 200_000.0):
    """A dense digital carrier: flat across its band, like noise."""

    def make(n: int, offset_hz: float, rate: float) -> np.ndarray:
        block = band_noise(
            n, 0.0, width_hz / 2.0, rms=amplitude, rate=rate, both_sides=True
        )
        if offset_hz:
            block = block * np.exp(2j * np.pi * offset_hz * _time(n, rate))
        return block.astype(np.complex64)

    return make


class Air:
    """Synthetic spectrum that responds to retuning, for sweeper tests."""

    def __init__(
        self,
        stations: dict[float, object] | None = None,
        rate: float = DEFAULT_RATE,
        noise_rms: float = 0.004,
        seed: int = 0,
    ) -> None:
        self.stations = dict(stations or {})
        self.rate = float(rate)
        self.noise_rms = float(noise_rms)
        self.seed = int(seed)
        self.center = 0
        self.tuned_to: list[int] = []
        self.rates: list[float] = []

    def tune(self, hz: int) -> None:
        self.center = int(hz)
        self.tuned_to.append(int(hz))

    def set_sample_rate(self, hz: float) -> None:
        """Answer a change of window, the way the real front end does.

        A sweep of several stretches of dial changes the window at every
        boundary - the AM band has to be measured through 240 kHz and FM
        through 2.4 MHz - so synthetic air that only ever produces one width
        cannot put that path under test.
        """
        self.rate = float(hz)
        self.rates.append(float(hz))

    def read(self, samples: int) -> np.ndarray:
        # Only what is actually inside the window, exactly as the hardware
        # would deliver it. A station outside simply is not there to be found.
        parts = [
            make(samples, hz - self.center, self.rate)
            for hz, make in self.stations.items()
            if abs(hz - self.center) < self.rate / 2.0
        ]
        # A new seed per read, so repeated passes see independent noise and the
        # persistence gate is genuinely exercised rather than handed identical
        # sweeps three times.
        self.seed += 1
        return scene(samples, parts, noise_rms=self.noise_rms, seed=self.seed)
