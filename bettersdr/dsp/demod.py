"""Demodulators: complex baseband in, listenable audio out.

Every mode follows the same three-stage shape, which is why they share a base
class rather than each being written out longhand:

    1. Decimate the 2.4 MS/s stream down to an intermediate rate, using the
       anti-alias filter as the channel filter. This is where selectivity
       comes from, and it is also where almost all the CPU goes.
    2. Recover the audio - a phase difference for FM, an envelope for AM, a
       sideband selection for SSB.
    3. Decimate again to the sound card rate.

Demodulators are stateful and accept arbitrary block sizes: the base class
buffers whatever does not divide evenly into the decimation chain and prepends
it next time. That keeps USB read sizes independent of DSP arithmetic, which
matters because `rtlsdr_read_sync` has its own 512-byte constraint that has no
reason to line up with an audio decimation factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import DcBlock, Deemphasis, Discriminator, FirDecimator, Squelch, power_dbfs

AUDIO_RATE = 48_000


def _divisors(n: int) -> list[int]:
    found = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            found.update((i, n // i))
    return sorted(found)


def _plan_decimation(
    sample_rate: float, audio_rate: int, min_if_rate: float
) -> tuple[int, int]:
    """Split the total decimation into an IF stage and an audio stage.

    Demodulation has to happen at a rate that still carries the whole signal,
    so we take as much decimation as possible up front - that is the cheapest
    place to do it - while keeping the IF rate at or above `min_if_rate`.
    """
    ratio = sample_rate / audio_rate
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"sample rate {sample_rate:.0f} is not a whole multiple of the "
            f"audio rate {audio_rate}"
        )
    total = int(round(ratio))
    if_factor = 1
    for divisor in _divisors(total):
        if sample_rate / divisor >= min_if_rate:
            if_factor = max(if_factor, divisor)
    return if_factor, total // if_factor


@dataclass(frozen=True)
class ModeInfo:
    """What a mode is called and what it is for, in words a beginner can use."""

    mode: str
    label: str
    description: str
    default_bandwidth_hz: float


class Demodulator:
    """Base class: block buffering, squelch and level metering."""

    mode = "raw"
    label = "Raw"
    description = "The signal with no processing applied."
    default_bandwidth_hz = 48_000.0
    audio_cutoff_hz = 15_000.0

    def __init__(
        self,
        sample_rate: float,
        bandwidth_hz: float | None = None,
        audio_rate: int = AUDIO_RATE,
        squelch_dbfs: float | None = None,
        volume: float = 0.5,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.audio_rate = int(audio_rate)
        self.bandwidth_hz = float(
            bandwidth_hz if bandwidth_hz is not None else self.default_bandwidth_hz
        )
        self.volume = float(volume)
        self.channel_power_dbfs = -120.0
        self._pending = np.zeros(0, dtype=np.complex64)
        self._build()
        self.squelch = (
            None
            if squelch_dbfs is None
            else Squelch(self.audio_rate, threshold_dbfs=squelch_dbfs)
        )

    # -- subclass hooks ----------------------------------------------------

    def _build(self) -> None:
        raise NotImplementedError

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- shared machinery --------------------------------------------------

    @property
    def block_multiple(self) -> int:
        """Input samples the chain consumes per audio sample."""
        return self.if_factor * self.audio_factor

    @property
    def if_rate(self) -> float:
        return self.sample_rate / self.if_factor

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.complex64)
        for stage in vars(self).values():
            if hasattr(stage, "reset") and stage is not self:
                stage.reset()

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Demodulate a block of complex baseband into float32 audio."""
        iq = np.asarray(iq, dtype=np.complex64)
        if self._pending.size:
            iq = np.concatenate((self._pending, iq))
        usable = iq.size - (iq.size % self.block_multiple)
        self._pending = iq[usable:].copy()
        if usable == 0:
            return np.zeros(0, dtype=np.float32)

        audio = self._demodulate(iq[:usable])
        if self.squelch is not None:
            self.squelch.update(self.channel_power_dbfs)
            audio = self.squelch.process(audio)
        audio = audio * self.volume
        # The sound card clips hard and ugly; clip gently here instead.
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class _FmBase(Demodulator):
    """Shared FM path: channel filter, discriminator, audio filter."""

    deviation_hz = 75_000.0
    deemphasis_us: float | None = 75.0
    if_headroom = 1.2

    def _build(self) -> None:
        self.if_factor, self.audio_factor = _plan_decimation(
            self.sample_rate, self.audio_rate, self.bandwidth_hz * self.if_headroom
        )
        self.channel = FirDecimator.lowpass(
            self.if_factor, self.bandwidth_hz / 2.0, self.sample_rate
        )
        self.discriminator = Discriminator()
        self.deemphasis = (
            None
            if self.deemphasis_us is None
            else Deemphasis(self.if_rate, self.deemphasis_us)
        )
        self.audio_stage = FirDecimator.lowpass(
            self.audio_factor,
            min(self.audio_cutoff_hz, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )
        # A phase step of this size means full deviation, so dividing by it
        # puts a fully modulated signal at +/-1 regardless of the IF rate.
        self._scale = self.if_rate / (2.0 * np.pi * self.deviation_hz)

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self.channel.process(iq)
        self.channel_power_dbfs = power_dbfs(channel)
        audio = self.discriminator.process(channel) * self._scale
        if self.deemphasis is not None:
            audio = self.deemphasis.process(audio)
        return self.audio_stage.process(audio)


class WfmDemodulator(_FmBase):
    mode = "wfm"
    label = "FM radio"
    description = "Ordinary broadcast FM radio, the kind car stereos receive."
    default_bandwidth_hz = 200_000.0
    deviation_hz = 75_000.0
    deemphasis_us = 75.0
    audio_cutoff_hz = 15_000.0


class NfmDemodulator(_FmBase):
    mode = "nfm"
    label = "Two-way radio"
    description = "Narrow FM: walkie-talkies, marine, weather and ham radio."
    default_bandwidth_hz = 12_500.0
    deviation_hz = 2_500.0
    # Two-way radio is not pre-emphasised the way broadcast FM is.
    deemphasis_us = None
    audio_cutoff_hz = 4_000.0
    if_headroom = 4.0


class AmDemodulator(Demodulator):
    mode = "am"
    label = "AM"
    description = "AM broadcast, shortwave and aircraft radio."
    default_bandwidth_hz = 10_000.0
    audio_cutoff_hz = 5_000.0

    def _build(self) -> None:
        self.if_factor, self.audio_factor = _plan_decimation(
            self.sample_rate, self.audio_rate, self.bandwidth_hz * 2.5
        )
        self.channel = FirDecimator.lowpass(
            self.if_factor, self.bandwidth_hz / 2.0, self.sample_rate
        )
        self.dc_block = DcBlock()
        self.audio_stage = FirDecimator.lowpass(
            self.audio_factor,
            min(self.audio_cutoff_hz, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self.channel.process(iq)
        self.channel_power_dbfs = power_dbfs(channel)
        # The envelope carries the audio, riding on the carrier as a DC term.
        envelope = np.abs(channel).astype(np.float32)
        return self.audio_stage.process(self.dc_block.process(envelope))


class _SidebandBase(Demodulator):
    """Select one sideband with a complex bandpass, then take the real part.

    A one-sided spectrum is an analytic signal, and the real part of an
    analytic signal is the audio that produced it. That is the whole trick -
    no oscillator, no Hilbert transform, just a filter that refuses to pass
    negative frequencies.
    """

    low_cut_hz = 300.0
    upper = True
    # SSB filters are narrow relative to their centre frequency, so they need
    # far more taps than the wideband stages do.
    sideband_taps = 512

    def _build(self) -> None:
        edge = self.low_cut_hz + self.bandwidth_hz
        self.if_factor, self.audio_factor = _plan_decimation(
            self.sample_rate, self.audio_rate, edge * 4.0
        )
        # Stage one only has to be wide enough not to clip the sideband; the
        # sharp edges come from stage two, where samples are 50x cheaper.
        self.channel = FirDecimator.lowpass(
            self.if_factor, min(edge * 2.0, 0.45 * self.if_rate), self.sample_rate
        )
        low, high = (self.low_cut_hz, edge) if self.upper else (-edge, -self.low_cut_hz)
        self.sideband = FirDecimator.bandpass(
            self.audio_factor, low, high, self.if_rate, taps_per_phase=self.sideband_taps
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self.channel.process(iq)
        self.channel_power_dbfs = power_dbfs(channel)
        selected = self.sideband.process(channel)
        # Doubling restores the amplitude the discarded sideband was carrying.
        return (2.0 * np.real(selected)).astype(np.float32)


class UsbDemodulator(_SidebandBase):
    mode = "usb"
    label = "Upper sideband"
    description = "Single sideband voice, standard above 10 MHz and on VHF."
    default_bandwidth_hz = 2_700.0
    upper = True


class LsbDemodulator(_SidebandBase):
    mode = "lsb"
    label = "Lower sideband"
    description = "Single sideband voice, standard on the lower ham bands."
    default_bandwidth_hz = 2_700.0
    upper = False


class CwDemodulator(_SidebandBase):
    mode = "cw"
    label = "Morse code"
    description = "Morse, shifted to an audible tone you can hear and count."
    default_bandwidth_hz = 500.0
    upper = True
    sideband_taps = 1024

    def __init__(self, *args: object, tone_hz: float = 700.0, **kwargs: object) -> None:
        # An unmodulated carrier sits at 0 Hz and is therefore silent. Offset
        # the filter so it lands on an audible pitch instead.
        self.tone_hz = float(tone_hz)
        super().__init__(*args, **kwargs)

    def _build(self) -> None:
        self.low_cut_hz = max(50.0, self.tone_hz - self.bandwidth_hz / 2.0)
        super()._build()


class DsbDemodulator(Demodulator):
    mode = "dsb"
    label = "Double sideband"
    description = "Both sidebands with the carrier suppressed."
    default_bandwidth_hz = 6_000.0

    def _build(self) -> None:
        self.if_factor, self.audio_factor = _plan_decimation(
            self.sample_rate, self.audio_rate, self.bandwidth_hz * 4.0
        )
        self.channel = FirDecimator.lowpass(
            self.if_factor, self.bandwidth_hz / 2.0, self.sample_rate
        )
        self.audio_stage = FirDecimator.lowpass(
            self.audio_factor,
            min(self.bandwidth_hz / 2.0, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self.channel.process(iq)
        self.channel_power_dbfs = power_dbfs(channel)
        return self.audio_stage.process(np.real(channel).astype(np.float32))


class RawDemodulator(DsbDemodulator):
    mode = "raw"
    label = "Raw"
    description = "The baseband signal itself, with no demodulation."
    default_bandwidth_hz = 48_000.0


MODES: dict[str, type[Demodulator]] = {
    cls.mode: cls
    for cls in (
        WfmDemodulator,
        NfmDemodulator,
        AmDemodulator,
        UsbDemodulator,
        LsbDemodulator,
        CwDemodulator,
        DsbDemodulator,
        RawDemodulator,
    )
}


def mode_table() -> list[ModeInfo]:
    """Mode metadata for the UI, so the view never hard-codes a mode list."""
    return [
        ModeInfo(cls.mode, cls.label, cls.description, cls.default_bandwidth_hz)
        for cls in MODES.values()
    ]


def create(mode: str, sample_rate: float, **kwargs: object) -> Demodulator:
    """Build a demodulator by mode name."""
    try:
        cls = MODES[mode.lower()]
    except KeyError:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {', '.join(MODES)}"
        ) from None
    return cls(sample_rate, **kwargs)


__all__ = [
    "AUDIO_RATE",
    "MODES",
    "AmDemodulator",
    "CwDemodulator",
    "Demodulator",
    "DsbDemodulator",
    "LsbDemodulator",
    "ModeInfo",
    "NfmDemodulator",
    "RawDemodulator",
    "UsbDemodulator",
    "WfmDemodulator",
    "create",
    "mode_table",
]
