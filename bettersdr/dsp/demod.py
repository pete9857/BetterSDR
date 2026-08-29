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

from .filters import (
    DEFAULT_TAPS_PER_PHASE,
    DcBlock,
    Deemphasis,
    Discriminator,
    FirDecimator,
    Squelch,
    power_dbfs,
)

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
        filter_taps: int = DEFAULT_TAPS_PER_PHASE,
    ) -> None:
        self.sample_rate = float(sample_rate)
        # Taps per polyphase branch in the channel filter. SDR# calls this the
        # filter order; more of them sharpens the skirt at linear CPU cost,
        # which is what lets a strong neighbour be pushed off a weak channel.
        self.filter_taps = max(4, int(filter_taps))
        self.audio_rate = int(audio_rate)
        self.bandwidth_hz = float(
            bandwidth_hz if bandwidth_hz is not None else self.default_bandwidth_hz
        )
        self.volume = float(volume)
        self.channel_power_dbfs = -120.0
        # Optional IF-rate stage, installed by the engine when the user turns
        # on IF noise reduction. It belongs here rather than on the raw stream
        # because after the channel filter there are ten to fifty times fewer
        # samples - measured at 33% of a core on the full 2.4 MS/s window
        # against 3% at a 240 kHz IF - and because noise outside the channel
        # is about to be thrown away regardless.
        self.if_stage: object | None = None
        self.clip = True
        self._pending = np.zeros(0, dtype=np.complex64)
        self._if_carry = np.zeros(0, dtype=np.complex64)
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
        self._if_carry = np.zeros(0, dtype=np.complex64)
        for stage in vars(self).values():
            if hasattr(stage, "reset") and stage is not self:
                stage.reset()

    def _front(self, iq: np.ndarray) -> np.ndarray:
        """Channel-filter a block, measure it, and run the optional IF stage.

        Every mode did the first two lines of this itself. Sharing them also
        gives the IF stage one place to live, and one place to solve the
        problem it creates: an overlap-add stage returns whole hops rather
        than whatever it was handed, and the audio decimator that follows
        insists on a multiple of its own factor. So the remainder is carried
        here, exactly as `process` carries the remainder of the input.

        The level is measured *before* the IF stage on purpose. It drives the
        squelch and the meter, and both should report what the radio actually
        received rather than what noise reduction made of it.
        """
        channel = self.channel.process(iq)
        self.channel_power_dbfs = power_dbfs(channel)
        if self.if_stage is None:
            return channel
        channel = self.if_stage.process(channel)
        if self._if_carry.size:
            channel = np.concatenate((self._if_carry, channel))
        usable = channel.size - (channel.size % self.audio_factor)
        self._if_carry = channel[usable:].copy()
        return channel[:usable]

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
        if not self.clip:
            # The engine turns this off when it owns the tail of the audio
            # path: an AGC that has to work behind a limiter cannot recover
            # anything the limiter has already flattened, so the limit has to
            # be the last thing that happens, not the first.
            return audio.astype(np.float32)
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
            self.if_factor,
            self.bandwidth_hz / 2.0,
            self.sample_rate,
            taps_per_phase=self.filter_taps,
        )
        self.discriminator = Discriminator()
        # A passive listener on the composite baseband - the RDS receiver.
        # It reads; it does not change what is heard.
        self.mpx_sink: object | None = None
        # The same again for the pager decoder, which reads the deviation
        # itself rather than a subcarrier riding on it. It gets a slot of its
        # own rather than sharing this one: RDS lives on a 200 kHz broadcast
        # channel and POCSAG on a 12.5 kHz two-way channel, so the two are
        # never wanted at once - but one slot would mean whichever attached
        # last silently evicted the other, which is exactly the kind of fault
        # that only shows up as a feature that stopped working.
        self.data_sink: object | None = None
        # A `dsp.stereo.StereoDecoder`, when the engine has attached one. This
        # one is not passive: it hands back a delayed sum along with the
        # difference, because the delay that aligns the multiplex with its own
        # pilot has to reach both channels or they arrive skewed.
        self.stereo: object | None = None
        self.deemphasis = self._deemphasis()
        # The difference channel gets its own copies of the two stages the sum
        # goes through, built from the same numbers. Identical filters have
        # identical group delay, which is the only reason L and R stay lined
        # up with each other once they are added and subtracted.
        self.side_deemphasis = self._deemphasis()
        self.audio_stage = self._audio_stage()
        self.side_stage = self._audio_stage()
        # A phase step of this size means full deviation, so dividing by it
        # puts a fully modulated signal at +/-1 regardless of the IF rate.
        self._scale = self.if_rate / (2.0 * np.pi * self.deviation_hz)

    def _deemphasis(self) -> Deemphasis | None:
        return (
            None
            if self.deemphasis_us is None
            else Deemphasis(self.if_rate, self.deemphasis_us)
        )

    def _audio_stage(self) -> FirDecimator:
        return FirDecimator.lowpass(
            self.audio_factor,
            min(self.audio_cutoff_hz, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self._front(iq)
        audio = self.discriminator.process(channel) * self._scale
        if self.mpx_sink is not None:
            # The multiplex, tapped before de-emphasis and before the audio
            # filter throws away everything above 15 kHz. Both are on the far
            # side of this line and both would destroy the subcarriers the
            # station puts up there - the stereo difference channel at 38 kHz
            # and RDS at 57 kHz. This is the only place they exist.
            self.mpx_sink.process(audio)
        if self.data_sink is not None:
            # The deviation itself, which is what an FSK decoder slices. The
            # audio filter below rounds the corners off the bits and a squelch
            # would mute them outright, so this is the last point at which
            # they are still square.
            self.data_sink.process(audio)
        side = None
        if self.stereo is not None:
            audio, side = self.stereo.process(audio)
        if self.deemphasis is not None:
            audio = self.deemphasis.process(audio)
        mono = self.audio_stage.process(audio)
        if side is None:
            return mono
        # Both channels were pre-emphasised at the transmitter before they
        # were matrixed, so both need the cut - de-emphasising only the sum
        # leaves the difference channel bright and the image wandering with
        # frequency.
        if self.side_deemphasis is not None:
            side = self.side_deemphasis.process(side)
        side = self.side_stage.process(side)
        count = min(mono.size, side.size)
        stereo = np.empty((count, 2), dtype=np.float32)
        stereo[:, 0] = mono[:count] + side[:count]
        stereo[:, 1] = mono[:count] - side[:count]
        return stereo


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
            self.if_factor,
            self.bandwidth_hz / 2.0,
            self.sample_rate,
            taps_per_phase=self.filter_taps,
        )
        self.dc_block = DcBlock()
        self.audio_stage = FirDecimator.lowpass(
            self.audio_factor,
            min(self.audio_cutoff_hz, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self._front(iq)
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
            self.if_factor,
            min(edge * 2.0, 0.45 * self.if_rate),
            self.sample_rate,
            taps_per_phase=self.filter_taps,
        )
        low, high = (self.low_cut_hz, edge) if self.upper else (-edge, -self.low_cut_hz)
        self.sideband = FirDecimator.bandpass(
            self.audio_factor, low, high, self.if_rate, taps_per_phase=self.sideband_taps
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self._front(iq)
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
            self.if_factor,
            self.bandwidth_hz / 2.0,
            self.sample_rate,
            taps_per_phase=self.filter_taps,
        )
        self.audio_stage = FirDecimator.lowpass(
            self.audio_factor,
            min(self.bandwidth_hz / 2.0, 0.45 * self.audio_rate),
            self.if_rate,
            dtype=np.float32,
        )

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        channel = self._front(iq)
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
