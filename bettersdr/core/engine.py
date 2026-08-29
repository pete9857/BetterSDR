"""The radio engine: everything between the dongle and the screen.

This owns the second of the three threads. The reader thread fills a ring
buffer; this drains it, demodulates to audio, measures a spectrum, and drops
the result into a single-slot mailbox. The GUI thread reads that mailbox on a
timer and never blocks, never queues, and never touches the device.

Keeping the engine out of `ui/` is deliberate: `listen.py` proves the audio
path with no Qt loaded at all, so an audio fault and a UI fault stay
distinguishable. The GUI is a view onto this, not the owner of it.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.output import AudioSink
from ..audio.record import (
    AudioRecorder,
    IqRecorder,
    RecordingLimits,
    timestamped_name,
)
from ..decode import hdradio
from ..decode.adsb import (
    FREQUENCY_HZ as ADSB_FREQUENCY_HZ,
)
from ..decode.adsb import (
    MIN_SAMPLE_RATE_HZ as ADSB_MIN_SAMPLE_RATE_HZ,
)
from ..decode.adsb import (
    AdsbReceiver,
    AdsbState,
)
from ..decode.hdradio import (
    OCCUPIED_BANDWIDTH_HZ as HD_BANDWIDTH_HZ,
)
from ..decode.hdradio import (
    SAMPLE_RATE_HZ as HD_SAMPLE_RATE_HZ,
)
from ..decode.hdradio import (
    HdRadio,
    HdState,
)
from ..decode.pocsag import (
    MIN_IF_RATE_HZ as POCSAG_MIN_IF_RATE_HZ,
)
from ..decode.pocsag import (
    PocsagReceiver,
    PocsagState,
)
from ..decode.rds import MIN_IF_RATE_HZ, RdsReceiver, RdsState
from ..dsp import convert, demod
from ..dsp.chain import AudioChain, FrontEnd
from ..dsp.denoise import SpectralNoiseReduction
from ..dsp.filters import DEFAULT_TAPS_PER_PHASE, Deemphasis
from ..dsp.psd import DEFAULT_FFT_SIZE, WINDOWS, Spectrum
from ..dsp.stereo import MIN_MPX_RATE_HZ, StereoDecoder
from ..scan.classifier import Signal
from ..scan.detector import DEFAULT_THRESHOLD_DB
from ..scan.sweeper import (
    DEFAULT_PASSES,
    DEFAULT_SETTLE_S,
    Sweeper,
    SweepProgress,
    SweepResult,
)
from .device import DEFAULT_SAMPLE_RATE, Device
from .frontend import GainChoice, choose_gain, safe_center_hz, safe_sample_rate
from .reader import Reader, read_bytes_for

# How much of the ring to take per pass. 64 KB is ~14 ms at 2.4 MS/s, which
# keeps the meter and spectrum responsive without making the loop spin.
DSP_BLOCK_BYTES = 65_536
# As with the reader's read size, what matters is the span of time a block
# covers rather than its size in bytes. 64 KB is 14 ms at 2.4 MS/s and 137 ms
# at 240 kS/s, and handing the audio sink 137 ms of audio at a time - against
# a 150 ms target buffer - underruns on every block.
DSP_BLOCK_SECONDS = DSP_BLOCK_BYTES / 2 / 2_400_000


def dsp_block_bytes_for(sample_rate_hz: float) -> int:
    raw = int(sample_rate_hz * 2 * DSP_BLOCK_SECONDS)
    return max(4_096, (raw // 512) * 512)
# Spectrum frames per second. Deliberately above the 30 Hz display rate so the
# GUI never waits on a frame, and far below the block rate so we do not compute
# frames nobody will look at.
SPECTRUM_HZ = 45.0
# How often the aircraft list is republished while ADS-B is running. Far
# slower than the spectrum on purpose: an aircraft reports about twice a
# second, so a faster rate only re-sorts a list that has not changed.
ADSB_UPDATE_HZ = 5.0
# How long an HD Radio session waits for the digital signal before handing
# the station back to the analog receiver. Acquisition was measured at 5.5 s
# on a strong local station, so this is twice that with margin: long enough
# that a station which does carry HD is never abandoned mid-search, short
# enough that leaving the switch on over a station which does not costs a few
# seconds of quiet rather than a radio that is silent until somebody works
# out why.
HD_ACQUIRE_TIMEOUT_S = 12.0
# What a block produces when the radio is doing something that is not
# listening. Shared rather than allocated per block, and shaped so the
# recorders can be handed it without a special case.
_NO_AUDIO = np.zeros(0, dtype=np.float32)


class Mailbox[T]:
    """A single slot where the newest value wins.

    Not a queue on purpose. If the GUI stalls for a moment we want it to
    resume with the current picture, not to work through a backlog of stale
    ones - dropping frames is the correct behaviour for a display.
    """

    def __init__(self) -> None:
        self._value: T | None = None
        self._lock = threading.Lock()

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value

    def peek(self) -> T | None:
        """The latest value, left in place."""
        with self._lock:
            return self._value


@dataclass(frozen=True)
class DisplayFrame:
    """One update for the GUI. Immutable, so it can cross threads freely."""

    spectrum_db: np.ndarray
    center_hz: float
    sample_rate: float
    bin_width_hz: float
    channel_power_dbfs: float
    bandwidth_hz: float
    squelch_open: bool | None
    audio_latency_s: float
    underruns: int
    ring_overruns: int
    # How much the AGC is currently adding. Worth showing rather than hiding:
    # an AGC winding 40 dB of gain into a channel is the explanation for why
    # the noise got louder when the signal went away.
    agc_gain_db: float = 0.0
    # What the station says about itself, where it says anything. None
    # means nothing is listening for it - a mode other than broadcast FM,
    # or the feature switched off - which is a different thing from a
    # station that carries no RDS, and the view says so differently.
    rds: RdsState | None = None
    # The digital programme, where one is being decoded. None means no HD
    # session is running, which is a different thing from a session that has
    # not found the signal yet - that one is running with `synced` false, and
    # the screen says so differently.
    hd: HdState | None = None
    # The pager traffic on this channel, where anything is listening for it.
    # None means nothing is - a mode or a channel too wide to carry POCSAG,
    # or the feature switched off - which is a different thing from a quiet
    # channel, and the screen says so differently.
    pocsag: PocsagState | None = None
    # Whether what just went to the sound card was two different channels.
    # Reported from the audio rather than from the pilot on purpose: audio
    # noise reduction mixes down, so a pilot-only flag would light the
    # indicator while both ears heard the same thing.
    stereo: bool = False
    # How much of the difference channel survived the blend, 0 to 1. 1.0 when
    # nothing is being blended away and when nothing is decoding stereo at
    # all, so a screen can show the number only when it is below 1 and say
    # nothing the rest of the time.
    stereo_blend: float = 1.0

    def frequencies(self) -> np.ndarray:
        """Absolute frequency of each bin, low to high."""
        bins = self.spectrum_db.size
        return self.center_hz + (np.arange(bins) - bins // 2) * self.bin_width_hz


@dataclass(frozen=True)
class ScanUpdate:
    """One update from a scan in progress, or the finished article.

    Carries the list found so far as well as the progress, so the discovery
    screen fills in as the sweep moves rather than staying empty until the end.
    `result` is set only on the last update.
    """

    progress: SweepProgress
    signals: tuple[Signal, ...]
    result: SweepResult | None = None

    @property
    def complete(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class RecordingStatus:
    """What the recorders are doing, for a status line that tells the truth."""

    audio_seconds: float = 0.0
    audio_path: str | None = None
    iq_seconds: float = 0.0
    iq_path: str | None = None
    message: str | None = None

    @property
    def active(self) -> bool:
        return self.audio_path is not None or self.iq_path is not None


class _Capture:
    """A one-shot request for raw IQ, filled by the DSP thread.

    The calibration assistant needs a block of samples that nothing has
    touched, and the DSP thread is the only place they exist. Rather than give
    another thread a way in, the request is left here and collected on the way
    past.
    """

    def __init__(self, samples: int) -> None:
        self.wanted = int(samples)
        self.blocks: list[np.ndarray] = []
        self.done = threading.Event()
        self.collected = 0

    def feed(self, iq: np.ndarray) -> None:
        take = min(iq.size, self.wanted - self.collected)
        if take > 0:
            self.blocks.append(iq[:take].copy())
            self.collected += take
        if self.collected >= self.wanted:
            self.done.set()

    def result(self) -> np.ndarray:
        if not self.blocks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(self.blocks)


class _ReaderSource:
    """Presents the reader thread to a sweeper as something tunable.

    Retuning has to be ordered against the sample stream, which is the whole
    reason the reader takes commands rather than exposing the device: the
    command runs between two reads, so once it has run every byte that arrives
    afterwards is on the new frequency. What is already in the ring is not, and
    is dropped - measuring the previous step's spectrum at this step's
    frequency would put signals in the wrong place, which is the one mistake a
    scanner must not make.
    """

    def __init__(self, reader: Reader, settle_s: float = DEFAULT_SETTLE_S) -> None:
        self.reader = reader
        self.settle_s = float(settle_s)

    def tune(self, hz: int) -> None:
        retuned = threading.Event()

        def command(device: Device) -> None:
            device.center_freq = int(hz)
            device.reset_buffer()
            retuned.set()

        self.reader.submit(command)
        retuned.wait(timeout=1.0)
        if self.settle_s > 0:
            # The tuner's PLL is still moving for a moment after the register
            # write, so these samples are neither one frequency nor the other.
            time.sleep(self.settle_s)
        self.reader.ring.clear()

    def read(self, samples: int) -> np.ndarray | None:
        raw = self.reader.ring.read(samples * 2, timeout=2.0)
        return None if raw is None else convert.to_complex(raw)


class Engine:
    """Owns the device, both worker threads, and the audio sink."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fft_size: int = DEFAULT_FFT_SIZE,
        audio_device: int | str | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.audio_device = audio_device

        self.device: Device | None = None
        self.reader: Reader | None = None
        self.sink: AudioSink | None = None
        self.gain: GainChoice | None = None
        self.last_error: str | None = None

        self.center_hz = 98_500_000
        self.mode = "wfm"
        self.volume = 0.5
        self.squelch_dbfs: float | None = None
        self.ppm = 0
        # "auto" means whatever the mode says, which is 75 microseconds for
        # broadcast FM and nothing for the rest. Only an Expert user has a
        # reason to override it, and only one of them - 50 microseconds
        # outside the Americas - is ever the right answer.
        self.deemphasis_us: float | str | None = "auto"
        self.if_noise_reduction = False
        self.if_reduction_db = 12.0
        # Taps per polyphase branch in the channel filter - SDR#'s "filter
        # order". The default is right for 8-bit data; raising it is for
        # pushing a strong neighbour off a weak channel.
        self.filter_taps = DEFAULT_TAPS_PER_PHASE
        # Costs 2.4% of a core and only runs on broadcast FM, so it is on
        # by default: a station's own name is the single most useful thing
        # this app can put on the screen.
        self.rds_enabled = True
        self._rds: RdsReceiver | None = None
        # Pager traffic, on the same tap. On by default for the same reason
        # RDS is: it costs 1.3% of a core, it only attaches on a two-way FM
        # channel narrow enough to be one, and a beginner who tunes across a
        # paging transmitter has no way of knowing to go and switch it on.
        self.pocsag_enabled = True
        self._pocsag: PocsagReceiver | None = None
        # Broadcast FM has been stereo since 1961 and the difference channel
        # is right there in the multiplex, so this is on by default too. It
        # costs nothing on any other mode, where no decoder is attached.
        self.stereo_enabled = True
        # Whether to fade the difference channel out as a station weakens.
        # The difference channel sits where FM noise is worst, so on a fringe
        # station stereo is the loudest thing about it - but that is a
        # judgement about noise rather than a fact, so it is a switch.
        self.stereo_blend = True
        self._stereo: StereoDecoder | None = None
        self._stereo_out = False
        # Aircraft tracking is a place the radio *goes*, not a decoder hung
        # off the audio path: 1090 MHz has nothing to listen to and the
        # receiver needs the whole 2.4 MHz window. So it is shaped like a
        # scan - park the audio, remember where to come back to - rather than
        # like RDS.
        self._adsb: AdsbReceiver | None = None
        # HD Radio. Shaped like neither of the above: it is a decoder, like
        # RDS, but it needs its own window and it *replaces* the sound rather
        # than annotating it, like an excursion. The switch is a standing
        # wish rather than a command - it stays on across retunes and the
        # engine starts a session wherever one is possible - because a
        # listener who wants the digital programme wants it on every station
        # that has one, not on the one station they happened to press it on.
        self.hd_enabled = False
        self.hd_program = 0
        # Why the last session stopped, for a screen that has to explain a
        # switch that is on with nothing coming out of it.
        self.hd_message = ""
        self._hd: HdRadio | None = None
        self._hd_resume_rate: int | None = None
        self._hd_resume_offset = 0.0
        self._hd_acquired = False
        self._hd_started = 0.0
        # The station and programme the running decoder was started for, so
        # retuning to where we already are does not cost a restart.
        self._hd_key: tuple[int, int] | None = None
        # Set when a station has been given its 12 seconds and produced
        # nothing. Keeps the switch on - the next station may well carry HD -
        # while stopping this one from being retried forever.
        self._hd_gave_up = False

        # The two optional chains either side of the demodulator. Both are a
        # no-op until something is switched on; see `dsp/chain.py`.
        self.front = FrontEnd(float(sample_rate))
        self.audio = AudioChain(float(demod.AUDIO_RATE))
        self.audio.volume = self.volume

        self.recording_dir = Path.home() / "BetterSDR Recordings"
        self.recording_limits = RecordingLimits()
        self._audio_recorder: AudioRecorder | None = None
        self._iq_recorder: IqRecorder | None = None
        self._recording_message: str | None = None
        self._capture: _Capture | None = None

        self._block_bytes = dsp_block_bytes_for(sample_rate)
        self._carry = np.empty(0, dtype=np.complex64)
        self._spectrum = Spectrum(fft_size=fft_size, sample_rate=float(sample_rate))
        self._demod = demod.create(self.mode, float(sample_rate), volume=1.0)
        self._demod.clip = False
        # What the demodulator *would* be built as. Kept apart from the
        # object itself because an HD Radio session runs at a window no
        # demodulator can be built for, and a mode or bandwidth chosen during
        # one has to survive until the window comes back. See `_rebuild`.
        self._wanted_mode = self.mode
        self._wanted_bandwidth_hz = self._demod.bandwidth_hz
        self._apply_rds()
        self._apply_pocsag()
        self._apply_stereo()
        self._mailbox: Mailbox[DisplayFrame] = Mailbox()
        self._scan_mailbox: Mailbox[ScanUpdate] = Mailbox()
        self._adsb_mailbox: Mailbox[AdsbState] = Mailbox()
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Owned by the DSP thread once a scan starts. Listening and scanning
        # are the same thread taking turns, not two consumers of one ring:
        # two things draining the same buffer would each get half the samples.
        self._sweeper: Sweeper | None = None
        self._scan_source: _ReaderSource | None = None
        self._resume_hz: int | None = None
        self._resume_rate: int | None = None
        # Set the instant a scan is asked for, cleared when the sweep ends.
        # See `scanning` for why this is not just "is there a sweeper".
        self._scan_wanted = threading.Event()
        # The same again for aircraft tracking, for the same reason: a view
        # polling at 20 Hz must not see "not receiving" in the gap between
        # the request and the DSP thread acting on it.
        self._adsb_wanted = threading.Event()
        # Where the tuner is parked while it is on loan. Deliberately *not*
        # `center_hz`, which means the frequency the user is listening to -
        # see `_begin_adsb`.
        self._adsb_center_hz: int | None = None
        self._adsb_resume_rate: int | None = None
        self._adsb_published = 0.0
        # Set while a gain measurement is queued on the reader thread, so two
        # callers asking at once produce one probe rather than two.
        self._gain_pending = threading.Event()
        # How many probes are actually in flight, which is a different
        # question from whether another one would be de-duplicated: the
        # excursion paths deliberately do not de-duplicate, so two can be
        # queued at once. A boolean cleared by the first of them unparked the
        # audio while the second was still running - measured at 5 to 12
        # underruns on the way back from the aircraft screen, and at none at
        # all on the same path without a view starting up beside it.
        self._probe_lock = threading.Lock()
        self._probes = 0
        # Whether audio is currently parked for the duration of one. See
        # `_run`; the DSP thread owns this and nothing else touches it.
        self._sink_parked = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def probing(self) -> bool:
        """Whether any gain measurement is running on the reader thread.

        Every probe is time with no capture in flight - 340 ms of it on a
        strong band - so this is what parks the audio. Counted rather than
        flagged because two probes can legitimately overlap.
        """
        return self._probes > 0

    def _probe_started(self) -> None:
        with self._probe_lock:
            self._probes += 1

    def _probe_finished(self) -> None:
        with self._probe_lock:
            self._probes = max(0, self._probes - 1)

    def start(self, center_hz: int | None = None) -> Engine:
        if self._thread is not None:
            return self
        if center_hz is not None:
            self.center_hz = int(center_hz)

        self.device = Device()
        self.device.open()
        self.device.configure(
            center_freq=self.device_center_hz,
            sample_rate=self.sample_rate,
            ppm=self.ppm,
        )
        # Before the reader starts, so the probe reads have the device to
        # themselves - afterwards only the reader thread may touch it.
        self.gain = choose_gain(self.device)

        self.reader = Reader(
            self.device, block_bytes=read_bytes_for(self.sample_rate)
        )
        self.reader.start()
        self.reader.wait_until_running()
        self.sink = AudioSink(rate=demod.AUDIO_RATE, device=self.audio_device).start()

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sdr-dsp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        # Before the thread is joined, so a recording in progress is closed
        # with a valid WAV header rather than left for the operating system to
        # tidy up. A truncated header makes the file unplayable, which would
        # lose the whole recording rather than its last block.
        self._end_recording(audio=True, iq=True)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # After the join, so the thread that owns it is definitely gone. A
        # child process outliving the app is the one failure mode a bundled
        # decoder must not have.
        if self._hd is not None:
            self._hd.stop()
            self._hd = None
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.sink is not None:
            self.sink.stop()
            self.sink = None
        if self.device is not None:
            self.device.close()
            self.device = None

    def __enter__(self) -> Engine:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- controls, all callable from the GUI thread ------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> DisplayFrame | None:
        return self._mailbox.peek()

    def tune(self, center_hz: int) -> None:
        # Clamped here rather than trusted from the caller: this is the one
        # choke point every tuning path goes through, and a frequency the
        # dongle cannot reach is rejected on the reader thread where nobody
        # can see it. See `frontend.safe_center_hz`.
        moved = safe_center_hz(center_hz) != self.center_hz
        self.center_hz = safe_center_hz(center_hz)
        if self.reader is not None:
            self.reader.tune(self.device_center_hz)
        if moved:
            # A station that had no HD gets another chance at the next one,
            # which is the whole point of the switch being a standing wish.
            self._hd_gave_up = False
            # And the subchannel does not travel: HD2 on one station has
            # nothing whatever to do with HD2 on the next, so asking for it
            # again on a station that has only HD1 would cost twelve seconds
            # of silence to discover.
            self.hd_program = 0
        # A new frequency is a new station, and the old one's name lingering
        # on screen over somebody else's signal is worse than a blank.
        self._commands.put(self._reset_rds)
        self._commands.put(self._reset_pocsag)
        self._commands.put(self._reset_stereo)
        # A new carrier needs a new decoder - nrsc5 cannot be moved to one -
        # and a station arrived at with the switch already on needs a session
        # starting. Queued in this order so the second is a no-op when the
        # first has already done the work.
        self._commands.put(self._restart_hd)
        self._commands.put(self._apply_hd)

    @property
    def device_center_hz(self) -> int:
        """Where the *tuner* is parked, which is not always what is displayed.

        With offset tuning on they differ: the hardware sits to one side and
        the front-end chain mixes the wanted signal back to the middle, so the
        DC spike lands somewhere harmless instead of on top of the carrier
        being listened to. Everything user-facing still talks about
        `center_hz`; only the reader is told the other number.
        """
        return safe_center_hz(self.center_hz + self.front.offset_hz)

    def auto_gain(self) -> None:
        """Re-pick the RF gain for whatever the radio is pointed at now.

        Gain used to be chosen once, in `start`, for whichever band the app
        happened to open on - and the FM band is the loudest thing the dongle
        ever sees, so that one measurement stepped the tuner down near the
        bottom of its range. Tuning from there to the AM band kept the FM
        setting and the station was barely audible; the fix from the user's
        side was to find the RF gain control and wind it up by hand, which is
        exactly the knowledge this app exists to not require.

        Runs on the reader thread because `choose_gain` reads from the device
        directly and the reader owns it. That blocks capture for the length of
        the probe, so it belongs at the moments the front end genuinely
        changed - a new band, a new window width - and not on every retune.
        """
        if self.reader is None or self._gain_pending.is_set():
            # Already queued and not yet run. A band change asks for this and
            # so does the window change it triggers, and the probe is the one
            # thing on the reader thread long enough to matter: measured at
            # ~85 ms on the AM band, where running it twice emptied the audio
            # buffer and cost 23 underruns for a single hop.
            return
        self._gain_pending.set()
        self._probe_started()

        def command(device: Device) -> None:
            try:
                self.gain = choose_gain(device)
            finally:
                self._gain_pending.clear()
                self._probe_finished()

        self.reader.submit(command)

    def set_gain(self, gain_db: float) -> None:
        if self.reader is not None:
            self.reader.set_gain(gain_db)

    def set_volume(self, volume: float) -> None:
        # A bare float assignment, so it takes effect on the next block with no
        # need to round-trip through the command queue. Applied at the end of
        # the audio chain rather than inside the demodulator, so an AGC in
        # front of it cannot spend its range undoing the setting.
        self.volume = float(volume)
        self.audio.volume = self.volume

    def set_mute(self, muted: bool) -> None:
        self.audio.mute = bool(muted)

    def set_audio_device(self, device: int | str | None) -> None:
        """Play through a different sound card.

        The stream has to be torn down and rebuilt, so this runs on the DSP
        thread - the only one that writes to the sink. A device that will not
        open leaves the old one running rather than leaving the app silent
        with no way back.
        """
        self.audio_device = device
        self._commands.put(lambda: self._swap_sink(device))

    def _swap_sink(self, device: int | str | None) -> None:
        previous = self.sink
        if previous is not None:
            previous.stop()
        try:
            self.sink = AudioSink(rate=demod.AUDIO_RATE, device=device).start()
        except Exception as exc:  # noqa: BLE001 - a bad device is not fatal
            self.last_error = f"That audio device could not be opened: {exc}"
            self.sink = previous
            if previous is not None:
                previous.start()

    def set_mode(self, mode: str, bandwidth_hz: float | None = None) -> None:
        """Swap the demodulator. Runs on the DSP thread, which owns it."""
        self.mode = mode
        self._commands.put(lambda: self._rebuild(mode, bandwidth_hz))
        # HD Radio exists only on broadcast FM, so leaving that mode ends a
        # session and coming back to it starts one.
        self._commands.put(self._apply_hd)

    def set_sample_rate(self, sample_rate_hz: int) -> None:
        """Change how wide a window the radio looks through.

        Narrowing it is how the HF bands become listenable at all - see
        `frontend.safe_sample_rate` - so this is a correctness control, not
        just a display preference. The request is clamped the same way
        everywhere else is, because a caller asking for a window that reaches
        0 Hz is asking for the upconverter's oscillator instead of a station.
        """
        if self._hd_resume_rate is not None:
            # An HD Radio session holds the window, and holds it at exactly
            # one rate: 1,488,375 S/s is fixed by the standard and there is
            # no other value nrsc5 will accept. So a request now is not a
            # request to change the window - it is a statement of what to
            # come back to when the session ends.
            #
            # `_guard_window` re-asserts the current rate on every retune,
            # and during a session that *is* the HD rate. It is not one of
            # the windows the app otherwise uses, so it is read as "leave the
            # resume window alone" rather than narrowed to the nearest one -
            # which would quietly take a listener from 2.4 MS/s to 1.44 the
            # moment they moved the dial.
            wanted = int(sample_rate_hz)
            if wanted == HD_SAMPLE_RATE_HZ:
                wanted = self._hd_resume_rate
            self._hd_resume_rate = safe_sample_rate(
                self.center_hz, preferred_hz=wanted
            )
            return
        rate = safe_sample_rate(self.center_hz, preferred_hz=int(sample_rate_hz))
        if rate == self.sample_rate:
            return
        self.sample_rate = rate
        if self.reader is not None:
            self.reader.set_sample_rate(rate)
            # After the rate command, so the probe measures through the new
            # window. Narrowing it is what moves the upconverter's oscillator
            # out of view, and the whole point of doing so is the headroom it
            # frees up - which is only collected by measuring again.
            self.auto_gain()
        self._commands.put(lambda: self._apply_sample_rate(rate))

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        self._commands.put(lambda: self._rebuild(self.mode, bandwidth_hz))

    def set_squelch(self, threshold_dbfs: float | None) -> None:
        self.squelch_dbfs = threshold_dbfs
        self._commands.put(lambda: self._rebuild(self.mode, None))

    # -- front end, display and hardware -----------------------------------

    def set_stereo(self, enabled: bool) -> None:
        """Decode the 38 kHz difference channel on broadcast FM."""
        self.stereo_enabled = bool(enabled)
        self._commands.put(self._apply_stereo)

    def set_stereo_blend(self, enabled: bool) -> None:
        """Fade towards mono as a station gets too weak to carry stereo."""
        self.stereo_blend = bool(enabled)
        self._commands.put(self._apply_stereo)

    def set_rds(self, enabled: bool) -> None:
        """Read the data a broadcast FM station sends alongside its audio."""
        self.rds_enabled = bool(enabled)
        self._commands.put(self._apply_rds)

    def set_pocsag(self, enabled: bool) -> None:
        """Read the text messages on a pager channel."""
        self.pocsag_enabled = bool(enabled)
        self._commands.put(self._apply_pocsag)

    def set_hd(self, enabled: bool) -> None:
        """Listen to the digital programme instead of the analog one.

        This is a wish rather than a command. HD Radio only exists on
        broadcast FM, needs the whole radio at a window nothing else uses,
        and takes about five and a half seconds to find - so the switch says
        "use it wherever it is available" and the engine starts and stops
        sessions to match. Retuning to a station with no HD falls back to the
        analog broadcast on its own and leaves the switch on for the next one.
        """
        self.hd_enabled = bool(enabled)
        if self.hd_enabled:
            # Pressing it again is an explicit retry, including on a station
            # that has already been given up on.
            self._hd_gave_up = False
            self.hd_message = ""
        self._commands.put(self._apply_hd)

    def set_hd_program(self, index: int) -> None:
        """Switch to another of the station's digital programmes.

        HD2 and HD3 are separate broadcasts sharing one carrier, and they are
        most of the reason to want this feature at all. Changing to one costs
        the acquisition again: nrsc5 takes a programme change only as a
        console keypress, which a pipe cannot deliver, so the only way to ask
        for a different one is a new process.
        """
        program = max(0, min(hdradio.MAX_PROGRAMS - 1, int(index)))
        if program == self.hd_program:
            return
        self.hd_program = program
        self._hd_gave_up = False
        self._commands.put(self._restart_hd)
        self._commands.put(self._apply_hd)

    @property
    def hd_available(self) -> bool:
        """Whether this build has a decoder to run at all."""
        return hdradio.available()

    @property
    def hd_running(self) -> bool:
        radio = self._hd
        return radio is not None and radio.running

    def set_deemphasis(self, tau_us: float | str | None) -> None:
        """`"auto"`, `None` for off, or 50 / 75 microseconds."""
        self.deemphasis_us = tau_us
        self._commands.put(self._apply_deemphasis)

    def set_if_noise_reduction(
        self, enabled: bool, reduction_db: float | None = None
    ) -> None:
        self.if_noise_reduction = bool(enabled)
        if reduction_db is not None:
            self.if_reduction_db = float(reduction_db)
        self._commands.put(self._apply_if_noise_reduction)

    def set_filter_taps(self, taps_per_phase: int) -> None:
        """How sharp the channel filter's skirt is, in taps per branch."""
        self.filter_taps = max(4, int(taps_per_phase))
        self._commands.put(lambda: self._rebuild(self.mode, self._demod.bandwidth_hz))

    def set_offset_tuning(self, offset_hz: float) -> None:
        """Park the tuner to one side and mix the signal back in software.

        Both halves have to move together, so this is not just a chain
        setting: the shift is only correct if the hardware actually went where
        the chain thinks it did.
        """
        self.front.set_offset_hz(offset_hz)
        if self._hd_resume_rate is not None:
            # The HD decoder is fed the bytes straight off the ring, ahead of
            # the front-end chain, so a shift applied in software would never
            # reach it. The offset is put away for the session - see
            # `_begin_hd` - and this only changes what comes back.
            self._hd_resume_offset = float(offset_hz)
            self.front.set_offset_hz(0.0)
            return
        if self.reader is not None:
            self.reader.tune(self.device_center_hz)

    def set_display(
        self,
        fft_size: int | None = None,
        window: str | None = None,
        smoothing: float | None = None,
    ) -> None:
        """Change how the spectrum is measured.

        Not merely a display preference: the scanner shares `dsp/psd.py` and
        inherits the FFT size, so these settings change what a detection
        threshold is measured against as well as what the picture looks like.
        The levels stay calibrated dBFS through all of it, which is what makes
        that safe.
        """
        if window is not None and window not in WINDOWS:
            raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")
        self._commands.put(lambda: self._apply_display(fft_size, window, smoothing))

    @property
    def fft_size(self) -> int:
        return self._spectrum.fft_size

    def set_ppm(self, ppm: int) -> None:
        """Correct for the dongle's crystal being a few parts per million out."""
        self.ppm = int(ppm)
        if self.reader is None:
            return

        def command(device: Device) -> None:
            device.freq_correction_ppm = self.ppm
            # The correction is applied when a frequency is programmed, so the
            # tuner keeps the old one until it is told to move. Without this
            # the calibration appears to do nothing until the next retune.
            device.center_freq = self.device_center_hz

        self.reader.submit(command)

    def set_bias_tee(self, enabled: bool) -> None:
        """Switch the antenna port's 4.5 V supply.

        Never called without the user having agreed to it in words: it feeds
        DC into whatever is plugged in, and equipment that was not expecting
        it can be damaged. `Device.set_bias_tee` says the same thing; this
        repeats it because the engine is where the GUI reaches it.
        """
        if self.reader is not None:
            self.reader.submit(lambda device: device.set_bias_tee(bool(enabled)))

    def set_tuner_agc(self, enabled: bool) -> None:
        """Hand gain control to the tuner, or take it back.

        Turning it on abandons the gain `choose_gain` measured. That is a
        legitimate Expert choice on a quiet band and a poor one on the FM
        band, where the front end overloads.
        """
        if self.reader is None:
            return
        if enabled:
            self.reader.set_gain(None)
        elif self.gain is not None:
            self.reader.set_gain(self.gain.gain_db)

    def set_digital_agc(self, enabled: bool) -> None:
        """The RTL2832U's own digital AGC, which is not the tuner's."""
        if self.reader is not None:
            self.reader.submit(lambda device: device.set_agc(bool(enabled)))

    # -- recording ---------------------------------------------------------

    @property
    def recording(self) -> RecordingStatus:
        audio, iq = self._audio_recorder, self._iq_recorder
        return RecordingStatus(
            audio_seconds=audio.seconds if audio is not None else 0.0,
            audio_path=str(audio.path) if audio is not None else None,
            iq_seconds=iq.seconds if iq is not None else 0.0,
            iq_path=str(iq.path) if iq is not None else None,
            message=self._recording_message,
        )

    def start_recording(self, audio: bool = False, iq: bool = False) -> None:
        """Open a recording. Runs on the DSP thread, which owns the files."""
        self._commands.put(lambda: self._begin_recording(audio, iq))

    def stop_recording(self, audio: bool = True, iq: bool = True) -> None:
        self._commands.put(lambda: self._end_recording(audio, iq))

    # -- calibration -------------------------------------------------------

    def capture(self, seconds: float = 0.5, timeout: float = 5.0) -> np.ndarray | None:
        """Collect raw IQ off the DSP thread, for the calibration assistant.

        Blocks the caller. That is acceptable for an explicit "measure this"
        button and would not be for anything on the display path, which is why
        this is the only synchronous route into the sample stream.
        """
        if not self.running or self.scanning:
            return None
        request = _Capture(int(self.sample_rate * max(0.05, seconds)))
        self._capture = request
        try:
            if not request.done.wait(timeout):
                return None
        finally:
            self._capture = None
        return request.result()

    # -- scanning ----------------------------------------------------------

    @property
    def scanning(self) -> bool:
        """Whether a scan is wanted, not whether one has started yet.

        Deliberately not `self._sweeper is not None`. Starting a scan queues a
        command for the DSP thread, so the sweeper appears a fraction of a
        second after the caller asks for it - and a view polling at 20 Hz sees
        "not scanning" in that gap and concludes the scan has already finished.
        Off air that left the discovery screen stuck on "Listening around
        108.7 MHz" after a completed sweep, because its one chance to write the
        summary had been spent before the sweep began.
        """
        return self._scan_wanted.is_set()

    def scan_update(self) -> ScanUpdate | None:
        """The latest news from a scan, or the finished result. Never blocks."""
        return self._scan_mailbox.peek()

    def start_scan(
        self,
        low_hz: float,
        high_hz: float,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        passes: int = DEFAULT_PASSES,
        sample_rate_hz: int | None = None,
    ) -> None:
        """Sweep a range and report what is in it.

        Audio stops for the duration - the radio is somewhere else entirely
        for most of it - and resumes on the frequency it was tuned to before.
        """
        if self.reader is None or self.scanning or self.receiving_adsb:
            return
        # The window belongs to the band being swept, not to whatever the
        # receiver happens to be listening through. This used to fall back to
        # `self.sample_rate`, and listening to an AM station leaves that at
        # 240 kHz: a sweep of FM broadcast started afterwards planned 141
        # steps instead of 12, through a window narrower than one FM station,
        # so every width and shape it measured was wrong as well as slow.
        # A band that wants something narrower says so in `sample_rate_hz`.
        #
        # The bottom of the range is where the window is most likely to reach
        # below 0 Hz, so that is what the guard is asked about. Sweeping the
        # AM band at 2.4 MHz would otherwise report the upconverter's own
        # oscillator as the strongest station in Seattle.
        rate = safe_sample_rate(
            low_hz, preferred_hz=int(sample_rate_hz or DEFAULT_SAMPLE_RATE)
        )
        sweeper = Sweeper(
            low_hz,
            high_hz,
            float(rate),
            fft_size=self._spectrum.fft_size,
            threshold_db=threshold_db,
            passes=passes,
        )
        self._scan_wanted.set()
        self._scan_mailbox.put(ScanUpdate(progress=sweeper.progress, signals=()))
        self._commands.put(lambda: self._begin_scan(sweeper))

    def stop_scan(self) -> None:
        """Abandon a scan and go back to listening."""
        self._commands.put(self._end_scan)

    # -- aircraft ----------------------------------------------------------

    @property
    def receiving_adsb(self) -> bool:
        """Whether aircraft tracking is wanted, not whether it has started.

        Same reasoning as `scanning`: the request is a command for the DSP
        thread, so a view polling at 20 Hz would otherwise see "not receiving"
        in the gap and conclude the radio had given up.
        """
        return self._adsb_wanted.is_set()

    def adsb_update(self) -> AdsbState | None:
        """The current sky, or None if nothing has been decoded yet."""
        return self._adsb_mailbox.peek()

    def start_adsb(self) -> None:
        """Point the radio at 1090 MHz and read what aircraft say about
        themselves.

        This takes the radio over completely - a different frequency, the full
        window, its own gain - so it is shaped like a scan rather than like
        RDS: audio stops for the duration and the frequency being listened to
        comes back afterwards. There is nothing to listen to at 1090 MHz; the
        signal is a 1 Mbit/s data burst and the audio path would only hiss.
        """
        if self.reader is None or self.scanning or self.receiving_adsb:
            return
        # Emptied here rather than when the receiver is built, so a view
        # polling every 200 ms cannot repopulate itself from the last
        # session's aircraft in the gap before the DSP thread acts.
        self._adsb_mailbox.put(AdsbState())
        self._adsb_wanted.set()
        self._commands.put(self._begin_adsb)

    def stop_adsb(self) -> None:
        """Go back to the frequency that was being listened to."""
        self._commands.put(self._end_adsb)

    # -- DSP thread --------------------------------------------------------

    def _begin_scan(self, sweeper: Sweeper) -> None:
        if self.reader is None:
            return
        # First, so the window it hands back is the one recorded below as
        # what to resume on. A sweep must take nothing from the band it was
        # listening to, and the HD window is the least representative one the
        # app ever holds.
        self._end_hd()
        self._resume_hz = self.center_hz
        self._resume_rate = self.sample_rate
        self._sweeper = sweeper
        if int(sweeper.sample_rate) != self.sample_rate:
            self.sample_rate = int(sweeper.sample_rate)
            if self.reader is not None:
                self.reader.set_sample_rate(self.sample_rate)
            self._apply_sample_rate(self.sample_rate)
        self._scan_source = _ReaderSource(self.reader)
        if self.sink is not None:
            # Rather than let it starve for the length of the sweep: an
            # underrun count is how the audio path reports a real fault, and
            # filling it with expected ones would make it useless.
            self.sink.stop()
        self._probe_scan_gain(sweeper)

    def _probe_scan_gain(self, sweeper: Sweeper) -> None:
        """Measure the gain for the band about to be swept.

        Gain belongs to the band, and a sweep points the front end somewhere
        entirely different from wherever it was listening. Arriving from an AM
        station the tuner sits near 34 dB, which is over 20 dB more than the
        FM band takes without pinning the 8-bit ADC at the rails - and a
        clipped front end manufactures spurs right across the sweep, which the
        detector then dutifully reports as stations. The other direction is
        just as wrong and quieter about it: sweeping the AM band on the FM
        band's 8-12 dB leaves 25 dB of signal on the table, so the weak half
        of the list never crosses the threshold.

        Measured at the *first step's* frequency rather than where the radio
        was, which is the whole point, and deliberately not through
        `auto_gain`: that probes wherever the tuner happens to be parked, and
        it de-duplicates against `_gain_pending`, which would let a probe
        queued moments before the scan stand in for this one.

        No sink parking here - the audio is already stopped for the length of
        the sweep - and no `_gain_pending` either, so the measurement
        `_end_scan` takes on the way back is not suppressed by this one.
        """
        reader = self.reader
        if reader is None:
            return
        first = safe_center_hz(sweeper.current_hz)
        self._probe_started()

        def command(device: Device) -> None:
            try:
                device.center_freq = first
                device.reset_buffer()
                self.gain = choose_gain(device)
            except Exception as exc:  # noqa: BLE001 - never abandon the sweep
                # A refused command is a diagnosable condition, not a reason
                # to stop: the sweep still measures, just at the old gain.
                self.last_error = f"Could not measure the gain to scan with: {exc}"
            finally:
                self._probe_finished()

        reader.submit(command)

    def _end_scan(self) -> None:
        sweeper, self._sweeper = self._sweeper, None
        # `stop_scan` queues this unconditionally and a sweep that ends on its
        # own has already run it, so a second pass through here must not
        # repeat the work - a gain probe in particular is 340 ms of dead air.
        swept = sweeper is not None
        self._scan_source = None
        self._scan_wanted.clear()
        if sweeper is not None:
            self._scan_mailbox.put(
                ScanUpdate(
                    progress=sweeper.progress,
                    signals=sweeper.signals(),
                    result=sweeper.result(),
                )
            )
        if self._resume_rate is not None:
            # Before the retune, so the reader's queue puts the window back
            # the way it was and only then moves the tuner.
            if self._resume_rate != self.sample_rate:
                self.sample_rate = self._resume_rate
                if self.reader is not None:
                    self.reader.set_sample_rate(self.sample_rate)
                self._apply_sample_rate(self.sample_rate)
            self._resume_rate = None
        if self._resume_hz is not None:
            self.tune(self._resume_hz)
            self._resume_hz = None
        if swept:
            # After the retune, so the probe measures at the frequency being
            # returned to. `_probe_scan_gain` set the front end up for the
            # band that was swept, which is not the band being listened to
            # unless the user scanned the one they were already on; leaving it
            # would hand back a station 25 dB down, or a clipped one.
            self.auto_gain()
        if self.sink is not None:
            self.sink.start()
        # Queued rather than called, so it runs after the `_restart_hd` that
        # the retune above put on the queue - which would otherwise restart a
        # session started here a moment earlier and pay the acquisition twice.
        self._commands.put(self._apply_hd)

    def _scan_step(self) -> None:
        sweeper, source = self._sweeper, self._scan_source
        if sweeper is None or source is None:
            return
        if sweeper.complete:
            self._end_scan()
            return

        source.tune(sweeper.current_hz)
        iq = source.read(sweeper.dwell_samples)
        if iq is None:
            self._end_scan()
            return
        sweeper.feed(iq)
        self._publish_scan_frame(sweeper)
        self._scan_mailbox.put(
            ScanUpdate(progress=sweeper.progress, signals=sweeper.signals())
        )
        if sweeper.complete:
            self._end_scan()

    def _publish_scan_frame(self, sweeper: Sweeper) -> None:
        """Keep the spectrum and waterfall alive while the sweep runs.

        A frozen display for the length of a scan reads as a hang. This shows
        the window actually moving across the band, which is both honest and
        the best explanation of what scanning is that the app can give.
        """
        spectrum_db = sweeper.last_spectrum_db
        if spectrum_db is None or self.reader is None:
            return
        self._mailbox.put(
            DisplayFrame(
                spectrum_db=spectrum_db,
                center_hz=sweeper.last_center_hz,
                sample_rate=float(self.sample_rate),
                bin_width_hz=self._spectrum.bin_width_hz,
                channel_power_dbfs=float(np.max(spectrum_db)),
                bandwidth_hz=self._demod.bandwidth_hz,
                squelch_open=None,
                audio_latency_s=0.0,
                underruns=0 if self.sink is None else self.sink.underruns,
                ring_overruns=self.reader.ring.overruns,
            )
        )

    # -- aircraft, on the DSP thread ---------------------------------------

    def _begin_adsb(self) -> None:
        """Take the radio to 1090 MHz and start the receiver.

        Ordered the way `_begin_scan` is, and for the same reasons: the window
        is set before the tuner moves, so the reader's queue puts both in
        place in one go, and the gain is measured last, at the frequency the
        receiver will actually run on. The audio sink is not touched here -
        `_run` parks it for as long as `self._adsb` is set, which keeps one
        mechanism responsible for parking rather than two.

        The tuner is moved directly rather than through `tune`, and that is
        the line that matters here. `center_hz` means *the frequency the user
        is listening to*, and every view reads it to decide what the radio is
        pointed at - so a screen shown during the excursion configures itself
        for 1090 MHz. Measured off air: leaving this screen took the listening
        screen through the band plan's aircraft entry, which gave an FM
        station `raw` mode and the 49.6 dB the quiet 1090 MHz band had asked
        for - 50 dB into overload, with no RDS and no stereo. A sweep borrows
        the tuner without touching `center_hz` for exactly this reason; so
        does this.
        """
        if self.reader is None:
            return
        # Before the window is recorded, for the same reason as `_begin_scan`:
        # what comes back has to be the window that was being listened
        # through, not the one HD Radio was borrowing.
        self._end_hd()
        self._adsb_resume_rate = self.sample_rate
        # ADS-B is 1 Mbit/s and the pulses are half a microsecond, so this is
        # the one feature in the app with a hard floor under the window width
        # rather than a preference about it. `safe_sample_rate` cannot narrow
        # anything at 1090 MHz - it only guards windows that would reach 0 Hz
        # - so the full rate is what arrives.
        rate = safe_sample_rate(
            ADSB_FREQUENCY_HZ, preferred_hz=int(DEFAULT_SAMPLE_RATE)
        )
        if rate < ADSB_MIN_SAMPLE_RATE_HZ:
            self._adsb_wanted.clear()
            self.last_error = (
                "Aircraft tracking needs a wider window than this radio can "
                "give at 1090 MHz."
            )
            return
        if rate != self.sample_rate:
            self.sample_rate = rate
            self.reader.set_sample_rate(rate)
            self._apply_sample_rate(rate)
        self._adsb_center_hz = safe_center_hz(ADSB_FREQUENCY_HZ)
        # Offset tuning is carried the same way `device_center_hz` carries it,
        # so the front end's shift lands the signal in the middle and the
        # display label stays honest. The decoder itself could not care less:
        # it works on magnitude, which a frequency shift does not change.
        self.reader.tune(safe_center_hz(ADSB_FREQUENCY_HZ + self.front.offset_hz))
        self._adsb = AdsbReceiver(float(self.sample_rate))
        self._adsb_published = 0.0
        self._probe_gain_directly("aircraft tracking")

    def _probe_gain_directly(self, what: str) -> None:
        """Measure the gain wherever the tuner is now, without de-duplicating.

        Gain belongs to the band, and 1090 MHz is the quietest the app ever
        visits: an aircraft 200 km away arrives a few dB above the noise, and
        going there on the FM band's 8-12 dB would throw away most of the sky.
        `choose_gain` steps down only for clipping, so on a quiet band it
        accepts a high setting almost at once. Coming back is the same problem
        in reverse - 49.6 dB measured at 1090 MHz turns an FM station into a
        solid wall.

        Deliberately not through `auto_gain`, at either end. That suppresses a
        probe while one is already queued, and the queued one may have been
        measured somewhere else entirely: the listening screen asks for a gain
        the instant it is shown, which during an aircraft session means
        1090 MHz. Submitting directly puts this measurement behind the retune
        in the reader's queue, so it is the last word. `_probe_scan_gain`
        exists for the mirror image of this.

        It still joins the probe count, because this is 340 ms on the reader
        thread with no capture in flight and the audio has to be parked for
        it. It does not touch `_gain_pending`: that flag means "another probe
        would be redundant", which is exactly what this one is not.
        """
        reader = self.reader
        if reader is None:
            return
        self._probe_started()

        def command(device: Device) -> None:
            try:
                self.gain = choose_gain(device)
            except Exception as exc:  # noqa: BLE001 - never abandon reception
                self.last_error = f"Could not measure the gain for {what}: {exc}"
            finally:
                # In a `finally` for the same reason `auto_gain` does it: a
                # count left standing parks the audio for good.
                self._probe_finished()

        reader.submit(command)

    def _end_adsb(self) -> None:
        """Put the radio back where it was listening.

        `stop_adsb` queues this unconditionally and it is also reached when
        the window is taken below what the receiver needs, so like `_end_scan`
        a second pass must not repeat the work - a gain probe in particular is
        340 ms of dead air.
        """
        receiver, self._adsb = self._adsb, None
        was_receiving = receiver is not None
        self._adsb_wanted.clear()
        self._adsb_center_hz = None
        if self._adsb_resume_rate is not None:
            # Before the retune, so the window is back the way it was and only
            # then does the tuner move. Same order as `_end_scan`.
            if self._adsb_resume_rate != self.sample_rate:
                self.sample_rate = self._adsb_resume_rate
                if self.reader is not None:
                    self.reader.set_sample_rate(self.sample_rate)
                self._apply_sample_rate(self.sample_rate)
            self._adsb_resume_rate = None
        if was_receiving and self.reader is not None:
            # `center_hz` never moved, so this is a retune only in the
            # hardware's terms: it puts the tuner back under the frequency the
            # user believed they were on the whole time.
            self.reader.tune(self.device_center_hz)
            # After the retune, so the probe measures at the frequency being
            # returned to. 1090 MHz takes a far higher gain than any broadcast
            # band, and leaving it would hand back a clipped station.
            self._probe_gain_directly("listening again")
        # Queued for the same reason as in `_end_scan`.
        self._commands.put(self._apply_hd)

    def _feed_adsb(self, iq: np.ndarray) -> None:
        """Hand one block to the receiver and republish the sky, slowly.

        The receiver keeps its own clock from the sample count, so it is fed
        every block; only the snapshot is throttled. An aircraft reports about
        twice a second, so rebuilding the list at the block rate would sort
        the same list forty times over.
        """
        receiver = self._adsb
        if receiver is None:
            return
        receiver.process(iq)
        now = time.perf_counter()
        if now - self._adsb_published < 1.0 / ADSB_UPDATE_HZ:
            return
        self._adsb_published = now
        self._adsb_mailbox.put(receiver.snapshot())

    def _apply_adsb_rate(self, rate: int) -> None:
        """Follow a window change, or give up honestly if it is too narrow.

        Every timing number in the receiver is derived from the sample rate,
        so one that outlived a rate change would be reading a grid that had
        moved. Below the floor there is no receiver to rebuild: reception
        stops and says so, rather than sitting there decoding nothing while
        the screen claims to be listening.

        Stopping this way still hands the tuner back - otherwise it would sit
        at 1090 MHz while `center_hz` said 94.9 and the app would simply have
        gone deaf - but it keeps the window the user just asked for. Only what
        was borrowed is returned.
        """
        if self._adsb is None:
            return
        if rate >= ADSB_MIN_SAMPLE_RATE_HZ:
            self._adsb = AdsbReceiver(float(rate))
            return
        self._adsb = None
        self._adsb_wanted.clear()
        self._adsb_center_hz = None
        self._adsb_resume_rate = None
        if self.reader is not None:
            self.reader.tune(self.device_center_hz)
            self._probe_gain_directly("listening again")
        self.last_error = (
            "Aircraft tracking stopped: the window is now too narrow to "
            "read 1 Mbit/s data bursts."
        )

    # -- HD Radio, on the DSP thread ---------------------------------------

    def _hd_possible(self) -> bool:
        """Whether a session could run right now.

        Every one of these can change under the user, which is why the answer
        is recomputed rather than remembered: a decoder to run, broadcast FM
        to point it at, and a radio that is not already on loan to a sweep or
        to the aircraft screen. `wfm` is the test for the band because HD
        Radio is an FM-band standard - on any other mode there is nothing for
        nrsc5 to find, and it would spend the whole session saying so.
        """
        return (
            self.hd_enabled
            and not self._hd_gave_up
            and self.mode == "wfm"
            and self.reader is not None
            and self._sweeper is None
            and self._adsb is None
            and hdradio.available()
        )

    def _apply_hd(self) -> None:
        """Bring the session into line with what is currently possible."""
        if self._hd_possible():
            if self._hd is None:
                self._begin_hd()
        elif self._hd is not None:
            self._end_hd()

    def _begin_hd(self) -> None:
        """Hand the station over to the digital decoder.

        Shaped like `_begin_adsb`, for the same reasons, with one difference
        that matters: **the frequency does not move.** HD Radio rides on the
        same carrier as the analog broadcast, so this borrows the window and
        nothing else, and `center_hz` stays exactly where the listener put it.

        What it borrows is unusual enough to state plainly. 1,488,375 S/s is
        fixed by the NRSC-5 standard and is not a whole multiple of 48 kHz -
        31.0078 - so for as long as this holds the radio there is **no
        demodulator at all**: `_rebuild` has nothing it can build and the
        analog audio path is not running. That is why the digital programme
        replaces the sound rather than being mixed into it, and it is the one
        thing to hold in mind for the rest of this section.

        Ordered the way the other two excursions are: the window before the
        tuner, the gain last and at the window the session will run through.
        """
        if self.reader is None or self._hd is not None:
            return
        if not self._start_hd_decoder():
            return
        self._hd_resume_rate = self.sample_rate
        # Offset tuning is put away for the duration. The decoder reads the
        # raw bytes off the ring, ahead of the front-end chain, so a shift
        # applied in software never reaches it - nrsc5 would be handed a
        # station sitting a few hundred kHz off the middle of its window and
        # would search for a frame that was not there. It comes back with the
        # window.
        self._hd_resume_offset = self.front.offset_hz
        moved = self._hd_resume_offset != 0.0
        if moved:
            self.front.set_offset_hz(0.0)
        self.sample_rate = HD_SAMPLE_RATE_HZ
        self.reader.set_sample_rate(HD_SAMPLE_RATE_HZ)
        self._apply_sample_rate(HD_SAMPLE_RATE_HZ)
        if moved:
            self.reader.tune(self.device_center_hz)
        # Narrowing from 2.4 MS/s takes the neighbouring channels out of the
        # window, which is headroom the front end can spend - and the digital
        # sidebands sit 12 to 18 dB below the carrier, so it is headroom the
        # part we are here for actually needs. Free at this moment either
        # way: nothing is playing for the next few seconds regardless.
        self._probe_gain_directly("HD Radio")
        # Last, and that order is the whole of it: the probe reads the
        # device directly, which cannot happen while a stream is running.
        # From here on several USB transfers stay in flight, because the
        # gap between one read and the next is the difference between this
        # decoding and not - see `core/reader.py`.
        self.reader.set_gapless(True)

    def _start_hd_decoder(self) -> bool:
        """Launch nrsc5 for the current station and programme.

        Touches neither the window nor the gain, so `_restart_hd` can use it
        for a programme change without paying for either.
        """
        radio = HdRadio(program=self.hd_program, audio_rate=demod.AUDIO_RATE)
        if not radio.start():
            # A build with no decoder, or one somebody deleted. The switch
            # goes off rather than sitting on over silence.
            self.hd_enabled = False
            self.hd_message = (
                radio.snapshot().error or "The HD Radio decoder would not start."
            )
            self.last_error = self.hd_message
            return False
        self._hd = radio
        self._hd_acquired = False
        self._hd_started = time.perf_counter()
        self._hd_key = (int(self.center_hz), self.hd_program)
        self.hd_message = ""
        return True

    def _restart_hd(self) -> None:
        """Point the decoder at whatever the radio is on now.

        nrsc5 locks onto one carrier and one programme and neither can be
        changed by talking to it, so a new station or a new subchannel costs
        a new process and the acquisition with it. The window and the gain
        are already right, so nothing else moves - and a retune that did not
        actually go anywhere costs nothing at all.
        """
        radio = self._hd
        if radio is None:
            return
        if self._hd_key == (int(self.center_hz), self.hd_program):
            return
        self._hd = None
        radio.stop()
        if not self._start_hd_decoder():
            self._end_hd()

    def _end_hd(self) -> None:
        """Give the station back to the analog receiver.

        Everything borrowed goes back in the order it was taken - the offset,
        then the window, then the frequency, then a fresh gain measurement at
        the window being returned to. Putting the window back is what rebuilds
        the demodulator, so it is also the moment the analog path exists
        again.

        Reached from several directions and sometimes twice, so like
        `_end_scan` and `_end_adsb` it must not repeat the work: a gain probe
        is 340 ms of dead air.
        """
        radio, self._hd = self._hd, None
        if radio is not None:
            radio.stop()
        self._hd_acquired = False
        self._hd_key = None
        borrowed = self._hd_resume_rate is not None
        if borrowed and self.reader is not None:
            # Before everything below, because none of it can touch the
            # device until the stream has been torn down: a rate change, a
            # retune and a gain probe all talk to it directly.
            self.reader.set_gapless(False)
        if self._hd_resume_offset:
            self.front.set_offset_hz(self._hd_resume_offset)
            self._hd_resume_offset = 0.0
        if self._hd_resume_rate is not None:
            if self._hd_resume_rate != self.sample_rate:
                self.sample_rate = self._hd_resume_rate
                if self.reader is not None:
                    self.reader.set_sample_rate(self.sample_rate)
                self._apply_sample_rate(self.sample_rate)
            self._hd_resume_rate = None
        if borrowed and self.reader is not None:
            self.reader.tune(self.device_center_hz)
            self._probe_gain_directly("listening again")

    def _feed_hd(self, raw: np.ndarray) -> np.ndarray:
        """Give the decoder a block and take back whatever it has finished.

        The bytes go across untouched: `cu8` on nrsc5's stdin is byte for
        byte what the ring buffer already holds, so the whole conversion is
        the absence of one.
        """
        radio = self._hd
        if radio is None:
            return _NO_AUDIO
        radio.feed(raw)
        if not radio.running:
            self.hd_message = radio.snapshot().error or "The HD Radio decoder stopped."
            self._end_hd()
            return _NO_AUDIO
        audio = radio.audio()
        if audio.shape[0]:
            return audio
        if (
            not self._hd_acquired
            and time.perf_counter() - self._hd_started > HD_ACQUIRE_TIMEOUT_S
        ):
            # This station does not carry HD, or does not carry it strongly
            # enough to decode indoors. Falling back is the whole reason the
            # switch survives it: leaving a receiver silent because a digital
            # signal it never advertised failed to appear is exactly the kind
            # of dead end a beginner cannot diagnose.
            self._hd_gave_up = True
            self.hd_message = (
                "Could not pick up HD Radio here - playing the normal "
                "broadcast instead."
            )
            self._end_hd()
        return _NO_AUDIO

    def _apply_sample_rate(self, rate: int) -> None:
        """Rebuild everything that was built around the old rate.

        The spectrum and the demodulator both bake the rate into their filter
        design, and `_publish_scan_frame` reports this spectrum's bin width
        alongside the sweeper's data - so letting the two disagree would put
        every frequency on the display wrong.
        """
        # The ring was emptied when the rate changed, so the sink is about to
        # run dry for as long as it takes to refill. That starvation is
        # expected, and letting it land in the underrun count is how the one
        # number that reports a real audio fault stops meaning anything -
        # measured at 25 underruns for a single hop from FM down to the AM
        # band. Same reasoning as `_begin_scan`; `start` re-primes with
        # silence and keeps the running total.
        if self.sink is not None:
            self.sink.stop()
        self._block_bytes = dsp_block_bytes_for(rate)
        # Samples captured at the old rate must not end up in a frame measured
        # at the new one.
        self._carry = np.empty(0, dtype=np.complex64)
        self._spectrum = Spectrum(
            fft_size=self._spectrum.fft_size,
            sample_rate=float(rate),
            window=self._spectrum.window_name,
            smoothing=self._spectrum.smoothing,
        )
        # Every filter in the front-end chain was designed around the old
        # rate, and a DC tracker with a quarter-second time constant at
        # 2.4 MS/s is a two-and-a-half second one at 240 kS/s.
        self.front.set_sample_rate(float(rate))
        # HD Radio's 1,488,375 S/s is not a whole multiple of 48 kHz, and
        # `demod.create` refuses rather than silently resampling by an
        # awkward ratio - which is right, and is why the check is here rather
        # than a try/except around it. There is no demodulator to build at
        # that window and none is needed; `_run` is feeding nrsc5 instead.
        # What the demodulator *would* be is kept on `_wanted_mode` and
        # `_wanted_bandwidth_hz`, so `_end_hd` putting the window back is
        # also what builds it again, with any choice made in the meantime.
        if rate % demod.AUDIO_RATE == 0:
            self._rebuild(self._wanted_mode, self._wanted_bandwidth_hz)
        self._apply_adsb_rate(rate)
        if self.sink is not None:
            self.sink.start()
            # Say so, whatever `_run` last believed. A rate change inside a
            # parked stretch leaves the loop thinking the audio is still
            # parked, so it never parks it again and the sink plays into
            # whatever comes next - which for the way back from the aircraft
            # screen into an HD session is a gain probe and a five-second
            # acquisition. Measured at 162 underruns on exactly that path.
            self._sink_parked = False

    def _apply_display(
        self, fft_size: int | None, window: str | None, smoothing: float | None
    ) -> None:
        current = self._spectrum
        self._spectrum = Spectrum(
            fft_size=current.fft_size if fft_size is None else int(fft_size),
            sample_rate=float(self.sample_rate),
            window=current.window_name if window is None else window,
            smoothing=current.smoothing if smoothing is None else float(smoothing),
        )
        # Samples gathered for the old transform are the wrong length for the
        # new one, and half a frame of them would put a seam in the first row.
        self._carry = np.empty(0, dtype=np.complex64)

    def _apply_deemphasis(self) -> None:
        """Override the mode's own de-emphasis, if the user asked for one.

        Set on the instance rather than plumbed through `demod.create`,
        because only the FM modes have the stage at all and the factory takes
        one keyword set for every mode.
        """
        if self.deemphasis_us == "auto" or not hasattr(self._demod, "deemphasis"):
            return
        self._demod.deemphasis = (
            None
            if self.deemphasis_us is None
            else Deemphasis(self._demod.if_rate, float(self.deemphasis_us))
        )

    def _apply_if_noise_reduction(self) -> None:
        self._demod.if_stage = (
            SpectralNoiseReduction(
                self._demod.if_rate,
                complex_input=True,
                reduction_db=self.if_reduction_db,
            )
            if self.if_noise_reduction
            else None
        )

    def _apply_rds(self) -> None:
        """Attach or detach the RDS receiver, on the DSP thread.

        Three things have to be true at once, and all of them can change under
        the user: broadcast FM, the feature on, and an IF wide enough to still
        contain a subcarrier 57 kHz off centre. Narrowing the channel filter
        past about 120 kHz removes RDS from the signal itself - so the
        receiver goes away rather than sitting there decoding nothing.
        """
        wanted = (
            self.rds_enabled
            and hasattr(self._demod, "mpx_sink")
            and self._demod.if_rate >= MIN_IF_RATE_HZ
            and self._demod.bandwidth_hz >= 2.0 * 60_000.0
        )
        if not wanted:
            self._rds = None
        elif self._rds is None or self._rds.if_rate != self._demod.if_rate:
            self._rds = RdsReceiver(self._demod.if_rate)
        if hasattr(self._demod, "mpx_sink"):
            self._demod.mpx_sink = self._rds

    def _reset_rds(self) -> None:
        if self._rds is not None:
            self._rds.reset()

    def _apply_pocsag(self) -> None:
        """Attach or detach the pager decoder, on the DSP thread.

        The same shape as `_apply_rds` with the conditions turned round.
        POCSAG is not a subcarrier riding above the audio, it *is* the
        deviation - so what it needs is a two-way FM channel rather than a
        broadcast one. The upper bound on the bandwidth is what keeps it off
        broadcast FM, where it would find nothing and cost CPU for the whole
        time somebody was listening to music.
        """
        wanted = (
            self.pocsag_enabled
            and hasattr(self._demod, "data_sink")
            and self._demod.if_rate >= POCSAG_MIN_IF_RATE_HZ
            and 10_000.0 <= self._demod.bandwidth_hz <= 50_000.0
        )
        if not wanted:
            self._pocsag = None
        elif self._pocsag is None or self._pocsag.if_rate != self._demod.if_rate:
            self._pocsag = PocsagReceiver(self._demod.if_rate)
        if hasattr(self._demod, "data_sink"):
            self._demod.data_sink = self._pocsag

    def _reset_pocsag(self) -> None:
        if self._pocsag is not None:
            self._pocsag.reset()

    def _apply_stereo(self) -> None:
        """Attach or detach the stereo decoder, on the DSP thread.

        The same three conditions as RDS, for the same reasons, with one
        number changed: the difference channel sits at 23-53 kHz rather than
        at 57, so it survives a slightly narrower IF. Narrowing the channel
        filter past about 106 kHz removes it from the signal, at which point
        the honest thing is to stop claiming the station is in stereo.
        """
        wanted = (
            self.stereo_enabled
            and hasattr(self._demod, "stereo")
            and self._demod.if_rate >= MIN_MPX_RATE_HZ
            and self._demod.bandwidth_hz >= 2.0 * 53_000.0
        )
        if not wanted:
            self._stereo = None
        elif self._stereo is None or self._stereo.sample_rate != self._demod.if_rate:
            self._stereo = StereoDecoder(self._demod.if_rate)
        if self._stereo is not None:
            self._stereo.blend_enabled = self.stereo_blend
        if hasattr(self._demod, "stereo"):
            self._demod.stereo = self._stereo

    def _reset_stereo(self) -> None:
        if self._stereo is not None:
            self._stereo.reset()

    def _rebuild(self, mode: str, bandwidth_hz: float | None) -> None:
        # Recorded before anything else, so a mode or a bandwidth chosen
        # while HD Radio holds the window is applied when the window comes
        # back rather than quietly dropped. Resolved to a number here because
        # `None` means "the mode's own default", and that has to be answered
        # while the mode is still the one being asked about.
        self._wanted_mode = mode
        self._wanted_bandwidth_hz = (
            float(bandwidth_hz)
            if bandwidth_hz is not None
            else demod.MODES[mode].default_bandwidth_hz
        )
        if self.sample_rate % demod.AUDIO_RATE:
            return
        self._demod = demod.create(
            mode,
            float(self.sample_rate),
            bandwidth_hz=bandwidth_hz,
            # Unity, with the limiter off: volume and the final clip belong to
            # `AudioChain`, at the end of the path, so the AGC in front of
            # them has the full range to work with.
            volume=1.0,
            squelch_dbfs=self.squelch_dbfs,
            filter_taps=self.filter_taps,
        )
        self._demod.clip = False
        self._apply_deemphasis()
        self._apply_if_noise_reduction()
        self._apply_rds()
        self._apply_pocsag()
        self._apply_stereo()

    # -- recording, on the DSP thread --------------------------------------

    def _begin_recording(self, audio: bool, iq: bool) -> None:
        self._recording_message = None
        if audio and self._audio_recorder is None:
            name = timestamped_name(self.center_hz, "AF")
            # The header is written once, so the channel count is decided
            # here from what is being heard right now. A station that drops
            # its pilot mid-recording keeps the file it started.
            recorder = AudioRecorder(
                self.recording_dir / name,
                demod.AUDIO_RATE,
                self.recording_limits,
                channels=2 if self._stereo_out else 1,
            ).start()
            self._audio_recorder = recorder if recorder.active else None
            self._recording_message = recorder.stopped_reason
        if iq and self._iq_recorder is None:
            name = timestamped_name(self.center_hz, "IQ")
            recorder = IqRecorder(
                self.recording_dir / name, self.sample_rate, self.recording_limits
            ).start()
            self._iq_recorder = recorder if recorder.active else None
            self._recording_message = recorder.stopped_reason or self._recording_message

    def _end_recording(self, audio: bool, iq: bool) -> None:
        if audio and self._audio_recorder is not None:
            self._audio_recorder.stop()
            self._audio_recorder = None
        if iq and self._iq_recorder is not None:
            self._iq_recorder.stop()
            self._iq_recorder = None

    def _service_recorders(self, raw: np.ndarray, audio: np.ndarray) -> None:
        """Feed both recorders, and notice when either has stopped itself.

        A recorder that hit its size limit or ran the disk low closes its own
        file. Dropping the reference here is what turns that into something
        the status line can report, rather than a recording that silently
        stopped growing.
        """
        if self._iq_recorder is not None:
            self._iq_recorder.write(raw)
            if not self._iq_recorder.active:
                self._recording_message = self._iq_recorder.stopped_reason
                self._iq_recorder = None
        if self._audio_recorder is not None and audio.size:
            self._audio_recorder.write(audio)
            if not self._audio_recorder.active:
                self._recording_message = self._audio_recorder.stopped_reason
                self._audio_recorder = None

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command()

    def _run(self) -> None:
        reader = self.reader
        if reader is None or self.sink is None:
            return
        interval = 1.0 / SPECTRUM_HZ
        last_frame = 0.0

        while not self._stop.is_set():
            self._drain_commands()
            # Re-read rather than bound once at the top of the thread:
            # choosing a different sound card replaces the sink object, and a
            # loop holding the old one would keep writing into a stream that
            # is no longer playing anywhere.
            sink = self.sink
            if sink is None:
                break
            if self._sweeper is not None:
                self._scan_step()
                continue

            # A gain probe runs on the reader thread and reads the device
            # directly, so no samples are captured while it does. That is
            # brief on a quiet band - it accepts a high gain almost at once -
            # but on the FM band it walks most of the tuner's table two reads
            # at a time, measured at about 340 ms, which is twice the jitter
            # buffer. So audio is parked for the duration rather than left to
            # starve: an underrun count full of expected underruns cannot
            # report a real fault. Same reasoning as `_begin_scan`.
            # Aircraft tracking parks it too, for its whole duration: the
            # radio is 1090 MHz away from anything anybody wanted to hear.
            # One expression, so parking cannot be started by one mechanism
            # and ended by the other.
            # An HD session parks it too, but only until the digital signal
            # arrives: acquisition is a known five and a half seconds of
            # silence, and forty expected underruns every time somebody
            # presses the switch is how the one number that reports a real
            # audio fault stops meaning anything. After that the sink stays
            # open - a drop-out mid-session is the signal going away, which
            # is worth seeing rather than papering over, and reopening the
            # sound card at every flutter would put a gap in the audio in
            # exactly the conditions that least need one.
            parked = (
                self.probing
                or self._adsb is not None
                or (self._hd is not None and not self._hd_acquired)
            )
            if parked != self._sink_parked:
                self._sink_parked = parked
                if parked:
                    sink.stop()
                else:
                    sink.start()

            raw = reader.ring.read(self._block_bytes, timeout=0.5)
            if raw is None:
                if reader.last_error:
                    self.last_error = reader.last_error
                    break
                continue

            # IQ recording is taken here, from the raw bytes, ahead of every
            # correction and every demodulator. That is deliberate: a capture
            # is meant to be replayable through software that does not exist
            # yet, and one written from further down the path would carry this
            # session's noise blanker and IQ correction baked in.
            iq = self.front.process(convert.to_complex(raw))
            if self._adsb is not None:
                # No demodulator and no audio. Mode S is a 1 Mbit/s data
                # burst: the audio path would produce hiss at the cost of
                # 63 ms per second of radio, and the sink is parked anyway.
                self._feed_adsb(iq)
                self._stereo_out = False
                audio = _NO_AUDIO
            elif self._hd is not None:
                # Fed from `raw`, not from `iq`: `cu8` is byte for byte what
                # nrsc5 reads, and the front-end chain is bypassed for the
                # decoder because a shift or a blanker applied here would be
                # something the digital receiver was never designed to see.
                # `iq` is still built above, and still drives the spectrum.
                block = self._feed_hd(raw)
                audio = self.audio.process(block) if block.size else _NO_AUDIO
                self._stereo_out = audio.ndim > 1 and audio.shape[1] > 1
                if audio.size:
                    if not self._hd_acquired:
                        # The first digital audio of the session. Unparked
                        # here rather than at the top of the next pass, so
                        # the block that ended the wait is the first one
                        # heard rather than the one thrown away.
                        self._hd_acquired = True
                        self._sink_parked = False
                        sink.start()
                    sink.write(audio)
            else:
                audio = self.audio.process(self._demod.process(iq))
                self._stereo_out = audio.ndim > 1 and audio.shape[1] > 1
                sink.write(audio)
            self._service_recorders(raw, audio)

            capture = self._capture
            if capture is not None:
                capture.feed(iq)

            # The display FFT needs a whole frame's worth of samples. A block
            # is eight frames at 2.4 MS/s but less than one at the 240 kS/s
            # the AM band uses, so the remainder is carried rather than
            # dropped - without this the spectrum silently stops updating the
            # moment the window narrows, while the audio carries on fine.
            self._carry = (
                np.concatenate((self._carry, iq)) if self._carry.size else iq
            )
            if self._carry.size < self._spectrum.fft_size:
                continue

            now = time.perf_counter()
            if now - last_frame < interval:
                self._carry = self._carry[-self._spectrum.fft_size :]
                continue
            last_frame = now
            spectrum_db = self._spectrum.process(self._carry)
            self._carry = np.empty(0, dtype=np.complex64)
            if spectrum_db.size == 0:
                continue
            # With the demodulator bypassed its last readings are from
            # whatever was being listened to minutes ago, so the strongest
            # thing in the window is reported instead - true, and about the
            # only meaningful level for a band of half-microsecond bursts.
            listening = self._adsb is None and self._hd is None
            squelch = self._demod.squelch if listening else None
            self._mailbox.put(
                DisplayFrame(
                    spectrum_db=spectrum_db,
                    # Where the samples came from, which during an aircraft
                    # session is not what the radio is tuned to in the user's
                    # terms. Labelling 1090 MHz data 94.9 MHz would put every
                    # signal on the display in the wrong place, which is the
                    # one mistake a spectrum must not make.
                    center_hz=float(
                        self.center_hz
                        if self._adsb_center_hz is None
                        else self._adsb_center_hz
                    ),
                    sample_rate=float(self.sample_rate),
                    bin_width_hz=self._spectrum.bin_width_hz,
                    channel_power_dbfs=(
                        self._demod.channel_power_dbfs
                        if listening
                        else float(np.max(spectrum_db))
                    ),
                    # During an HD session the passband marker covers the
                    # whole hybrid signal - the analog core plus both digital
                    # sidebands - which is the clearest explanation the
                    # display can give of where the extra sound is coming
                    # from. The demodulator's own figure is for a window that
                    # is not currently being used.
                    bandwidth_hz=(
                        HD_BANDWIDTH_HZ
                        if self._hd is not None
                        else self._demod.bandwidth_hz
                    ),
                    squelch_open=None if squelch is None else squelch.is_open,
                    audio_latency_s=sink.latency_s,
                    underruns=sink.underruns,
                    ring_overruns=reader.ring.overruns,
                    agc_gain_db=self.audio.agc.gain_db if self.audio.agc_enabled else 0.0,
                    # The analog path is not running during an HD session,
                    # so whatever the RDS receiver last decoded is minutes
                    # old. The digital signal carries the same information
                    # and carries it better; `hd` is where the screen reads
                    # it from.
                    rds=(
                        None
                        if self._rds is None or self._hd is not None
                        else self._rds.snapshot()
                    ),
                    hd=None if self._hd is None else self._hd.snapshot(),
                    pocsag=(
                        None if self._pocsag is None else self._pocsag.snapshot()
                    ),
                    stereo=self._stereo_out,
                    stereo_blend=(
                        1.0 if self._stereo is None else self._stereo.blend
                    ),
                )
            )


__all__ = ["DisplayFrame", "Engine", "Mailbox", "RecordingStatus", "ScanUpdate"]
