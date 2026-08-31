"""HD Radio (NRSC-5): the digital programme hiding beside the analog one.

Seven of the eight strongest local FM stations carry it, `dsp/features.py`
already finds it, and the Discover cards already say so. This is the part that
turns that badge into sound - and it is the first thing in the app that does
not do its own signal processing.

**The decoder is a separate program.** `vendor/nrsc5/win-x64/nrsc5.exe` is
bundled with BetterSDR and started as a child process; IQ goes down its
stdin, audio comes back up its stdout, and everything the station says about
itself arrives as log lines on its stderr. Three reasons, in the order any
one of them would have been sufficient:

* **The codec has no public specification.** HDC is Xperi's, derived from
  HE-AAC and documented nowhere. There is no version of this module that
  decodes the audio itself.
* **The licence.** nrsc5 is GPL-3. A separate program talking over pipes is
  mere aggregation and leaves BetterSDR's licence alone; loading `libnrsc5`
  in-process - ctypes, cffi, anything - would be linking, and would place the
  whole of BetterSDR under GPL-3. The process boundary is a legal boundary
  and must not be "simplified" away. See `vendor/nrsc5/README.md`.
* **Crash isolation.** A fault in several thousand lines of C decoding an
  8-bit stream off the air takes down a child process, not the radio.

Two consequences of that boundary shape everything below.

**The pipe must never block the DSP thread.** A write to a full pipe waits
for the far end, and the DSP thread waiting is the radio stopping. So `feed`
only appends to a bounded queue and returns; a writer thread does the
blocking, and when the queue is full the *oldest* IQ is dropped. Same policy
as `core/ringbuffer.py`, and for the same reason: falling behind should cost
the past, not the present.

**The audio comes back at 44,100 Hz and everything else here runs at
48,000.** NRSC-5 fixes it, and 1,488,375 / 48,000 is 31.0078 - not a whole
number, which is why HD cannot use the `dsp/demod.py` skeleton at all and why
the window during an HD session is one the rest of the app never otherwise
uses. `filters.RationalResampler` converts at 160/147 on the way out, so what
leaves this module looks like every other audio block in the app.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..dsp.filters import RationalResampler

# The only rate nrsc5 accepts on the pipe. A hybrid IBOC constant rather than
# a choice: it is what the NRSC-5 receiver reference is defined against.
SAMPLE_RATE_HZ = 1_488_375
# What comes back. Fixed by the standard; see the module docstring.
AUDIO_RATE_HZ = 44_100
# NRSC-5 allows up to eight audio programmes on one carrier. Most stations
# carrying HD run two or three.
MAX_PROGRAMS = 8
# How much of the dial a hybrid IBOC station actually occupies: the analog
# core plus a digital sideband reaching 198 kHz out on each side, measured
# off air and recorded under "HD Radio facts" in CLAUDE.md. Worth having as a
# number because it is what the passband marker should cover during a
# session - the picture then explains where the extra sound is coming from.
OCCUPIED_BANDWIDTH_HZ = 396_000

# How much un-sent IQ to hold. Half a second is far more than the writer
# thread ever needs - nrsc5 decodes many times faster than real time - and it
# bounds what a stalled child process can cost us at about 1.5 MB.
_INPUT_QUEUE_S = 0.5
# Audio waiting for the DSP thread to collect it. If a second piles up,
# nothing is playing it and keeping more would only add latency.
_AUDIO_QUEUE_S = 1.0
# How much of stdout to ask for at a time. The pipe is unbuffered, so this
# is a cap rather than a quantum - a read returns whatever has arrived. It
# still sets how lumpy the audio is: at 93 ms a chunk the jitter buffer
# measured 168-273 ms against a 150 ms target, which is most of the way to
# the 400 ms cap on the strength of one lump. 2048 frames is 46 ms.
_STDOUT_CHUNK = 2048 * 4
# Audio has to have arrived this recently for `playing` to be true. Long
# enough to ride out the gap between two of nrsc5's output bursts.
_PLAYING_TIMEOUT_S = 1.0

# nrsc5 prefixes every line with a wall-clock time. Everything this module
# wants is in what follows it.
_LINE = re.compile(r"^(?:\d{2}:\d{2}:\d{2}\s+)?(.*)$")
_MER = re.compile(r"^MER: (-?[\d.]+) dB \(lower\), (-?[\d.]+) dB \(upper\)")
_BER = re.compile(r"^BER: ([\d.]+)")
_BIT_RATE = re.compile(r"^Audio bit rate: ([\d.]+) kbps")
_SERVICE = re.compile(r"^Audio service (\d+): (\w+), type: ([^,]+),")

_TEXT_FIELDS = (
    ("Station name: ", "station"),
    ("Slogan: ", "slogan"),
    ("Message: ", "message"),
    ("Title: ", "title"),
    ("Artist: ", "artist"),
    ("Album: ", "album"),
    ("Genre: ", "genre"),
)


def _roots() -> list[Path]:
    """Where `vendor/nrsc5/` might be, in the order worth looking.

    Two roots, and the first expression finds two different things: in a
    checkout it is the repository, and in a frozen build it is the bundle
    directory the package was unpacked into, which is where the packaging
    puts the decoder. The second is the executable's own folder, for a build
    that lays `vendor/` beside the program instead of inside it.
    """
    launcher = sys.executable if getattr(sys, "frozen", False) else sys.argv[0]
    return [
        Path(__file__).resolve().parents[2],
        Path(launcher).resolve().parent,
    ]


def executable() -> Path | None:
    """Where the bundled decoder is, or None if this build has none.

    Everything HD Radio is gated on this returning a path. A source checkout
    without `vendor/`, a platform we have not built for, and a user who
    deleted the file should all mean "the feature is not offered" - never a
    stack trace on a button press.
    """
    override = os.environ.get("BETTERSDR_NRSC5")
    if override:
        found = Path(override)
        return found if found.is_file() else None
    name = "nrsc5.exe" if sys.platform == "win32" else "nrsc5"
    platform = "win-x64" if sys.platform == "win32" else sys.platform
    for root in _roots():
        candidate = root / "vendor" / "nrsc5" / platform / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def available() -> bool:
    return executable() is not None


@dataclass(frozen=True)
class HdProgram:
    """One of the programmes a station is carrying."""

    index: int
    kind: str = ""
    restricted: bool = False

    @property
    def label(self) -> str:
        """HD1, HD2 - what the station calls it and what a car radio shows."""
        return f"HD{self.index + 1}"


@dataclass(frozen=True)
class HdState:
    """A snapshot of the digital signal, safe to hand across threads."""

    running: bool = False
    synced: bool = False
    playing: bool = False
    program: int = 0
    programs: tuple[HdProgram, ...] = ()
    station: str = ""
    slogan: str = ""
    message: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    bit_rate_kbps: float | None = None
    mer_db: float | None = None
    ber: float | None = None
    lost_sync: int = 0
    audio_seconds: float = 0.0
    error: str = ""

    @property
    def name(self) -> str:
        """The best name the digital signal has given for the station."""
        return self.station.strip()

    @property
    def track(self) -> str:
        """What is playing, as one line, or an empty string."""
        parts = [part for part in (self.artist.strip(), self.title.strip()) if part]
        return " - ".join(parts)

    @property
    def label(self) -> str:
        """Which programme this is: HD1, HD2 and so on."""
        return f"HD{self.program + 1}"


class HdMetadata:
    """nrsc5's stderr, turned into an `HdState`.

    Kept apart from the process management so it can be tested against a
    recording of a real session rather than only against a live radio - the
    same reason `core/doctor.py` has no UI in it.
    """

    def __init__(self, program: int = 0) -> None:
        self._program = int(program)
        self._lock = threading.Lock()
        self._state = HdState(program=self._program)
        self._services: dict[int, HdProgram] = {}

    def reset(self) -> None:
        with self._lock:
            self._state = HdState(program=self._program)
            self._services = {}

    @property
    def state(self) -> HdState:
        with self._lock:
            return self._state

    def feed(self, line: str) -> None:
        """Take one line of nrsc5's log.

        Anything unrecognised is ignored rather than reported. The log also
        carries traffic maps, weather images, emergency-alert plumbing and a
        warning from the audio driver about channel mapping, and a parser that
        objected to any of them would be broken by the next nrsc5 release.
        """
        match = _LINE.match(line.strip())
        if match is None:
            return
        body = match.group(1).strip()
        if not body:
            return
        with self._lock:
            self._state = self._apply(self._state, body)

    def _apply(self, state: HdState, body: str) -> HdState:
        if body == "Synchronized":
            return replace(state, synced=True)
        if body == "Lost synchronization":
            # The station's name and the track survive. Sync comes and goes at
            # the edge of coverage, and blanking the screen every time it does
            # says "nothing is here" about a station that is still there.
            return replace(state, synced=False, lost_sync=state.lost_sync + 1)

        for prefix, field in _TEXT_FIELDS:
            if body.startswith(prefix):
                return replace(state, **{field: body[len(prefix) :].strip()})

        found = _MER.match(body)
        if found is not None:
            # One number for the screen. The weaker sideband is the one that
            # will fail first, so it is the honest one to show.
            return replace(
                state, mer_db=min(float(found.group(1)), float(found.group(2)))
            )
        found = _BER.match(body)
        if found is not None:
            return replace(state, ber=float(found.group(1)))
        found = _BIT_RATE.match(body)
        if found is not None:
            return replace(state, bit_rate_kbps=float(found.group(1)))
        found = _SERVICE.match(body)
        if found is not None:
            index = int(found.group(1))
            if 0 <= index < MAX_PROGRAMS:
                # nrsc5 prints "None" for a programme that declared no type.
                # That is a name for the absence of one, not a genre, and a
                # screen reading "HD1 - None" is worse than one reading "HD1".
                kind = found.group(3).strip()
                self._services[index] = HdProgram(
                    index=index,
                    kind="" if kind == "None" else kind,
                    restricted=found.group(2) != "public",
                )
                order = tuple(self._services[key] for key in sorted(self._services))
                return replace(state, programs=order)
        return state

    def update(self, **fields: object) -> None:
        """Set what the log does not carry - liveness, errors, counters."""
        with self._lock:
            self._state = replace(self._state, **fields)  # type: ignore[arg-type]


class HdRadio:
    """The nrsc5 child process, as a block-oriented stage.

    Used from the DSP thread exactly like a demodulator: `feed` it the raw
    bytes the ring buffer holds - which are already `cu8`, byte for byte what
    nrsc5 wants, so nothing converts them - and `audio` returns whatever has
    come back since the last call. Both return promptly whatever the child
    process is doing.

    Nothing here touches `Device`, and the three worker threads belong to one
    instance, so an HD session starting or stopping cannot disturb the reader
    or DSP threads beyond the block it happens in.
    """

    def __init__(
        self,
        program: int = 0,
        audio_rate: int = 48_000,
        path: Path | None = None,
    ) -> None:
        self.program = max(0, min(MAX_PROGRAMS - 1, int(program)))
        self.audio_rate = int(audio_rate)
        self.path = path if path is not None else executable()
        self.metadata = HdMetadata(self.program)
        self.dropped_blocks = 0

        self._process: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

        self._input: deque[bytes] = deque()
        self._input_bytes = 0
        self._input_cap = int(SAMPLE_RATE_HZ * 2 * _INPUT_QUEUE_S)
        self._input_ready = threading.Condition()

        self._audio: deque[np.ndarray] = deque()
        self._audio_frames = 0
        self._audio_cap = int(self.audio_rate * _AUDIO_QUEUE_S)
        self._audio_lock = threading.Lock()
        self._audio_total = 0
        self._last_audio = 0.0

        # Owned by the stdout thread alone, along with the odd bytes left when
        # a read lands in the middle of a frame.
        self._resampler = RationalResampler(self.audio_rate, AUDIO_RATE_HZ)
        self._partial = b""

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def playing(self) -> bool:
        """Whether audio has arrived recently enough to still be arriving.

        Not the same question as `running`. Acquisition takes about five and a
        half seconds before the first sample - that is nrsc5 finding the OFDM
        frame, not our plumbing - and a caller deciding what to do with the
        silence needs to tell that apart from a decoder that has died.
        """
        return self.running and time.monotonic() - self._last_audio < _PLAYING_TIMEOUT_S

    def start(self) -> bool:
        """Launch the decoder. False if this build cannot, with a reason set."""
        if self.running:
            return True
        if self.path is None:
            self.metadata.update(error="HD Radio decoder is not installed")
            return False
        self._stopping.clear()
        # No console window: the app is a GUI, and a black rectangle appearing
        # when a user presses HD is a bug report waiting to happen.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [
            str(self.path),
            "-r",
            "-",
            "--iq-input-format",
            "cu8",
            "-o",
            "-",
            "-t",
            "raw",
            str(self.program),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
                bufsize=0,
            )
        except OSError as error:
            self._process = None
            self.metadata.update(error=f"HD Radio decoder would not start: {error}")
            return False

        self.metadata.reset()
        self.metadata.update(running=True, program=self.program)
        self._threads = [
            threading.Thread(target=self._pump, name="hd-stdin", daemon=True),
            threading.Thread(target=self._drain, name="hd-stdout", daemon=True),
            threading.Thread(target=self._watch, name="hd-stderr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return True

    def stop(self) -> None:
        """Shut the decoder down and let go of everything it was holding.

        Closing stdin is the polite exit - nrsc5 treats the end of its IQ as
        the end of the session - but a child that has wedged must not keep the
        app waiting, so the wait is short and terminating is the fallback.
        """
        self._stopping.set()
        with self._input_ready:
            self._input.clear()
            self._input_bytes = 0
            self._input_ready.notify_all()
        process, self._process = self._process, None
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1.0)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        with self._audio_lock:
            self._audio.clear()
            self._audio_frames = 0
        self.metadata.update(running=False, synced=False, playing=False)

    # -- the DSP thread's two calls ----------------------------------------

    def feed(self, raw: bytes | np.ndarray) -> None:
        """Hand the decoder a block of `cu8` IQ. Never blocks.

        The bytes go through untouched: what the ring buffer holds is already
        the interleaved unsigned pairs nrsc5 reads, so the whole conversion is
        the absence of one.
        """
        if not self.running:
            return
        data = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        if not data:
            return
        with self._input_ready:
            self._input.append(data)
            self._input_bytes += len(data)
            # Drop the oldest rather than the newest. A decoder fed the past
            # decodes the past; one that skips forward loses sync for a moment
            # and then carries on with what is happening now.
            while self._input_bytes > self._input_cap and len(self._input) > 1:
                self._input_bytes -= len(self._input.popleft())
                self.dropped_blocks += 1
            self._input_ready.notify()

    def audio(self) -> np.ndarray:
        """Everything decoded since the last call, as `(frames, 2)` float32."""
        with self._audio_lock:
            if not self._audio:
                return np.zeros((0, 2), dtype=np.float32)
            blocks = list(self._audio)
            self._audio.clear()
            self._audio_frames = 0
        return np.concatenate(blocks, axis=0)

    def snapshot(self) -> HdState:
        return replace(
            self.metadata.state,
            running=self.running,
            playing=self.playing,
            audio_seconds=self._audio_total / self.audio_rate,
        )

    # -- the three worker threads ------------------------------------------

    def _pump(self) -> None:
        """Move queued IQ into the child's stdin, blocking here and nowhere."""
        process = self._process
        if process is None or process.stdin is None:
            return
        while not self._stopping.is_set():
            with self._input_ready:
                while not self._input and not self._stopping.is_set():
                    self._input_ready.wait(0.2)
                if self._stopping.is_set():
                    return
                block = self._input.popleft()
                self._input_bytes -= len(block)
            try:
                process.stdin.write(block)
                process.stdin.flush()
            except (OSError, ValueError):
                # The child has gone. `_watch` reports why, and there is
                # nothing useful this thread could add.
                return

    def _drain(self) -> None:
        """Read audio, convert it to the app's rate, and queue it."""
        process = self._process
        if process is None or process.stdout is None:
            return
        while not self._stopping.is_set():
            try:
                chunk = process.stdout.read(_STDOUT_CHUNK)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            data = self._partial + chunk if self._partial else chunk
            # A read can land in the middle of a frame; four bytes is one
            # stereo pair, and the odd tail waits for the next one.
            usable = len(data) - len(data) % 4
            self._partial = data[usable:]
            if usable == 0:
                continue
            frames = np.frombuffer(data[:usable], dtype="<i2").reshape(-1, 2)
            block = self._resampler.process(frames.astype(np.float32) / 32768.0)
            if block.shape[0] == 0:
                continue
            with self._audio_lock:
                self._audio.append(block)
                self._audio_frames += block.shape[0]
                self._audio_total += block.shape[0]
                while self._audio_frames > self._audio_cap and len(self._audio) > 1:
                    self._audio_frames -= self._audio.popleft().shape[0]
            self._last_audio = time.monotonic()

    def _watch(self) -> None:
        """Turn the child's stderr into state, and its death into a message."""
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for raw in process.stderr:
                if self._stopping.is_set():
                    return
                self.metadata.feed(raw.decode("utf-8", "replace"))
        except (OSError, ValueError):
            return
        if self._stopping.is_set():
            return
        code = process.poll()
        if code not in (None, 0):
            self.metadata.update(error=f"HD Radio decoder stopped (code {code})")
        self.metadata.update(running=False, synced=False)


__all__ = [
    "AUDIO_RATE_HZ",
    "MAX_PROGRAMS",
    "OCCUPIED_BANDWIDTH_HZ",
    "SAMPLE_RATE_HZ",
    "HdMetadata",
    "HdProgram",
    "HdRadio",
    "HdState",
    "available",
    "executable",
]
