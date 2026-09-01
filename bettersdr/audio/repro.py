"""Repro-Radio: unattended recording, and songs pulled out of a broadcast.

Two features that share one switch, because they are the same wish at two
scales - *keep what comes out of this frequency while I am not listening.*

**Clips.** While the squelch is open, the audio is written to an MP3. When the
squelch closes the recording stays open for a hang time, because the gap
between two overs in a conversation is silence on the same channel and a file
per sentence is not what anybody meant. Files are named
`RR-162.550MHz-0831143012-0831143145.mp3`, so a folder of them sorts by
frequency and then by time, and every one says exactly what stretch of the
afternoon it covers. On a broadcast station, where nothing ever closes
the squelch, this is simply a continuous recording cut into pieces by the
per-recording time cap.

**Songs.** On broadcast FM the station is already telling you what it is
playing - `decode/rds.py` recovers the RadioText and `decode/songtag.py`
reads a song out of it. So a song is the stretch between two changes of that
text, saved as its own file, tagged with the artist and title, and refused if
it is an advertisement. Everything else the station transmits still ends up in
the clips; only the songs are pulled out separately.

Five things about that are worth setting out, because each is a place where
the obvious implementation is wrong:

* **RadioText is only worth reading once all of it has arrived.** The field
  fills four characters at a time and every state on the way is a well-formed
  artist and title, so a segmenter fed the buffer as it stands starts a new
  song several times a second and finishes none of them. That is the whole
  reason `RdsState.text_steady` exists, and `core/engine.py` hands over an
  empty string until it is true.

* **RadioText is late.** The station's playout system updates it some seconds
  after the song actually starts. A recording that began at the text change
  would miss the intro of every song. So the audio is held in a rolling
  buffer and the boundary is placed *backwards* from the text change: at the
  last moment the sound changed between speech and music if there was one,
  and at a fixed lag if there was not.
* **The same lateness applies to the end**, which is why the song file is not
  written as the audio arrives. It is written from the buffer, deliberately
  running `WRITE_LAG_S` behind real time, so that when the boundary turns out
  to have been twenty seconds ago there is still something that can be done
  about it. Writing in real time and trimming afterwards is not an option: an
  MP3 is a stream of frames and there is no going back into one.
* **A title is not enough to prove a song.** Every advertisement break is a
  stretch of RadioText too - 96.5 MHz sends `Accident? Boohoff Law. Better
  Off With Boohoff! - 96.5 Jack FM` during one. So the audio has to agree:
  `scan/voice.py` already tells music from speech, and a segment is only
  saved if most of it measured as music *and* little of it measured as
  somebody talking. The checks are independent, which is the point - an
  advertisement has to pass all of them to get through, and it fails all of
  them.
* **The volume knob must not reach the file.** `dsp/chain.py` applies volume
  and mute at the very end of the audio path, which is right for the sound
  card and wrong for a recording somebody will play back next week. The tap
  is `AudioChain.body` - after the AGC, before the volume - so an unattended
  session recorded at a whisper is not a folder of whispered files. This is a
  deliberate difference from the Record audio button, which records what was
  heard because that is what a manual recording is for.

The whole thing is a state machine over blocks with an injectable clock, the
same shape as `scan/monitor.py`, so all of it can be tested without a radio,
a sound card or an hour of waiting.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..decode.songtag import SongTag, TextTracker
from ..scan import voice
from . import encode
from .encode import Mp3Recorder, Tags
from .library import SongLibrary, safe_stem
from .record import RecordingLimits

AUDIO_RATE = 48_000

# -- clips -------------------------------------------------------------------

# How long the squelch may stay shut before a recording is closed. The gap
# between two overs is a second or two; the same reasoning as the monitor's
# release timer, and the same failure if it is too short - the radio is left
# recording one half of an exchange per file.
DEFAULT_HANG_S = 3.0
# How long one recording may run before it is closed and the next is started.
# An hour: long enough that a normal session is one file per station, short
# enough that a night of it is a folder rather than a single object no player
# will seek inside.
DEFAULT_MAX_CLIP_S = 3600.0
# How long the whole session may run before Repro-Radio switches itself off.
# Four hours, which is the "I am going to bed" default rather than a limit -
# anything longer is a deliberate choice and the control is right there.
DEFAULT_MAX_SESSION_S = 4 * 3600.0
# How long the audio may stop arriving before the session is treated as
# having been interrupted. Anything that borrows the radio - a sweep, the
# aircraft screen, a monitor session - simply stops feeding this, so there is
# no call to make at each of those sites and no new one to remember when the
# next such feature is written. Two seconds: longer than the 340 ms a gain
# probe costs and than any block the ring can be slow with, shorter than the
# briefest sweep. The rolling buffer is the reason it matters at all - it
# finds positions by elapsed time, so a buffer spanning an interruption would
# place a boundary in audio from a different frequency.
FEED_GAP_S = 2.0

# Below this much *signal* a recording is not worth keeping - a squelch that
# opened on a noise burst rather than on somebody transmitting. Measured on
# how long the gate was open and not on how long the file is, because the file
# always contains the hang time as well: with a three second hang, a file
# length test could never discard anything. Half a second, which is shorter
# than the shortest thing anybody says and longer than a click.
MIN_SIGNAL_S = 0.5

# -- songs -------------------------------------------------------------------

# The furthest back a boundary may be placed from the RadioText change that
# announced it, and therefore how far behind real time the song writer runs.
MAX_BACKUP_S = 25.0
WRITE_LAG_S = MAX_BACKUP_S
# How long a segment has to last before a file is opened for it. Nothing is
# lost by waiting - the audio is in the buffer either way - and it means a
# station whose text fragments never writes anything at all, rather than
# creating and deleting a file every eight seconds. A song refused by the
# five-copies rule never touches the disk for the same reason.
LAZY_OPEN_S = 20.0
# How much audio is held: far enough back to reach a boundary, plus the wait
# above. 45 seconds of stereo float is 17 MB, which is the price of being able
# to start a recording in the past.
PREROLL_S = MAX_BACKUP_S + LAZY_OPEN_S
# Where a boundary goes when the sound gives no clue - a segue from one song
# straight into the next, where nothing changed between speech and music.
# This is the typical lag of a station's playout metadata.
DEFAULT_LAG_S = 6.0
# Shorter than this is a jingle, a station identification or a song joined
# halfway through, none of which belongs in a music folder.
MIN_SONG_S = 45.0
# How long the song's own title may be absent from RadioText before the song
# *may* be taken to be over. This is what makes the alternating stations
# work: a great many of them flip between their slogan and the song every few
# seconds, so "the text changed" cannot mean "the song ended" - only the
# title going away for a while can. Comfortably inside `MAX_BACKUP_S`, so the
# boundary that closes the song is still inside the buffer that can reach it,
# and it cannot usefully be any longer for that reason.
TAG_GAP_S = 20.0
# The title going away is not on its own enough to end a song, and
# `MUSIC_HANG_S` above is the other half of the rule. Measured off 96.5 MHz:
# a play of `Teenage Dirtbag` lost its own title for 32 seconds in the middle
# while the record kept playing, and closing on the gap alone cut one song
# into two files and saved both - copy 1 and copy 2 of a song played once.
# ...and the backstop, for a station that stops naming what it is playing and
# never stops playing. Without one a segue-heavy hour is a single file named
# after whatever was on when the titles dried up. A minute: long enough that
# the gap in the middle of a record does not reach it, short enough that what
# is kept is still mostly the song. The boundary then has no audio evidence
# behind it, so it goes where the file already ends - see `_stopped_at`.
TAG_LOST_S = 60.0
# Longer than this is RadioText that has stopped changing. The recording is
# closed and kept; nothing new starts until the text moves again.
MAX_SONG_S = 900.0
# How much of a segment has to have measured as music before it is saved, and
# how much of it may have measured as somebody talking. Measured off 96.5 MHz
# over a quarter of an hour, against five real songs and two real breaks:
#
#     Notorious B.I.G. - Juicy   music 0.30   voice 0.05
#     Teenage Dirtbag            music 0.53   voice 0.00
#     Teenage Dirtbag            music 0.67   voice 0.03
#     Beck - Loser               music 0.53   voice 0.00
#     Beck - Loser               music 0.46   voice 0.04
#     advertisement break        music 0.14   voice 0.18
#     news and advertisements    music 0.37   voice 0.25
#
# **Speech is the discriminator and music is not.** The music shares overlap
# outright - a rap record read as music less often than a news bulletin did,
# because `scan/voice.py` calls a drum machine `data` - so a threshold high
# enough to reject the break at 0.37 throws away half the music on the
# station. That was the first version of this and it refused four of the five
# songs above. The speech shares do not overlap at all: every song measured
# at or under 0.05 and every break at or over 0.18, so 0.10 sits between them
# with a factor of two either side.
#
# The music share stays as a floor rather than a test - below a quarter is
# not a record playing under anything - and the real first line of defence is
# neither of these: an advertisement's RadioText does not name a song.
MIN_MUSIC_SHARE = 0.25
MAX_SPEECH_SHARE = 0.1

# How often the audio is asked what it sounds like, and how much of it is
# handed over each time. 0.8 s costs 3.6 ms - see the monitor facts - so at
# one reading a second this is under half a percent of a core, and it is only
# paid while song capture is on.
VERDICT_EVERY_S = 1.0
VERDICT_WINDOW_S = 0.8
# How long the record of those readings is kept: enough to cover the backwards
# step and to measure the music share of a whole song.
VERDICT_MEMORY_S = MAX_SONG_S + MAX_BACKUP_S
# How long the music has to have been absent before it counts as having
# stopped - both for placing a boundary and for deciding a song is over.
# Real broadcast music does not read as music every single second: eight
# minutes measured off 96.5 MHz had twenty-seven one-second readings of
# `tone` or `data` scattered through one song, and a rap record read as
# `data` more often than as music. A boundary placed at the last of those
# blips is a boundary placed at random. Eight seconds is longer than the
# longest such run measured (three) and shorter than any advertisement.
MUSIC_HANG_S = 8.0


def clip_name(
    center_hz: float,
    start: datetime,
    end: datetime | None = None,
    extension: str = encode.EXTENSION,
) -> str:
    """`RR-98.500MHz-0831143012-0831143145.mp3`.

    Three decimal places on the megahertz because that is the finest raster in
    the band plan - 5 kHz - rounded to something a person reads at a glance,
    and because a fixed number of digits is what makes a folder sort properly.

    The timestamp is month, day, hour, minute and second, two digits each, in
    **local** time. The year is left out because the only thing a name has to
    do is separate one afternoon's recordings from the next and say which is
    which, and the file's own date carries the year anyway. Local rather than
    UTC because these are the one set of files a person reads the time off
    directly: a recording made at eight in the evening that calls itself the
    next day is not a name anybody can match to what they remember hearing.
    The cost is that the hour before a daylight-saving change and the hour
    after it produce the same name, and `_free_path` already numbers those.

    `end` of None is the name a recording carries while it is still running.
    It is renamed when it closes, which is possible at all only because an MP3
    has no header to go back and fix - so a file left behind by a crash is a
    playable recording with an honest name, not a broken one.
    """
    stamp = "%m%d%H%M%S"
    tail = end.astimezone().strftime(stamp) if end is not None else "recording"
    head = start.astimezone().strftime(stamp)
    return f"RR-{center_hz / 1e6:.3f}MHz-{head}-{tail}.{extension}"


@dataclass(frozen=True)
class ReproSettings:
    """What the user asked for. Persisted; see `core/settings.py`."""

    enabled: bool = False
    songs: bool = False
    hang_s: float = DEFAULT_HANG_S
    max_clip_s: float = DEFAULT_MAX_CLIP_S
    max_session_s: float = DEFAULT_MAX_SESSION_S


@dataclass(frozen=True)
class ReproStatus:
    """What Repro-Radio is doing, for a status line that tells the truth."""

    enabled: bool = False
    recording: bool = False
    clip_path: str | None = None
    clip_seconds: float = 0.0
    clips: int = 0
    session_seconds: float = 0.0
    session_remaining: float | None = None
    songs_enabled: bool = False
    song_title: str = ""
    song_seconds: float = 0.0
    songs_saved: int = 0
    songs_skipped: int = 0
    last_song: str = ""
    message: str | None = None


class _Preroll:
    """The last few seconds of audio, kept so a song can start in the past.

    Always two channels. A station that drops its pilot mid-song would
    otherwise change the shape of the buffer halfway through the one thing
    that has to come out of it as a single array, and a music file is worth
    the doubled buffer either way. At 30 seconds that is 11 MB, against the
    ring buffer's own several.

    Positions are found by counting back from the newest sample rather than
    from per-block timestamps. `audio/output.py` resamples by at most 0.5% to
    track the sound card, so across the whole buffer time and samples can
    disagree by 150 ms - a fraction of the lag being corrected for, and far
    less than the uncertainty in the correction itself.

    The buffer keeps its *own* notion of when that newest sample arrived, and
    answers questions against it rather than against the caller's clock. The
    two are the same thing right up until the audio stops arriving - which is
    exactly what a sweep or an aircraft excursion does - and a caller asking
    "the last thirty seconds" five seconds after the last block must be given
    the last thirty seconds of *audio*, not a window shifted five seconds into
    a silence that was never recorded.
    """

    def __init__(self, seconds: float = PREROLL_S, rate: int = AUDIO_RATE) -> None:
        self.rate = int(rate)
        self.capacity = int(seconds * self.rate)
        self.newest = 0.0
        self._blocks: deque[np.ndarray] = deque()
        self._frames = 0

    def push(self, audio: np.ndarray, now: float = 0.0) -> None:
        block = np.asarray(audio, dtype=np.float32)
        if block.size == 0:
            return
        if block.ndim == 1:
            block = np.repeat(block[:, None], 2, axis=1)
        elif block.shape[1] == 1:
            block = np.repeat(block, 2, axis=1)
        elif block.shape[1] > 2:
            block = block[:, :2]
        self._blocks.append(block)
        self._frames += block.shape[0]
        self.newest = now
        while self._blocks and self._frames - self._blocks[0].shape[0] >= self.capacity:
            self._frames -= self._blocks.popleft().shape[0]

    def _tail_frames(self, wanted: int) -> np.ndarray:
        wanted = min(int(wanted), self._frames)
        if wanted <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        taken: list[np.ndarray] = []
        have = 0
        for block in reversed(self._blocks):
            taken.append(block)
            have += block.shape[0]
            if have >= wanted:
                break
        return np.concatenate(list(reversed(taken)), axis=0)[-wanted:]

    def tail(self, seconds: float) -> np.ndarray:
        """The last `seconds` of audio, as one `(frames, 2)` array."""
        return self._tail_frames(int(seconds * self.rate))

    def span(self, start: float, end: float) -> np.ndarray:
        """The audio between two instants, measured against `newest`.

        Whatever of it is still held: a caller asking for something older than
        the buffer gets the oldest part it has, which is the same answer the
        ring buffer gives and for the same reason - dropping the request
        outright would put a silent gap in a file instead of a short one. A
        caller asking for something *newer* than the last block gets the audio
        up to that block, which is the whole of what exists.
        """
        if end <= start:
            return np.zeros((0, 2), dtype=np.float32)
        now = self.newest
        from_end = min(int(round((now - start) * self.rate)), self._frames)
        to_end = max(int(round((now - end) * self.rate)), 0)
        if to_end >= from_end:
            return np.zeros((0, 2), dtype=np.float32)
        return self._tail_frames(from_end)[: from_end - to_end]

    def clear(self) -> None:
        self._blocks.clear()
        self._frames = 0
        self.newest = 0.0

    @property
    def seconds(self) -> float:
        return self._frames / self.rate


class ReproRadio:
    """The whole feature: one object, fed one block at a time.

    Owns its recorders and its library, and touches no device and no Qt. The
    clock is injected so a four-hour session cap can be tested in a
    millisecond, the same bargain `scan/monitor.py` made.
    """

    def __init__(
        self,
        folder: str | Path,
        settings: ReproSettings | None = None,
        limits: RecordingLimits | None = None,
        rate: int = AUDIO_RATE,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.settings = settings or ReproSettings()
        self.limits = limits or RecordingLimits()
        self.rate = int(rate)
        self._clock = clock
        self.library = SongLibrary(self.folder / "Songs")

        self._session_started: float | None = None
        self._message: str | None = None
        self._finished = False
        self._center_hz: float | None = None
        self._last_fed = 0.0

        # Clips
        self._clip: Mp3Recorder | None = None
        self._clip_started_wall = datetime.now(UTC)
        self._clip_center_hz = 0.0
        self._clip_open_until = 0.0
        self._clip_signal_s = 0.0
        self._clips = 0

        # Songs
        self._text = TextTracker()
        self._preroll = _Preroll(PREROLL_S, self.rate)
        self._verdicts: deque[tuple[float, str]] = deque()
        self._last_verdict_at = 0.0
        self._song: Mp3Recorder | None = None
        self._song_tag: SongTag | None = None
        self._song_started = 0.0
        self._song_written_to = 0.0
        self._tag_seen_at = 0.0
        self._song_refused = False
        self._song_temp: Path | None = None
        self._song_station = ""
        self._song_genre = ""
        self._song_center_hz = 0.0
        self._songs_saved = 0
        self._songs_skipped = 0
        self._last_song = ""

    # -- clock -------------------------------------------------------------

    def _now(self) -> float:
        return float(self._clock()) if self._clock is not None else time.monotonic()

    def _wall(self) -> datetime:
        """Wall time for a filename.

        Taken from the injected clock where there is one, so a test's timeline
        and the names of the files it produces are the same timeline.
        """
        if self._clock is None:
            return datetime.now(UTC)
        return datetime.fromtimestamp(self._now(), UTC)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> ReproStatus:
        """Begin a session. Restarting one that ran out resets its clock."""
        if not encode.available():
            self._message = encode.unavailable_reason()
            self.settings = replace(self.settings, enabled=False)
            return self.status
        self._session_started = self._now()
        self._last_fed = self._session_started
        self._finished = False
        self._message = None
        self.settings = replace(self.settings, enabled=True)
        self.library.load()
        return self.status

    def stop(self, reason: str | None = None) -> ReproStatus:
        """End the session, closing and naming everything still open."""
        self._close_clip()
        self._close_song(keep=True)
        self.settings = replace(self.settings, enabled=False)
        self._session_started = None
        self._preroll.clear()
        self._verdicts.clear()
        if reason:
            self._message = reason
        return self.status

    def interrupt(self) -> None:
        """The radio moved, or something else took it. Close what was open.

        A file whose name says one frequency must not contain another, and a
        song is a property of the station that was playing it - so both are
        closed here rather than allowed to run across the break. The song is
        still kept if it had earned it, because what was recorded before the
        break really was that song.

        The rolling buffer and the run of verdicts are cleared for a stronger
        reason than tidiness: both are indexed by elapsed time, so either one
        left spanning the break would answer a question about the last minute
        with audio from a different frequency.
        """
        self._close_clip()
        self._close_song(keep=True)
        self._text.reset()
        self._preroll.clear()
        self._verdicts.clear()
        self._tag_seen_at = 0.0
        self._last_verdict_at = 0.0

    # -- the block ---------------------------------------------------------

    def feed(
        self,
        audio: np.ndarray,
        *,
        squelch_open: bool | None,
        center_hz: float,
        station: str = "",
        radio_text: str = "",
        genre: str = "",
    ) -> None:
        """One block of audio, taken *before* volume and mute are applied.

        `squelch_open` of None means no squelch is set, which is the normal
        state on a broadcast band. That is treated as permanently open: the
        user asked to record this frequency, and the honest reading of "no
        squelch" is "nothing is gating this", not "record nothing".
        """
        if not self.settings.enabled or self._session_started is None:
            return
        now = self._now()

        limit = self.settings.max_session_s
        if limit and now - self._session_started >= limit:
            self._finished = True
            self.stop("Repro-Radio reached its session limit and stopped.")
            return

        # Asked of every block rather than only while a clip is open: a song
        # can be in progress with the squelch shut, and a retune during the
        # hang would otherwise run one station's audio into the next.
        moved = self._center_hz is not None and center_hz != self._center_hz
        # A gap in the audio means something else had the radio. What was open
        # belongs to before it, and - more importantly - the rolling buffer
        # must not be left spanning the gap, because it finds positions by
        # elapsed time and would place a boundary in the wrong audio.
        interrupted = now - self._last_fed > FEED_GAP_S
        self._last_fed = now
        if moved or interrupted:
            self.interrupt()
        self._center_hz = center_hz
        if audio.size == 0:
            return

        self._service_clip(audio, squelch_open, center_hz, now)
        if self.settings.songs:
            self._preroll.push(audio, now)
            self._measure(now)
            self._service_songs(now, station, radio_text, genre, center_hz)

    # -- clips -------------------------------------------------------------

    def _service_clip(
        self,
        audio: np.ndarray,
        squelch_open: bool | None,
        center_hz: float,
        now: float,
    ) -> None:
        gate_open = True if squelch_open is None else bool(squelch_open)
        if gate_open:
            self._clip_open_until = now + max(0.0, self.settings.hang_s)

        if self._clip is None:
            if not gate_open:
                return
            self._open_clip(center_hz, audio)
            if self._clip is None:
                return
            # ...and then fall through, so the block that opened the file is
            # written into it. Returning here dropped the first block of every
            # recording, which is a fraction of a second of somebody starting
            # to speak - and, for a recording closed straight away, left a
            # file with nothing in it at all.

        if gate_open:
            self._clip_signal_s += audio.shape[0] / self.rate
        self._clip.write(audio)
        if not self._clip.active:
            # The recorder stopped itself, which now only ever means the size
            # cap or the disk - the time cap is enforced here, where the
            # difference between "roll over to the next file" and "something
            # is wrong" is known.
            self._message = self._clip.stopped_reason
            self._finish_clip()
            self.stop()
            return
        if self._clip.seconds >= self.settings.max_clip_s:
            self._finish_clip()
            if now < self._clip_open_until:
                self._open_clip(center_hz, audio)
            return
        if now >= self._clip_open_until:
            self._finish_clip()

    def _open_clip(self, center_hz: float, audio: np.ndarray) -> None:
        started = self._wall()
        channels = 2 if audio.ndim > 1 and audio.shape[1] > 1 else 1
        path = self.folder / clip_name(center_hz, started)
        recorder = Mp3Recorder(path, self.rate, self.limits, channels=channels).start()
        if not recorder.active:
            self._message = recorder.stopped_reason
            self.stop()
            return
        self._clip = recorder
        self._clip_signal_s = 0.0
        self._clip_started_wall = started
        self._clip_center_hz = center_hz

    def _close_clip(self) -> None:
        if self._clip is not None:
            self._finish_clip()

    def _finish_clip(self) -> None:
        """Close the file and give it its end time.

        A recording with too little signal in it is deleted rather than left
        as a two-second file: a folder of those is how an unattended session
        on a noisy channel becomes unreadable.
        """
        recorder, self._clip = self._clip, None
        if recorder is None:
            return
        recorder.stop()
        if self._clip_signal_s < MIN_SIGNAL_S:
            recorder.path.unlink(missing_ok=True)
            return
        final = recorder.path.with_name(
            clip_name(self._clip_center_hz, self._clip_started_wall, self._wall())
        )
        self._clips += 1
        try:
            recorder.path.replace(_free_path(final))
        except OSError:
            # The recording is on disk under its in-progress name, which is
            # still a playable file that says where and when it started. That
            # is a much better outcome than losing it over a rename.
            self._message = f"Could not rename {recorder.path.name}."

    # -- what the audio sounds like ---------------------------------------

    def _measure(self, now: float) -> None:
        if now - self._last_verdict_at < VERDICT_EVERY_S:
            return
        clip = self._preroll.tail(VERDICT_WINDOW_S)
        if clip.shape[0] < int(voice.MIN_SECONDS * self.rate):
            return
        self._last_verdict_at = now
        self._verdicts.append((now, voice.classify(clip, float(self.rate)).kind))
        while self._verdicts and now - self._verdicts[0][0] > VERDICT_MEMORY_S:
            self._verdicts.popleft()

    def _share(self, since: float, until: float, kind: str) -> float:
        """What fraction of the segment's readings came back as `kind`.

        Two of these decide whether a segment is kept: enough music, and not
        too much speech. They are not the same question and one does not
        imply the other - a station talking over a music bed reads as neither,
        so a segment can fail the music test with no speech in it at all, and
        a jingle with a spoken tag can pass it with plenty.
        """
        inside = [reading for at, reading in self._verdicts if since <= at <= until]
        if not inside:
            return 0.0
        return sum(reading == kind for reading in inside) / len(inside)

    def _music_window(self, now: float) -> list[tuple[float, str]]:
        return [at_kind for at_kind in self._verdicts if at_kind[0] >= now - MAX_BACKUP_S]

    def _music_stopped(self, now: float) -> float | None:
        """The last moment music was still playing, within the backup window."""
        for at, kind in reversed(self._music_window(now)):
            if kind == voice.MUSIC:
                return at
        return None

    def _music_started(self, now: float) -> float | None:
        """When the music now playing began, if the start of it was heard.

        Walked back through the music readings alone, stopping at the first
        gap longer than `MUSIC_HANG_S`. None where there is no such gap in
        the window - a segue from one song into the next - because then there
        is no audible boundary to find and the caller must fall back on the
        station's own timing rather than on the oldest reading it happens to
        still hold.
        """
        heard = [at for at, kind in self._music_window(now) if kind == voice.MUSIC]
        if not heard:
            return None
        start = heard[-1]
        for at in reversed(heard[:-1]):
            if start - at > MUSIC_HANG_S:
                return start
            start = at
        return None

    def _music_playing(self, now: float) -> bool:
        """Whether music has been heard recently enough to still be on."""
        heard = self._music_stopped(now)
        return heard is not None and now - heard <= MUSIC_HANG_S

    def _announced_at(self, now: float) -> float:
        """Where a boundary goes when a *new song was announced*.

        The station said so late, so the audio is asked first: where the music
        now playing started is a far better answer than any fixed lag. Where
        it has been playing all along - one song straight into the next - the
        typical lag of a playout system is all there is to go on.
        """
        heard = self._music_started(now)
        return heard if heard is not None else now - DEFAULT_LAG_S

    def _stopped_at(self, now: float) -> float:
        """Where a boundary goes when *the title went away*.

        Not `now`: the title has been gone for `TAG_GAP_S` by the time this is
        asked, and that whole stretch belongs to whatever came next. Two
        things bound it and the earlier of them wins. The music stopping is
        one - an advertisement break begins before the text catches up. The
        last moment the station still had the title up is the other, and it is
        what answers a segue, where the music never stopped at all and taking
        the audio's word for it would run the file into the next song.
        """
        heard = self._music_stopped(now)
        ended = min(self._tag_seen_at, heard if heard is not None else self._tag_seen_at)
        # ...and never further back than the file already reaches. The writer
        # runs `WRITE_LAG_S` behind and cannot be rewound - an MP3 is a stream
        # of frames - so a boundary older than that is a boundary that cannot
        # be honoured, and saying so here is better than quietly appending a
        # minute of whatever came next.
        return max(ended, now - MAX_BACKUP_S)

    # -- songs -------------------------------------------------------------

    def _service_songs(
        self,
        now: float,
        station: str,
        radio_text: str,
        genre: str,
        center_hz: float,
    ) -> None:
        """One block of the song segmenter.

        A song is identified by its *tag*, not by a change of RadioText, and
        that distinction is the whole of this method. Plenty of stations
        alternate their slogan with the song every few seconds; treating each
        of those as a boundary produces nothing but eight-second fragments,
        every one of them below `MIN_SONG_S`, so the feature would appear to
        run and never save anything - the failure nobody would report.
        """
        if self._song is not None:
            self._write_song(now - WRITE_LAG_S)
            if not self._song.active:
                self._message = self._song.stopped_reason
                self._close_song(keep=True, ended_at=now)
            elif now - self._song_started >= MAX_SONG_S:
                # Kept, but the tag is remembered: RadioText that has stopped
                # moving must not start the same recording over again every
                # quarter of an hour.
                self._close_song(keep=True, ended_at=now, forget_tag=False)

        self._text.update(radio_text, now)
        # The frequency goes in as well as the name, because a station's own
        # field is very often just its dial position - `96.5 Jack FM` - and
        # matching it against where the receiver is pointed recognises that
        # on the first message rather than after the third song.
        tag = (
            self._text.tag(self._text.text, station, center_hz)
            if self._text.text
            else None
        )
        current = self._song_tag

        if tag is not None and current is not None and tag.key == current.key:
            self._tag_seen_at = now
            self._ensure_song_file(now)
            return
        if tag is not None:
            # A different song was announced. One instant ends the old segment
            # and begins the new one, so a segue is cut once rather than
            # twice, and neither side of it is counted in both files.
            boundary = self._announced_at(now)
            self._close_song(keep=True, ended_at=boundary)
            self._open_song(tag, boundary, now, station, genre, center_hz)
            return
        gone = now - self._tag_seen_at
        if current is not None and gone >= TAG_GAP_S and (
            not self._music_playing(now) or gone >= TAG_LOST_S
        ):
            # The title has gone away, nothing has replaced it and the music
            # has stopped, which together are what the start of an
            # advertisement break looks like. The music is the half that
            # matters: this station drops its own title for half a minute in
            # the middle of a record, and closing on the gap alone made two
            # files out of one play of it.
            self._close_song(keep=True, ended_at=self._stopped_at(now))
            return
        if current is not None:
            # Still the same song; the text on the air at this instant is
            # simply the station's own, which is most of the time on a
            # station that alternates the two.
            self._ensure_song_file(now)

    def _write_song(self, until: float) -> None:
        """Feed the song recorder up to `until`, from the rolling buffer."""
        if self._song is None or until <= self._song_written_to:
            return
        block = self._preroll.span(self._song_written_to, until)
        self._song_written_to = until
        if block.size:
            self._song.write(block)

    def _open_song(
        self,
        tag: SongTag,
        boundary: float,
        now: float,
        station: str,
        genre: str,
        center_hz: float,
    ) -> None:
        # Claimed before anything else can fail. A song that is refused is
        # still the song currently on the air, and forgetting that would have
        # this method run again on the very next block - counting another
        # refusal, and another, at the block rate.
        self._song_tag = tag
        self._song_started = boundary
        self._song_written_to = boundary
        self._tag_seen_at = now
        self._song_station = station
        self._song_genre = genre
        self._song_center_hz = center_hz

        placement = self.library.plan(tag)
        self._song_refused = not placement.wanted
        if self._song_refused:
            # Refused before a byte is encoded. The five-copies rule is about
            # not filling a disk, so noticing at the end would be most of the
            # cost for none of the benefit.
            self._songs_skipped += 1
            self._message = placement.reason

    def _ensure_song_file(self, now: float) -> None:
        """Open the file, once the segment has lasted long enough to be one.

        The wait costs nothing: the audio back to the boundary is still in the
        buffer, and it is written the moment the file exists. What it buys is
        that a station alternating two texts that both parse as songs - which
        produces a string of eight-second segments until the tracker works out
        which is the slogan - writes no files at all instead of a file and a
        deletion every eight seconds.
        """
        tag = self._song_tag
        if (
            self._song is not None
            or tag is None
            or self._song_refused
            or now - self._song_started < LAZY_OPEN_S
        ):
            return
        temp = (
            self.library.folder
            / f".{safe_stem(tag.display)}.recording.{encode.EXTENSION}"
        )
        # Always stereo: a music file is what this is for, and a station
        # dropping its pilot for a few seconds must not be able to change what
        # the file is halfway through.
        recorder = Mp3Recorder(temp, self.rate, self.limits, channels=2).start()
        if not recorder.active:
            self._message = recorder.stopped_reason
            self._song_refused = True
            return
        self._song = recorder
        self._song_temp = temp

    def _close_song(
        self, keep: bool, ended_at: float | None = None, forget_tag: bool = True
    ) -> None:
        """End the open segment. Safe when there is nothing open.

        `forget_tag` false leaves the song claimed while releasing the
        recorder, which is how a segment that ran past `MAX_SONG_S` stops
        without immediately starting itself again on the next block. Keeping
        the claim is not enough on its own: the lazy opener would find a
        claimed tag with no file and open one, which is the same recording
        starting over every quarter of an hour. So the claim is marked
        refused as well, and only a change of title clears it.
        """
        recorder, self._song = self._song, None
        temp, self._song_temp = self._song_temp, None
        tag = self._song_tag
        if forget_tag:
            self._song_tag = None
            self._song_refused = False
        else:
            self._song_refused = True
        if recorder is None or tag is None or temp is None:
            return
        ended = ended_at if ended_at is not None else self._now()
        # The writer runs behind real time on purpose; this is where it
        # catches up, to the boundary rather than to the moment the boundary
        # was noticed. Restored to `self._song` for the length of the call
        # because that is what `_write_song` writes to.
        self._song = recorder
        self._write_song(ended)
        self._song = None
        recorder.stop()

        seconds = recorder.seconds
        music = self._share(self._song_started, ended, voice.MUSIC)
        speech = self._share(self._song_started, ended, voice.VOICE)
        sounded_wrong = music < MIN_MUSIC_SHARE or speech > MAX_SPEECH_SHARE
        if not keep or seconds < MIN_SONG_S or sounded_wrong:
            temp.unlink(missing_ok=True)
            self._songs_skipped += 1
            if seconds >= MIN_SONG_S and sounded_wrong:
                self._message = (
                    f"{tag.display} did not sound like music "
                    f"({music * 100:.0f}% of it did, {speech * 100:.0f}% was "
                    "talking), so it was not kept."
                )
            return

        placement = self.library.plan(tag)
        if placement.path is None:
            temp.unlink(missing_ok=True)
            self._songs_skipped += 1
            self._message = placement.reason
            return
        final = _free_path(placement.path)
        try:
            temp.replace(final)
        except OSError:
            temp.unlink(missing_ok=True)
            self._message = f"Could not save {tag.display}."
            return
        encode.write_tags(
            final,
            Tags(
                title=tag.title,
                artist=tag.artist,
                album=self._album(),
                genre=self._song_genre,
                date=self._wall().strftime("%Y-%m-%dT%H:%M"),
                comment=(
                    f"Recorded from {self._song_center_hz / 1e6:.3f} MHz by "
                    "BetterSDR, named from RDS RadioText."
                ),
            ),
        )
        self.library.remember(tag, final, when=self._wall().isoformat())
        self._songs_saved += 1
        self._last_song = tag.display
        self._message = f"Saved {tag.display} - {placement.reason}"

    def _album(self) -> str:
        """Where it came from, which is the only album a broadcast has."""
        station = self._song_station.strip()
        dial = f"{self._song_center_hz / 1e6:.1f} MHz"
        return f"{station} {dial}" if station else f"Off air, {dial}"

    # -- reporting ---------------------------------------------------------

    @property
    def status(self) -> ReproStatus:
        now = self._now()
        elapsed = now - self._session_started if self._session_started else 0.0
        limit = self.settings.max_session_s
        return ReproStatus(
            enabled=self.settings.enabled,
            recording=self._clip is not None,
            clip_path=None if self._clip is None else str(self._clip.path),
            clip_seconds=0.0 if self._clip is None else self._clip.seconds,
            clips=self._clips,
            session_seconds=elapsed,
            session_remaining=max(0.0, limit - elapsed) if limit else None,
            songs_enabled=self.settings.songs,
            song_title="" if self._song_tag is None else self._song_tag.display,
            # Elapsed rather than what the encoder has swallowed: the writer
            # runs `WRITE_LAG_S` behind, and a screen counting that up from
            # zero would look stuck for the first half minute of every song.
            song_seconds=(
                0.0 if self._song is None else max(0.0, now - self._song_started)
            ),
            songs_saved=self._songs_saved,
            songs_skipped=self._songs_skipped,
            last_song=self._last_song,
            message=self._message,
        )

    @property
    def finished(self) -> bool:
        """Whether the session ended by reaching its own limit."""
        return self._finished


def _free_path(path: Path) -> Path:
    """`path`, or the first numbered variant of it that does not exist.

    Two recordings can legitimately want one name - a station whose clock puts
    two clips in the same second - and overwriting one of them silently is how
    an unattended session loses a recording nobody knows was made.
    """
    if not path.exists():
        return path
    for number in range(2, 100):
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        if not candidate.exists():
            return candidate
    return path


__all__ = [
    "MAX_SPEECH_SHARE",
    "MIN_MUSIC_SHARE",
    "DEFAULT_HANG_S",
    "DEFAULT_MAX_CLIP_S",
    "DEFAULT_MAX_SESSION_S",
    "FEED_GAP_S",
    "LAZY_OPEN_S",
    "MIN_SIGNAL_S",
    "MIN_SONG_S",
    "PREROLL_S",
    "MUSIC_HANG_S",
    "TAG_GAP_S",
    "TAG_LOST_S",
    "ReproRadio",
    "ReproSettings",
    "ReproStatus",
    "clip_name",
]
