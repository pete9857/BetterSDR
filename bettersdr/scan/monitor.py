"""Watching a band instead of photographing it once.

A sweep answers "what is transmitting right now", which is the right question
for broadcast bands where everything is on all the time. It is close to
useless for the bands people actually want a scanner for. A fire crew's
channel is silent for fifty-nine minutes an hour; a marine channel carries
four seconds of somebody calling a bridge; a business radio licence is a
handful of transmissions across a whole afternoon. Sweep once and the list is
whatever happened to be keyed in that half second, which is usually nothing,
and the honest report - "nothing found" - is wrong about the band.

So: sweep the same range over and over, and keep a ledger. What matters is no
longer whether a channel is transmitting, but **how often** it has been, how
strong it was, and when it was last heard. That is what this module holds.

It also decides where the radio goes next, and that is the second half of the
job. Between passes the monitor picks a busy channel, parks on it for a
moment and demodulates it, so `scan/voice.py` can say whether that was
somebody talking, a pager, a tone or an open squelch - none of which a power
spectrum can tell apart. When it turns out to be a voice, the monitor stays
there and the audio plays, the way every scanner made since 1975 does, and
goes back to sweeping when the channel goes quiet.

Pure logic, no Qt and no device: the engine tunes and demodulates, and hands
the results back here. The state machine is testable against a clock that
does not tick unless the test says so, which is the only way the release
timer and the revisit policy can be checked at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .classifier import UNMODULATED, Signal
from .detector import PERSISTENCE_TOLERANCE_HZ
from .voice import Verdict

# What the radio is doing. Three states and no more: the whole design is that
# sweeping is the resting state and everything else is a short excursion from
# it that always returns.
SWEEPING = "sweeping"
AUDITIONING = "auditioning"
HOLDING = "holding"

# How long to sit on a channel and listen before deciding what it is. Under
# `voice.MIN_SECONDS` the answer is refused outright, and under
# `voice.TRUSTED_SECONDS` it is never presented as certain - so this is set
# comfortably above both. It is also, directly, how long the sweep pauses.
DEFAULT_AUDITION_S = 0.8

# How long a held channel may be uninteresting before the sweep resumes. Long
# enough to ride out the gap between two overs of a conversation, which is the
# whole reason a scanner holds rather than simply following the strongest
# signal.
DEFAULT_RELEASE_S = 3.0

# The soonest a channel is worth listening to again. Without it the strongest
# channel in the band is auditioned every single cycle and nothing else ever
# is - which is a scanner locked onto one frequency, wearing the disguise of
# one that is scanning.
DEFAULT_REVISIT_S = 8.0
# ...except where somebody was talking last time, which is worth coming back
# to sooner: a conversation is a sequence of short transmissions with gaps in
# it, and the gaps are longer than this.
VOICE_REVISIT_S = 3.0

# How many passes a channel has to appear in before it is worth a line on
# screen or a moment of the radio's time. The same argument as the sweeper's
# persistence gate, and the same number: a noise peak clears a threshold once,
# a transmitter does it again and again.
DEFAULT_MIN_SIGHTINGS = 2

# How far apart two readings must be before they are two channels. Taken from
# the detector rather than chosen here, because it is the same question the
# persistence gate asks between passes.
MATCH_TOLERANCE_HZ = PERSISTENCE_TOLERANCE_HZ


def _key(frequency_hz: float) -> int:
    """Which channel a reading belongs to. One key, one line on screen."""
    return int(round(frequency_hz / MATCH_TOLERANCE_HZ))


def _rank(signal: Signal, contradicted: bool = False) -> tuple[float, ...]:
    """Which of two views of one channel to keep. The sweeper's rule, plus one.

    Confident first, then widest, then loudest. A single dwell measures a
    signal's *instantaneous* width, so the widest view of a channel across
    many passes is the truest one - and a station that widens when somebody
    talks is thereby told apart from a spur that never does.

    The addition is `contradicted`, and it exists because this screen has
    evidence the sweeper never had. A 50 ms dwell that lands between two words
    sees a bare carrier and says so, which is a fair reading of what it
    measured; having then *listened* to the channel and heard somebody
    talking, it is no longer a fair reading of the channel. So a bare-carrier
    label loses to anything at all once listening has contradicted it, rather
    than sitting at the top of the card over a badge saying Voice.
    """
    demoted = contradicted and signal.label == UNMODULATED
    return (0.0 if demoted else 1.0, signal.confidence, signal.bandwidth_hz,
            signal.snr_db)


@dataclass
class _Channel:
    """The ledger's row. Mutable; `Activity` is the frozen view of it."""

    signal: Signal
    first_heard: float
    last_heard: float
    sightings: int = 1
    first_pass: int = 0
    snr_db: float = 0.0
    peak_snr_db: float = 0.0
    active: bool = False
    verdict: Verdict | None = None
    auditions: int = 0
    voice_heard: int = 0
    last_audited: float = 0.0
    skipped: bool = False
    held: bool = False
    # Set once listening has found something on a channel the sweep read as a
    # bare carrier. See `_rank`.
    contradicted: bool = False


@dataclass(frozen=True)
class Activity:
    """One channel, and everything the monitor has learned about it.

    The `Signal` is the same object the discovery list already knows how to
    draw - what it is, how to hear it, and why the classifier thinks so. What
    is added here is the part a single sweep cannot answer: how often, how
    recently, and what it sounded like.
    """

    signal: Signal
    sightings: int
    passes: int
    first_heard: float
    last_heard: float
    snr_db: float
    peak_snr_db: float
    active: bool
    verdict: Verdict | None
    auditions: int
    voice_heard: int
    skipped: bool
    held: bool
    now: float
    contradicted: bool = False

    @property
    def frequency_hz(self) -> float:
        return self.signal.frequency_hz

    @property
    def label(self) -> str:
        """What to call this channel, given everything now known about it.

        Almost always the classifier's answer. The exception is the one case
        where listening has directly disproved it: a channel the sweep caught
        between two words reads as a bare carrier, and once somebody has been
        heard talking on it, saying "Unmodulated carrier" over a badge reading
        Voice is the app disagreeing with itself on one line. The band plan's
        name for the allocation is the honest fallback until a later sweep
        catches the channel mid-transmission and names it properly.
        """
        if self.contradicted and self.signal.label == UNMODULATED:
            return self.signal.band_name or "Active channel"
        return self.signal.label

    @property
    def icon(self) -> str:
        """The glyph for this channel. Beside `label` so the two agree.

        `ui/results.summarise` counts the chips off whatever it is handed, and
        it has to be handed *these* rather than the signals underneath: a
        channel whose label listening has corrected would otherwise be counted
        under one name on the chip and hidden under another.
        """
        return self.signal.icon

    @property
    def duty(self) -> float:
        """Share of the passes since it was first heard that it was up in.

        Measured from when the channel was *first heard*, not from the start
        of the session. A channel found in the twentieth pass and heard in
        every pass since is busy, and dividing by twenty would report it as
        idle - which would then take an hour to correct itself.
        """
        return self.sightings / max(1, self.passes)

    @property
    def silent_for(self) -> float:
        """Seconds since it was last heard transmitting."""
        return max(0.0, self.now - self.last_heard)

    @property
    def heard_voice(self) -> bool:
        """Whether somebody has been heard talking on this channel."""
        return self.voice_heard > 0

    @property
    def sound(self) -> str:
        """What it sounded like, in one word, or nothing if never listened to."""
        return "" if self.verdict is None else self.verdict.label

    @property
    def activity_phrase(self) -> str:
        """How busy this channel is, said the way a person would say it."""
        if self.sightings <= 1:
            return "heard once"
        return f"heard {self.sightings} times, up {self.duty * 100:.0f}% of the time"


@dataclass(frozen=True)
class MonitorState:
    """What the monitor is doing and what it has found. The whole screen."""

    phase: str
    target_hz: float | None
    passes: int
    elapsed_s: float
    channels: tuple[Activity, ...]
    band_name: str = ""

    @property
    def holding(self) -> bool:
        return self.phase == HOLDING

    @property
    def listening(self) -> bool:
        """Whether audio from a channel is playing right now."""
        return self.phase == HOLDING

    @property
    def voices(self) -> int:
        return sum(1 for channel in self.channels if channel.heard_voice)

    @property
    def busy(self) -> int:
        return sum(1 for channel in self.channels if channel.active)


class Monitor:
    """The ledger, and the decision about where the radio goes next.

    Fed one completed sweep pass at a time, plus the verdict from each
    audition. Everything it decides is a pure function of what it has been
    told and what time it is, so the whole of the scanner's behaviour - when
    it stops, how long it stays, what it goes back to - can be driven from a
    test with a clock that only moves when the test says so.
    """

    def __init__(
        self,
        low_hz: float,
        high_hz: float,
        band_name: str = "",
        audition_s: float = DEFAULT_AUDITION_S,
        release_s: float = DEFAULT_RELEASE_S,
        revisit_s: float = DEFAULT_REVISIT_S,
        min_sightings: int = DEFAULT_MIN_SIGHTINGS,
        listen: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.band_name = band_name
        self.audition_s = float(audition_s)
        self.release_s = float(release_s)
        self.revisit_s = float(revisit_s)
        self.min_sightings = max(1, int(min_sightings))
        # Whether to stop and play a channel that turns out to carry a voice.
        # Off makes this a pure activity survey, which is what somebody
        # mapping a band wants and what a test asserting the sweep never
        # stalls needs.
        self.listen = bool(listen)
        self._clock = clock

        self.phase = SWEEPING
        self.target_hz: float | None = None
        self.passes = 0
        self._started = clock()
        self._channels: dict[int, _Channel] = {}
        # When the currently held channel last had something on it. The
        # release timer runs from here rather than from `last_heard`, which
        # the sweep writes and the sweep is not running during a hold.
        self._interesting = 0.0

    # -- the ledger --------------------------------------------------------

    def note_pass(self, signals: Iterable[Signal]) -> None:
        """Record one completed sweep of the range.

        Everything found is folded into the ledger; everything not found this
        time is marked as no longer up, which is what makes "active" mean
        *now* rather than "at some point this session".
        """
        now = self._clock()
        self.passes += 1
        for channel in self._channels.values():
            channel.active = False
        for signal in signals:
            key = _key(signal.frequency_hz)
            channel = self._channels.get(key)
            if channel is None:
                self._channels[key] = _Channel(
                    signal=signal,
                    first_heard=now,
                    last_heard=now,
                    first_pass=self.passes - 1,
                    snr_db=signal.snr_db,
                    peak_snr_db=signal.snr_db,
                    active=True,
                )
                continue
            channel.sightings += 1
            channel.last_heard = now
            channel.active = True
            channel.snr_db = signal.snr_db
            channel.peak_snr_db = max(channel.peak_snr_db, signal.snr_db)
            # Keyed on the channel alone and resolved by rank, never replaced
            # outright: one transmitter classifies differently from pass to
            # pass, and the confident reading is the one to keep.
            if _rank(signal, channel.contradicted) > _rank(
                channel.signal, channel.contradicted
            ):
                channel.signal = signal

    # -- where to go next --------------------------------------------------

    def choose_target(self) -> float | None:
        """The channel worth listening to next, or None to carry on sweeping.

        Two rules and an order. Nothing is auditioned until it has been heard
        enough times to be believed - the same gate the discovery list uses,
        for the same reason. And nothing is auditioned twice in quick
        succession, because without that the strongest channel in the band is
        the only one the radio ever visits: a scanner locked to one frequency,
        wearing the disguise of one that is scanning.

        Among what is left, a channel never listened to comes first - it is
        the only one about which the app has nothing at all to say - and after
        that, whatever has been waiting longest. A channel that carried a
        voice last time is treated as having waited longer than it has, since
        a conversation is short transmissions with gaps between them and the
        gaps are what this has to see through.
        """
        if not self.listen:
            return None
        now = self._clock()
        best: tuple[tuple[int, float], float] | None = None
        for channel in self._channels.values():
            if channel.skipped or not channel.active:
                continue
            if channel.sightings < self.min_sightings:
                continue
            if channel.auditions == 0:
                score = (0, -channel.snr_db)
            else:
                waited = now - channel.last_audited
                due = VOICE_REVISIT_S if channel.voice_heard else self.revisit_s
                if waited < due:
                    continue
                score = (1, -waited)
            if best is None or score < best[0]:
                best = (score, channel.signal.frequency_hz)
        return None if best is None else best[1]

    def begin_audition(self, frequency_hz: float) -> None:
        """The radio is on its way to `frequency_hz` to listen to it."""
        self.phase = AUDITIONING
        self.target_hz = float(frequency_hz)

    def note_audition(self, frequency_hz: float, verdict: Verdict) -> bool:
        """Record what a channel sounded like. True means stay and play it.

        Staying is for a voice or for music, and for nothing else. Data is
        worth knowing about and not worth listening to; static and a bare tone
        are neither. The distinction is `Verdict.carries_audio`, which is one
        place rather than a condition repeated here and in the engine.
        """
        now = self._clock()
        channel = self._channels.get(_key(frequency_hz))
        if channel is not None:
            channel.auditions += 1
            channel.last_audited = now
            channel.verdict = verdict
            if verdict.is_voice:
                channel.voice_heard += 1
            if verdict.carries_audio:
                # Listening has disproved a bare-carrier reading of this
                # channel; see `_rank` and `Activity.label`.
                channel.contradicted = True
            if verdict.carries_audio or verdict.kind not in ("silence", "noise"):
                # Something was actually transmitting, whatever it turned out
                # to be. Static means the sweep's detection has already
                # stopped being true, which is not a sighting.
                channel.last_heard = now
        stay = bool(
            self.listen
            and verdict.carries_audio
            and channel is not None
            and not channel.skipped
        )
        if stay:
            self.phase = HOLDING
            self.target_hz = float(frequency_hz)
            self._interesting = now
        else:
            self.resume()
        return stay

    def note_hold(self, frequency_hz: float, verdict: Verdict) -> bool:
        """One more window of a held channel. True means keep holding.

        A channel is not released the moment it stops being a voice. A
        conversation is a sequence of short transmissions and the gaps between
        them are silence on the same frequency, so releasing on the first
        quiet window would send the radio away in the middle of the exchange
        and bring it back too late for the reply. `release_s` is how long the
        quiet has to last.

        A channel the user has explicitly held is never released here. That is
        the one thing on this screen that overrides the state machine, and it
        has to, because the user can hear something the classifier cannot.
        """
        now = self._clock()
        channel = self._channels.get(_key(frequency_hz))
        if channel is not None:
            channel.verdict = verdict
            channel.last_audited = now
            if verdict.is_voice:
                channel.voice_heard += 1
            if verdict.carries_audio:
                channel.last_heard = now
                channel.contradicted = True
        if channel is not None and channel.held:
            self._interesting = now
            return True
        if channel is not None and channel.skipped:
            self.resume()
            return False
        if verdict.carries_audio:
            self._interesting = now
            return True
        if now - self._interesting < self.release_s:
            return True
        self.resume()
        return False

    def resume(self) -> None:
        """Back to sweeping. Everything that borrows the radio ends here."""
        self.phase = SWEEPING
        self.target_hz = None

    # -- what the user can override ---------------------------------------

    def skip(self, frequency_hz: float) -> None:
        """Leave this channel now, and stop offering it.

        Aimed at the channel that is technically a voice and is not what
        anybody wanted - a repeater's identifier, a hold tone with speech in
        it, the neighbour's baby monitor. It stays on the list with everything
        known about it; the radio simply stops going there.
        """
        channel = self._channels.get(_key(frequency_hz))
        if channel is not None:
            channel.skipped = True
            channel.held = False
        if self.target_hz is not None and _key(self.target_hz) == _key(frequency_hz):
            self.resume()

    def unskip(self, frequency_hz: float) -> None:
        channel = self._channels.get(_key(frequency_hz))
        if channel is not None:
            channel.skipped = False

    def hold(self, frequency_hz: float) -> None:
        """Stay on this channel until told otherwise.

        Exclusive: holding a second channel releases the first, because two
        held channels is a request the radio cannot honour and silently
        honouring one of them is worse than not offering it.
        """
        for channel in self._channels.values():
            channel.held = False
        channel = self._channels.get(_key(frequency_hz))
        if channel is not None:
            channel.held = True
            channel.skipped = False

    def release_hold(self) -> None:
        for channel in self._channels.values():
            channel.held = False

    def signal_at(self, frequency_hz: float) -> Signal | None:
        """The classified signal on a channel, so a caller can demodulate it.

        The mode and bandwidth an audition uses are the ones the classifier
        already chose, which are the ones the Listen button would use. Picking
        differently here would mean the app judged a channel through a
        receiver it never offers anybody.
        """
        channel = self._channels.get(_key(frequency_hz))
        return None if channel is None else channel.signal

    @property
    def held_hz(self) -> float | None:
        for channel in self._channels.values():
            if channel.held:
                return channel.signal.frequency_hz
        return None

    # -- the answer --------------------------------------------------------

    def snapshot(self) -> MonitorState:
        """Everything worth showing, in frequency order.

        Filtered by the same persistence gate that governs auditions: a
        channel seen once is not yet a channel. The order is frequency because
        it is the only one that holds still - the caller sorts it into
        whatever the user asked for, exactly as it does for a sweep.
        """
        now = self._clock()
        channels = tuple(
            Activity(
                signal=channel.signal,
                sightings=channel.sightings,
                passes=max(1, self.passes - channel.first_pass),
                first_heard=channel.first_heard,
                last_heard=channel.last_heard,
                snr_db=channel.snr_db,
                peak_snr_db=channel.peak_snr_db,
                active=channel.active,
                verdict=channel.verdict,
                auditions=channel.auditions,
                voice_heard=channel.voice_heard,
                skipped=channel.skipped,
                held=channel.held,
                now=now,
                contradicted=channel.contradicted,
            )
            for channel in sorted(
                self._channels.values(), key=lambda c: c.signal.frequency_hz
            )
            if channel.sightings >= self.min_sightings
        )
        return MonitorState(
            phase=self.phase,
            target_hz=self.target_hz,
            passes=self.passes,
            elapsed_s=now - self._started,
            channels=channels,
            band_name=self.band_name,
        )


# -- ordering the list -------------------------------------------------------
# The discovery screen's own orders work on a `Signal` and still apply here.
# These two are the ones only a monitor can answer, and they are what somebody
# watching a band actually wants at the top.

ACTIVITY_SORTS: tuple[tuple[str, str], ...] = (
    ("Busiest first", "activity"),
    ("Heard most recently", "recent"),
)


def sort_activities(
    channels: Sequence[Activity], order: str = "activity"
) -> tuple[Activity, ...]:
    """Order the monitor's list. Frequency is always the final tiebreak.

    Fully determined, for the same reason the sweep's orders are: two equally
    busy channels that could swap places between updates would slam a card's
    expander shut under the cursor of somebody reading it.
    """
    items = list(channels)
    if order == "recent":
        return tuple(
            sorted(items, key=lambda c: (-c.last_heard, c.frequency_hz))
        )
    if order == "frequency":
        return tuple(sorted(items, key=lambda c: c.frequency_hz))
    if order == "strength":
        return tuple(sorted(items, key=lambda c: (-c.snr_db, c.frequency_hz)))
    if order == "kind":
        best: dict[str, float] = {}
        for channel in items:
            if channel.snr_db > best.get(channel.label, float("-inf")):
                best[channel.label] = channel.snr_db
        return tuple(
            sorted(items, key=lambda c: (-best[c.label], c.label, c.frequency_hz))
        )
    # Busiest first: how often, then how much of the time, then how loud.
    return tuple(
        sorted(
            items,
            key=lambda c: (-c.sightings, -c.duty, -c.snr_db, c.frequency_hz),
        )
    )


def with_voice(channels: Iterable[Activity]) -> tuple[Activity, ...]:
    """Only the channels somebody has been heard talking on."""
    return tuple(channel for channel in channels if channel.heard_voice)


__all__ = [
    "ACTIVITY_SORTS",
    "AUDITIONING",
    "DEFAULT_AUDITION_S",
    "DEFAULT_MIN_SIGHTINGS",
    "DEFAULT_RELEASE_S",
    "DEFAULT_REVISIT_S",
    "HOLDING",
    "SWEEPING",
    "Activity",
    "Monitor",
    "MonitorState",
    "sort_activities",
    "with_voice",
]
