"""What has actually been listened to: recently played, and this session's trail.

Bookmarks are what the user chose to keep. This is the other half - what they
did, recorded without being asked - and it exists because the landing screen
should be able to say "here is where you were last night" before it has swept
anything at all. A scan takes five seconds; a list that is already there takes
none.

Pure logic with no Qt in it, the same rule as `bookmarks.py` and for the same
reason: the Discover strip, the listening screen's recall list and anything
later that wants to know what has been played all have to agree about what a
visit is, and the way to guarantee that is for there to be one implementation.

Three decisions carry this module.

**Tuning across a band is not listening to it.** The digit tuner emits a
frequency per keystroke and click-to-tune emits one per click, so a history
that recorded every frequency the radio visited would be a list of the journey
rather than of the destinations. A visit becomes an entry only once it has
lasted `DWELL_SECONDS`, which is the same argument as the scanner's persistence
gate: seen once is not seen.

**That threshold has a floor, and the floor is how long a station takes to say
its name.** RDS needs a few seconds to confirm a name and HD Radio needs five
and a half to produce its first audio, so a dwell shorter than that would
promote every entry before anything could name it and the list would be bare
frequencies. Ten seconds is comfortably past both, which is why a name arrives
through `name()` afterwards rather than as an argument to `tune()`.

**Time is accrued, not inferred from the clock.** `update()` is called from a
view's own timer, and a view stops when its page is not showing - so a station
left playing behind the Discover screen accrues nothing rather than accruing
the whole time the user spent scanning. That is the honest reading of
"played": it counts the time somebody was actually listening.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .bookmarks import MATCH_TOLERANCE_HZ, Bookmark, format_hz
from .settings import _open_temporary, config_dir

# How long the radio has to stay somewhere before it counts as having been
# listened to. Longer than the few seconds RDS and HD Radio take to name a
# station, so an entry is named by the time it is created; short enough that
# stopping on something to hear what it is still records it.
DWELL_SECONDS = 10.0

# The recent list is a convenience, not an archive. Sixty is more than fits on
# any screen and small enough that the file stays trivial to read and write.
MAX_ENTRIES = 60

# A gap this long between two `update()` calls is treated as the view having
# been away rather than as listening time. Without it a window left minimised
# for an hour would come back and claim the hour.
MAX_TICK_SECONDS = 2.0


@dataclass
class Station:
    """One place the radio has been, and how much time was spent there."""

    frequency_hz: int
    mode: str = "wfm"
    bandwidth_hz: float = 200_000.0
    name: str = ""
    group: str = ""
    plays: int = 0
    seconds: float = 0.0
    first_heard: float = 0.0
    last_heard: float = 0.0

    @property
    def label(self) -> str:
        """`KUOW - 94.9 MHz`, the way it should read on a chip."""
        shown = format_hz(self.frequency_hz)
        return f"{self.name} - {shown}" if self.name else shown

    def matches(
        self, frequency_hz: float, tolerance_hz: float = MATCH_TOLERANCE_HZ
    ) -> bool:
        return abs(self.frequency_hz - frequency_hz) <= tolerance_hz

    def as_bookmark(self, group: str = "Recently played") -> Bookmark:
        """The bridge to the saved list: keep this one.

        The mode and bandwidth go with it for the same reason they do on a
        scan result - recalling a marine channel and getting wideband FM
        because that was the last thing playing is exactly the "the app is
        broken" moment a bookmark carrying them avoids.
        """
        return Bookmark(
            name=self.name,
            frequency_hz=self.frequency_hz,
            mode=self.mode,
            bandwidth_hz=self.bandwidth_hz,
            group=self.group or group,
        )


@dataclass
class Visit:
    """One stretch of listening, inside this session only.

    Kept apart from `Station` because the session's trail is a sequence and
    the recent list is a set: coming back to a station for the third time is
    one entry in the list and three steps in the trail, and a Back button that
    collapsed the repeats would not go where the user just was.
    """

    frequency_hz: int
    mode: str = "wfm"
    bandwidth_hz: float = 200_000.0
    name: str = ""
    group: str = ""
    started_at: float = 0.0
    seconds: float = 0.0
    counted: bool = False

    @property
    def label(self) -> str:
        shown = format_hz(self.frequency_hz)
        return f"{self.name} - {shown}" if self.name else shown


class History:
    """The recent list, this session's trail, and the visit in progress."""

    def __init__(
        self,
        path: Path | None = None,
        dwell_seconds: float = DWELL_SECONDS,
        max_entries: int = MAX_ENTRIES,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.dwell_seconds = float(dwell_seconds)
        self.max_entries = int(max_entries)
        self.entries: list[Station] = []
        self.session: list[Visit] = []
        self.current: Visit | None = None
        # Bumped whenever the recent list changes in a way a view would have
        # to redraw. Views poll at 20 Hz and this list changes a few times an
        # hour, so comparing one integer is what keeps the chips from being
        # rebuilt under the user's cursor - the same reasoning as the
        # Discover list's signature.
        self.revision = 0
        # The entry the visit in progress is being credited to, and how much
        # of that visit has already been credited. Without the second number
        # every tick would add the visit's whole running total again.
        self._record: Station | None = None
        self._credited = 0.0
        self._ticked_at = 0.0

    @classmethod
    def open(cls, path: Path | None = None, **kwargs) -> History:
        store = cls(path=Path(path) if path else config_dir() / "history.json", **kwargs)
        store.load()
        return store

    def __len__(self) -> int:
        return len(self.entries)

    # -- recording ---------------------------------------------------------

    def tune(
        self,
        frequency_hz: float,
        mode: str = "wfm",
        bandwidth_hz: float = 200_000.0,
        group: str = "",
        now: float | None = None,
    ) -> None:
        """The radio has moved. Close the visit in progress and open another.

        Retuning inside the match tolerance is the same station - a click
        landing a bin or two off, or the band plan snapping to its raster - so
        it continues the visit rather than ending it. That matters more than
        it looks: without it, nudging the dial by 100 Hz would restart the
        dwell timer, and a station could be listened to all evening without
        ever being recorded.
        """
        now = time.time() if now is None else now
        current = self.current
        if (
            current is not None
            and abs(current.frequency_hz - frequency_hz) <= MATCH_TOLERANCE_HZ
        ):
            self.update(now)
            current.mode = mode
            current.bandwidth_hz = bandwidth_hz
            if self._record is not None:
                self._record.mode = mode
                self._record.bandwidth_hz = bandwidth_hz
            return
        self.leave(now)
        self.current = Visit(
            frequency_hz=int(round(frequency_hz)),
            mode=mode,
            bandwidth_hz=bandwidth_hz,
            group=group,
            started_at=now,
        )
        self._credited = 0.0
        self._ticked_at = now

    def name(self, text: str) -> None:
        """A station has said who it is, several seconds after being tuned to.

        Only a non-empty name is taken. A decoder that has lost the signal
        reports nothing rather than reporting an empty name, and letting that
        erase a name already read off the air would have the list forgetting
        stations while they were playing.
        """
        text = (text or "").strip()
        current = self.current
        if not text or current is None or current.name == text:
            return
        current.name = text
        if self._record is not None:
            self._record.name = text
            self.revision += 1

    def update(self, now: float | None = None) -> None:
        """Accrue listening time, and promote the visit once it has earned it.

        Called from a view's refresh timer, so it has to be cheap and has to
        tolerate being called at any rate at all - including not at all for a
        while, which is what `MAX_TICK_SECONDS` is for.
        """
        now = time.time() if now is None else now
        current = self.current
        if current is None:
            self._ticked_at = now
            return
        elapsed = now - self._ticked_at
        self._ticked_at = now
        if 0.0 < elapsed <= MAX_TICK_SECONDS:
            current.seconds += elapsed
        if not current.counted:
            if current.seconds < self.dwell_seconds:
                return
            current.counted = True
            self.session.append(current)
            self._record = self._promote(current, now)
            self._credited = current.seconds
            self.revision += 1
            return
        record = self._record
        if record is not None:
            record.seconds += current.seconds - self._credited
            self._credited = current.seconds
            record.last_heard = now

    def leave(self, now: float | None = None) -> None:
        """End the visit in progress: the page is going away, or the app is."""
        now = time.time() if now is None else now
        if self.current is not None:
            self.update(now)
        self.current = None
        self._record = None
        self._credited = 0.0

    def _promote(self, visit: Visit, now: float) -> Station:
        """Turn a visit that lasted into an entry in the recent list."""
        existing = self.find(visit.frequency_hz)
        if existing is None:
            existing = Station(
                frequency_hz=visit.frequency_hz,
                first_heard=visit.started_at,
                group=visit.group,
            )
            self.entries.append(existing)
        elif visit.group and not existing.group:
            existing.group = visit.group
        existing.mode = visit.mode
        existing.bandwidth_hz = visit.bandwidth_hz
        if visit.name:
            existing.name = visit.name
        existing.plays += 1
        existing.seconds += visit.seconds
        existing.last_heard = now
        self._trim()
        return existing

    def _trim(self) -> None:
        if len(self.entries) <= self.max_entries:
            return
        # Oldest by when it was last heard rather than by when it was first
        # added: a station listened to every morning should outlive one
        # visited once a year ago, whichever of them was discovered first.
        self.entries.sort(key=lambda e: e.last_heard, reverse=True)
        del self.entries[self.max_entries :]

    # -- reading -----------------------------------------------------------

    def find(
        self, frequency_hz: float, tolerance_hz: float = MATCH_TOLERANCE_HZ
    ) -> Station | None:
        for entry in self.entries:
            if entry.matches(frequency_hz, tolerance_hz):
                return entry
        return None

    def recent(self, limit: int = MAX_ENTRIES) -> tuple[Station, ...]:
        """Most recently listened to first."""
        ordered = sorted(self.entries, key=lambda e: e.last_heard, reverse=True)
        return tuple(ordered[: max(0, limit)])

    def most_played(self, limit: int = MAX_ENTRIES) -> tuple[Station, ...]:
        """Most time spent first, which is a different list from `recent`.

        Ties break on the recent order rather than arbitrarily, so a fresh
        history where everything has been heard once is still in a sensible
        order instead of whatever order it happened to be built in.
        """
        ordered = sorted(
            self.entries, key=lambda e: (e.seconds, e.last_heard), reverse=True
        )
        return tuple(ordered[: max(0, limit)])

    def previous(self) -> Visit | None:
        """Where the radio was before it came here - the Back button.

        The trail holds only visits that lasted, so Back goes to the last
        station actually listened to and not to whatever the dial passed
        through on the way. Repeats are kept, so Back after A, B, A goes to B
        and Back again goes to A.
        """
        trail = self.session
        if self.current is not None and self.current.counted:
            return trail[-2] if len(trail) >= 2 else None
        return trail[-1] if trail else None

    def clear(self) -> None:
        self.entries.clear()
        self.session.clear()
        self.current = None
        self._record = None
        self._credited = 0.0
        self.revision += 1

    # -- persistence -------------------------------------------------------

    def load(self) -> History:
        if self.path is None:
            return self
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Same bargain as the settings file: a history nobody can read is
            # an empty history, never a reason the app will not open.
            return self
        if isinstance(stored, list):
            fields = set(Station.__annotations__)
            self.entries = [
                Station(**{key: item[key] for key in item.keys() & fields})
                for item in stored
                if isinstance(item, dict) and "frequency_hz" in item
            ]
            self._trim()
        return self

    def save(self) -> bool:
        """Atomic, for the same reason `Settings.save` is: a half-written file
        would lose the whole list rather than the last change."""
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = _open_temporary(self.path)
            with temporary:
                json.dump([asdict(e) for e in self.recent()], temporary, indent=2)
            os.replace(temporary.name, self.path)
            return True
        except OSError:
            return False


__all__ = [
    "DWELL_SECONDS",
    "MAX_ENTRIES",
    "MAX_TICK_SECONDS",
    "History",
    "Station",
    "Visit",
]
