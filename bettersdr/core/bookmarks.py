"""Named frequencies: the memory channels every radio has had since the 1970s.

Pure logic, no Qt, for the same reason `doctor.py` has no UI imports: the
frequency manager window, the "save this" button on a signal card and any
future import tool all have to agree about what a bookmark is, and the way to
guarantee that is for there to be one implementation of it.

A bookmark carries the mode and bandwidth alongside the frequency. That is not
padding - recalling a marine channel and getting wideband FM because the last
thing you listened to was a broadcast station is exactly the sort of "the app
is broken" moment this project exists to avoid.

Import and export are CSV rather than SDR#'s XML. CSV is what people actually
exchange in forum posts and what a spreadsheet opens, and the columns are
readable enough that someone can build a list by hand.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .settings import _open_temporary, config_dir

# Two entries closer together than this are the same station as far as
# "is this already saved" is concerned. Wide enough to cover a click landing
# a bin or two off, narrow enough not to merge adjacent FM channels.
MATCH_TOLERANCE_HZ = 5_000.0

CSV_COLUMNS = ("name", "frequency_hz", "mode", "bandwidth_hz", "group", "notes")


@dataclass(frozen=True)
class Bookmark:
    """One saved frequency and how to listen to it."""

    name: str
    frequency_hz: int
    mode: str = "wfm"
    bandwidth_hz: float = 200_000.0
    group: str = "General"
    notes: str = ""

    @property
    def label(self) -> str:
        """`KUOW 94.9 MHz`, the way it should read on a button."""
        hz = self.frequency_hz
        if hz >= 1_000_000:
            # Trailing zeros trimmed off the number itself, not off the unit:
            # rstripping the whole string would eat the "z" of "MHz" on a
            # frequency that happened to end in one.
            shown = f"{f'{hz / 1e6:.4f}'.rstrip('0').rstrip('.')} MHz"
        else:
            shown = f"{hz / 1e3:.1f} kHz"
        return f"{self.name} - {shown}" if self.name else shown

    def matches(
        self, frequency_hz: float, tolerance_hz: float = MATCH_TOLERANCE_HZ
    ) -> bool:
        return abs(self.frequency_hz - frequency_hz) <= tolerance_hz


@dataclass
class BookmarkStore:
    """An ordered list of bookmarks with JSON persistence and CSV exchange."""

    path: Path | None = None
    entries: list[Bookmark] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path | None = None) -> BookmarkStore:
        store = cls(path=Path(path) if path else config_dir() / "bookmarks.json")
        store.load()
        return store

    # -- collection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def groups(self) -> list[str]:
        """Every group in use, in the order a list should offer them."""
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.group or "General", None)
        return sorted(seen)

    def in_group(self, group: str) -> list[Bookmark]:
        return [entry for entry in self.entries if (entry.group or "General") == group]

    def find(
        self, frequency_hz: float, tolerance_hz: float = MATCH_TOLERANCE_HZ
    ) -> Bookmark | None:
        """The saved entry on this frequency, if there is one.

        Used to make the save control a toggle rather than a way to
        accumulate six copies of the same station.
        """
        for entry in self.entries:
            if entry.matches(frequency_hz, tolerance_hz):
                return entry
        return None

    # -- editing -----------------------------------------------------------

    def add(self, bookmark: Bookmark, replace_existing: bool = True) -> Bookmark:
        existing = self.find(bookmark.frequency_hz)
        if existing is not None:
            if not replace_existing:
                return existing
            self.entries[self.entries.index(existing)] = bookmark
        else:
            self.entries.append(bookmark)
        self._sort()
        return bookmark

    def remove(self, bookmark: Bookmark) -> bool:
        try:
            self.entries.remove(bookmark)
        except ValueError:
            return False
        return True

    def rename(self, bookmark: Bookmark, name: str) -> Bookmark:
        updated = replace(bookmark, name=name)
        self.entries[self.entries.index(bookmark)] = updated
        return updated

    def clear(self) -> None:
        self.entries.clear()

    def _sort(self) -> None:
        # By group then frequency: a list sorted by when things were added is
        # a list nobody can find anything in.
        self.entries.sort(key=lambda e: ((e.group or "General").lower(), e.frequency_hz))

    # -- persistence -------------------------------------------------------

    def load(self) -> BookmarkStore:
        if self.path is None:
            return self
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self
        if isinstance(stored, list):
            fields = set(Bookmark.__annotations__)
            self.entries = [
                Bookmark(**{key: item[key] for key in item.keys() & fields})
                for item in stored
                if isinstance(item, dict) and "frequency_hz" in item
            ]
            self._sort()
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
                json.dump([asdict(e) for e in self.entries], temporary, indent=2)
            os.replace(temporary.name, self.path)
            return True
        except OSError:
            return False

    # -- exchange ----------------------------------------------------------

    def to_csv(self) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for entry in self.entries:
            writer.writerow({key: getattr(entry, key) for key in CSV_COLUMNS})
        return buffer.getvalue()

    def from_csv(self, text: str, merge: bool = True) -> int:
        """Read a CSV list, returning how many entries were taken from it.

        Deliberately forgiving. A row missing a name, a group or a bandwidth
        is still a usable frequency, and rejecting somebody's list because one
        line of it is short would be the wrong trade - so only an unreadable
        frequency disqualifies a row.
        """
        if not merge:
            self.clear()
        taken = 0
        for row in csv.DictReader(io.StringIO(text)):
            try:
                frequency = int(float(row.get("frequency_hz") or ""))
            except (TypeError, ValueError):
                continue
            self.add(
                Bookmark(
                    name=(row.get("name") or "").strip(),
                    frequency_hz=frequency,
                    mode=(row.get("mode") or "wfm").strip().lower(),
                    bandwidth_hz=_as_float(row.get("bandwidth_hz"), 200_000.0),
                    group=(row.get("group") or "General").strip() or "General",
                    notes=(row.get("notes") or "").strip(),
                )
            )
            taken += 1
        return taken


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def from_signal(signal: object, group: str = "Found by scanning") -> Bookmark:
    """Turn a scan result into a bookmark, keeping the classifier's decision.

    The scanner already worked out what this is and how to demodulate it, and
    a bookmark that threw that away would recall the frequency but not the
    station - which on the airband is the difference between hearing a pilot
    and hearing nothing at all.
    """
    return Bookmark(
        name=str(getattr(signal, "label", "") or ""),
        frequency_hz=int(round(float(signal.frequency_hz))),
        mode=str(getattr(signal, "mode", "wfm")),
        bandwidth_hz=float(getattr(signal, "demod_bandwidth_hz", 200_000.0)),
        group=group,
        notes=str(getattr(signal, "description", "") or ""),
    )


__all__ = [
    "CSV_COLUMNS",
    "MATCH_TOLERANCE_HZ",
    "Bookmark",
    "BookmarkStore",
    "from_signal",
]
