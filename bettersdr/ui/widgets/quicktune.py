"""Favourites and recently played, as a row of chips.

This is what the landing screen shows before it has swept anything. A scan
takes five seconds and finds what is on the air; this takes none and shows
what the user already cared about, which on the second run of the app is the
faster route to hearing something.

Favourites come first because they were chosen deliberately, and recently
played follows because it was not. A station that is both appears once, as a
favourite - two chips for one frequency would be the list disagreeing with
itself.

The whole strip hides when there is nothing in it, so a first run is not met
by an empty rectangle labelled with a promise. The same argument as the pager
log: the moment it appears is itself worth something.

Ordering and wording are plain functions at the top, testable without a
window, the same as `pagerlog.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.bookmarks import MATCH_TOLERANCE_HZ, Bookmark, format_hz
from ...core.history import Station
from .icons import glyph

# How many chips fit across the window before they start being more work to
# read than a scan would be. Favourites are never crowded out by recents -
# see `quick_list`.
MAX_CHIPS = 8

STRIP_STYLE = """
QWidget#quickTune { background: transparent; }
QLabel#quickTitle { color: #8b98a5; font-size: 12px; }
QPushButton#quickChip {
    background: #10151c; color: #cbd5e0;
    border: 1px solid #2b323b; border-radius: 13px; padding: 5px 14px;
}
QPushButton#quickChip:hover { border-color: #5ad1ff; color: #e6edf3; }
QPushButton#quickFavourite {
    background: #131c22; color: #e6edf3;
    border: 1px solid #35505e; border-radius: 13px; padding: 5px 14px;
}
QPushButton#quickFavourite:hover { border-color: #5ad1ff; }
"""


@dataclass(frozen=True)
class QuickEntry:
    """One chip: somewhere to go, and how to listen when it gets there."""

    name: str
    frequency_hz: int
    mode: str
    bandwidth_hz: float
    favourite: bool
    detail: str = ""

    @property
    def label(self) -> str:
        shown = format_hz(self.frequency_hz)
        return f"{self.name} - {shown}" if self.name else shown


def ago(seconds: float) -> str:
    """How long ago, in the units somebody would actually say it in.

    Never more precise than it is useful to be. "2 days ago" is what the user
    wants from a list they are scanning with their eyes; a timestamp is what
    they want when they are debugging, and this is not that screen.
    """
    if seconds < 0:
        return ""
    if seconds < 90:
        return "just now"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{round(minutes)} minutes ago"
    hours = minutes / 60.0
    if hours < 24:
        count = round(hours)
        return "an hour ago" if count == 1 else f"{count} hours ago"
    days = round(hours / 24.0)
    return "yesterday" if days == 1 else f"{days} days ago"


def spent(seconds: float) -> str:
    """How long was spent listening, worded the same way."""
    if seconds < 90:
        return f"{max(1, round(seconds))} seconds"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{round(minutes)} minutes"
    hours = minutes / 60.0
    return "an hour" if hours < 1.5 else f"{hours:.0f} hours"


def quick_list(
    favourites: Sequence[Bookmark],
    recent: Sequence[Station],
    now: float,
    limit: int = MAX_CHIPS,
) -> list[QuickEntry]:
    """The chips to show, favourites first and no frequency twice.

    Favourites are taken before the limit is applied to the recents rather
    than competing with them for the same places. Somebody who stars nine
    stations has said what they want on this row; quietly dropping the ninth
    in favour of something they listened to once would be the app overruling
    that.
    """
    chips: list[QuickEntry] = []
    for entry in favourites:
        chips.append(
            QuickEntry(
                name=entry.name,
                frequency_hz=entry.frequency_hz,
                mode=entry.mode,
                bandwidth_hz=entry.bandwidth_hz,
                favourite=True,
                detail=entry.notes or entry.group,
            )
        )
    room = max(0, limit - len(chips))
    for station in recent:
        if room <= 0:
            break
        if any(
            abs(chip.frequency_hz - station.frequency_hz) <= MATCH_TOLERANCE_HZ
            for chip in chips
        ):
            continue
        heard = ago(now - station.last_heard) if station.last_heard else ""
        if heard and station.seconds:
            heard = f"{heard}, listened to for {spent(station.seconds)}"
        chips.append(
            QuickEntry(
                name=station.name,
                frequency_hz=station.frequency_hz,
                mode=station.mode,
                bandwidth_hz=station.bandwidth_hz,
                favourite=False,
                detail=heard,
            )
        )
        room -= 1
    return chips


class QuickTune(QWidget):
    """The strip itself: a heading and a row of chips that tune when pressed."""

    tuneRequested = QtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("quickTune")
        self.setStyleSheet(STRIP_STYLE)
        self._chips: list[QPushButton] = []
        # What is currently drawn, so a 20 Hz poll rebuilds nothing until the
        # list actually changes. Rebuilding buttons under a cursor is the same
        # fault the Discover list's signature avoids.
        self._shown: tuple[tuple[int, str, bool, str], ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title = QLabel("Pick up where you left off")
        self.title.setObjectName("quickTitle")
        layout.addWidget(self.title)
        self.row = QHBoxLayout()
        self.row.setSpacing(6)
        self.row.addStretch(1)
        layout.addLayout(self.row)
        self.setVisible(False)

    def show_entries(self, entries: Sequence[QuickEntry]) -> bool:
        """Draw these chips. Returns whether anything actually changed."""
        signature = tuple(
            (e.frequency_hz, e.label, e.favourite, e.detail) for e in entries
        )
        if signature == self._shown:
            return False
        self._shown = signature
        self._clear()
        for entry in entries:
            button = QPushButton(self._chip_text(entry))
            button.setObjectName(
                "quickFavourite" if entry.favourite else "quickChip"
            )
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(self._chip_tooltip(entry))
            # The default argument is not style: a lambda closing over the
            # loop variable would give every chip the last entry.
            button.clicked.connect(
                lambda _checked=False, chosen=entry: self.tuneRequested.emit(chosen)
            )
            self.row.insertWidget(self.row.count() - 1, button)
            self._chips.append(button)
        self.setVisible(bool(entries))
        return True

    @staticmethod
    def _chip_text(entry: QuickEntry) -> str:
        mark = glyph("star") if entry.favourite else glyph("clock")
        return f"{mark}  {entry.label}"

    @staticmethod
    def _chip_tooltip(entry: QuickEntry) -> str:
        lead = "One of your favourites" if entry.favourite else "Recently played"
        return f"{lead}\n{entry.detail}" if entry.detail else lead

    def _clear(self) -> None:
        for chip in self._chips:
            self.row.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips.clear()


__all__ = ["MAX_CHIPS", "QuickEntry", "QuickTune", "ago", "quick_list", "spent"]
