"""The pager messages heard on the channel being listened to.

This sits under the waterfall rather than beside it, and it stays hidden until
a pager transmission has actually been seen. Two reasons. A panel that appears
on every narrow FM channel and never says anything teaches a beginner to
ignore that part of the screen; and the moment it does appear is itself the
finding - "there is pager traffic here" - which is the same argument the
Discover list makes.

The formatting lives in plain functions at the top so it can be tested without
a window, the same as the aircraft card's.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...decode.pocsag import Page, PocsagState
from ..levels import Level

# Enough rows to see a burst of traffic without the panel taking the screen
# away from the spectrum it sits under.
PANEL_HEIGHT = 190
# How many rows are kept on screen. The receiver remembers more than this;
# a scrollback longer than this costs widgets for messages nobody is reading.
MAX_ROWS = 60

LOG_STYLE = """
QWidget#pagerLog { background: #0d1219; border-top: 1px solid #1d232b; }
QLabel#pagerTitle { color: #e6edf3; font-size: 13px; font-weight: 600; }
QLabel#pagerStatus { color: #6d7b89; font-size: 11px; }
QLabel#pagerEmpty { color: #6d7b89; font-size: 12px; }
/* Both the scroll area and its viewport have to be named - see
   `discover_view.py` for what happens when only one of them is. */
QScrollArea { border: none; background: #0d1219; }
QWidget#pagerRows { background: #0d1219; }
QFrame#pagerRow { background: transparent; }
QLabel#pagerTime { color: #6d7b89; font-size: 11px; }
QLabel#pagerWho { color: #5ad1ff; font-size: 12px; font-weight: 600; }
QLabel#pagerText { color: #e6edf3; font-size: 12px; }
QLabel#pagerBeep { color: #8b98a5; font-size: 12px; font-style: italic; }
QLabel#pagerNote { color: #6d7b89; font-size: 11px; }
"""

WAITING = (
    "Pager traffic on this channel. Messages appear here as they are sent - "
    "they arrive in bursts, so there may be nothing for a minute at a time."
)


def moment(received: float) -> str:
    """The wall-clock time a message arrived, to the second."""
    return time.strftime("%H:%M:%S", time.localtime(received))


def capcode_text(capcode: int) -> str:
    """The pager's own number, padded the way pager operators write it."""
    return f"{capcode:07d}"


def message_text(page: Page) -> str:
    """What the page actually says, or a plain sentence when it says nothing.

    A page with no message is not a fault and not an empty string: it is a
    real thing pagers do, and the honest answer is to say which it was.
    """
    if page.kind == "tone":
        return "Beep - no message"
    return page.text or "Empty message"


def detail_text(page: Page) -> str:
    """The engineering behind one row, for Expert."""
    parts = [f"{page.baud} bps", page.kind]
    if page.errors:
        parts.append(f"{page.errors} lost codeword{'s' if page.errors > 1 else ''}")
    return "   ".join(parts)


def status_text(state: PocsagState | None) -> str:
    """The line under the heading: what is being read, and how well."""
    if state is None:
        return ""
    if not state.batches:
        return "Listening"
    rate = "" if state.baud is None else f"{state.baud} bps   "
    return (
        f"{rate}{len(state.pages)} message{'' if len(state.pages) == 1 else 's'}   "
        f"{state.quality * 100:.0f}% of codewords intact"
    )


class _Row(QFrame):
    """One message. Built once and never updated - a page does not change."""

    def __init__(self, page: Page, level: Level, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pagerRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(10)

        clock = QLabel(moment(page.received))
        clock.setObjectName("pagerTime")
        clock.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(clock)

        who = QLabel(capcode_text(page.capcode))
        who.setObjectName("pagerWho")
        who.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(who)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        text = QLabel(message_text(page))
        text.setObjectName("pagerBeep" if page.kind == "tone" else "pagerText")
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(text)
        self.detail = QLabel(detail_text(page))
        self.detail.setObjectName("pagerNote")
        self.detail.setVisible(level >= Level.EXPERT)
        body.addWidget(self.detail)
        layout.addLayout(body, 1)

    def set_level(self, level: Level) -> None:
        self.detail.setVisible(level >= Level.EXPERT)


class PagerLog(QWidget):
    """Pager messages, newest at the top."""

    def __init__(
        self, level: Level = Level.SIMPLE, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.level = level
        self._rows: list[_Row] = []
        # The last page put on screen, held by identity. The receiver appends
        # to its list and drops the oldest off the front, so counting is not
        # enough to tell what is new - but the objects themselves are the same
        # ones from snapshot to snapshot.
        self._last: Page | None = None
        self._build()
        self.setMaximumHeight(PANEL_HEIGHT)
        self.setVisible(False)

    def _build(self) -> None:
        self.setObjectName("pagerLog")
        self.setStyleSheet(LOG_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(10)
        title = QLabel("Pager messages")
        title.setObjectName("pagerTitle")
        head.addWidget(title)
        self.status = QLabel("")
        self.status.setObjectName("pagerStatus")
        head.addWidget(self.status)
        head.addStretch(1)
        outer.addLayout(head)

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.viewport().setAutoFillBackground(True)
        palette = self.area.viewport().palette()
        palette.setColor(self.area.viewport().backgroundRole(), QColor("#0d1219"))
        self.area.viewport().setPalette(palette)
        holder = QWidget()
        holder.setObjectName("pagerRows")
        self.rows = QVBoxLayout(holder)
        self.rows.setContentsMargins(0, 0, 6, 0)
        self.rows.setSpacing(2)
        self.empty = QLabel(WAITING)
        self.empty.setObjectName("pagerEmpty")
        self.empty.setWordWrap(True)
        self.rows.addWidget(self.empty)
        self.rows.addStretch(1)
        self.area.setWidget(holder)
        outer.addWidget(self.area, 1)

    def set_level(self, level: Level) -> None:
        self.level = level
        for row in self._rows:
            row.set_level(level)

    def clear(self) -> None:
        """Forget everything - a retune is a different transmitter."""
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        self._last = None
        self.empty.setVisible(True)
        self.status.setText("")
        self.setVisible(False)

    def update_state(self, state: PocsagState | None) -> None:
        """Show whatever is new since the last call.

        Rows are appended rather than rebuilt. A message never changes once it
        has been decoded, and rebuilding the list every frame would throw away
        the user's scroll position thirty times a second - which on a busy
        channel means they could never finish reading one.
        """
        if state is None:
            if self.isVisible():
                self.clear()
            return
        # The panel earns its place only once the channel has been shown to
        # carry pager data. A sync codeword is proof of that; a message is
        # not needed, and may be minutes away.
        self.setVisible(state.batches > 0)
        self.status.setText(status_text(state))
        for page in self._fresh(state.pages):
            self._add(page)

    def _fresh(self, pages: tuple[Page, ...]) -> list[Page]:
        if self._last is None:
            return list(pages)
        for index in range(len(pages) - 1, -1, -1):
            if pages[index] is self._last:
                return list(pages[index + 1 :])
        # The last row we showed has aged off the receiver's own list, which
        # takes two hundred messages. Everything held is new to the screen.
        return list(pages)

    def _add(self, page: Page) -> None:
        row = _Row(page, self.level)
        self.rows.insertWidget(0, row)
        self._rows.insert(0, row)
        self._last = page
        self.empty.setVisible(False)
        while len(self._rows) > MAX_ROWS:
            old = self._rows.pop()
            old.setParent(None)
            old.deleteLater()


__all__ = [
    "MAX_ROWS",
    "PANEL_HEIGHT",
    "PagerLog",
    "capcode_text",
    "detail_text",
    "message_text",
    "moment",
    "status_text",
]
