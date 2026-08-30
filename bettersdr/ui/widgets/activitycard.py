"""One watched channel, as a card in the monitor list.

The difference from `signalcard.py` is the difference between the two screens.
A sweep's card answers "what is this?", and once answered it is done. A
monitor's card answers "how busy is this, and what does it sound like?", which
are questions whose answers keep changing - so this one is *updated* rather
than rebuilt.

That is not an optimisation. A card carries an expanded explanation and two
buttons somebody may be reaching for, and rebuilding it five times a second
would slam the first shut and move the others out from under the cursor. So
the list only rebuilds when the set of channels changes, and everything that
merely ticks - the count, the share, how long ago, what it last sounded like -
is written into the labels that are already there.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...scan import voice
from ...scan.monitor import Activity
from .icons import glyph
from .signalcard import StrengthBars

# What each verdict is worth saying in colour. Voice is the one anybody came
# for, so it is the only one that gets the accent; data is worth noticing and
# not worth stopping for, so it is warm but not bright; the rest are grey,
# because "this channel is static" is information the eye should skate over.
SOUND_COLOURS: dict[str, str] = {
    voice.VOICE: "#4ade80",
    voice.MUSIC: "#5ad1ff",
    voice.DATA: "#c9a33a",
    voice.TONE: "#6d7b89",
    voice.NOISE: "#6d7b89",
    voice.SILENCE: "#6d7b89",
    voice.UNCLEAR: "#6d7b89",
}

CARD_STYLE = """
QFrame#activity { background: #10151c; border: 1px solid #1d232b; border-radius: 6px; }
QFrame#activity:hover { border-color: #2b323b; }
QFrame#activity[playing="true"] { border-color: #4ade80; background: #101a15; }
QLabel#cardIcon { font-size: 24px; }
QLabel#cardTitle { color: #e6edf3; font-size: 15px; font-weight: 600; }
QLabel#cardWhere { color: #8b98a5; font-size: 12px; }
QLabel#cardBusy { color: #6d7b89; font-size: 11px; }
QLabel#cardAbout { color: #8b98a5; font-size: 12px; }
QLabel#cardPlaying {
    color: #0b0e13; background: #4ade80; border-radius: 3px;
    padding: 0px 5px; font-size: 10px; font-weight: 600;
}
QLabel#cardSkipped {
    color: #6d7b89; border: 1px solid #2b323b; border-radius: 3px;
    padding: 0px 5px; font-size: 10px;
}
QLabel#cardGuess {
    color: #0b0e13; background: #c9a33a; border-radius: 3px;
    padding: 0px 5px; font-size: 10px; font-weight: 600;
}
QPushButton#listen {
    background: #5ad1ff; color: #0b0e13; border: none;
    border-radius: 4px; padding: 6px 18px; font-weight: 600;
}
QPushButton#listen:hover { background: #7cdcff; }
QPushButton#minor {
    background: #161b22; color: #8b98a5;
    border: 1px solid #2b323b; border-radius: 4px; padding: 5px 12px;
}
QPushButton#minor:hover { border-color: #3d4650; color: #e6edf3; }
QPushButton#explain {
    background: transparent; color: #6d7b89; border: none;
    padding: 0px; text-align: left; font-size: 11px;
}
QPushButton#explain:hover { color: #8b98a5; }
"""


def heard_phrase(seconds: float) -> str:
    """How long ago, in the units somebody would actually say.

    Never a bare number of seconds past a minute: "last heard 214s ago" is a
    stopwatch reading, and what the reader wants to know is whether this
    channel is alive now, alive lately, or a note about earlier.
    """
    if seconds < 2.0:
        return "just now"
    if seconds < 60.0:
        return f"{seconds:.0f}s ago"
    if seconds < 3600.0:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.0f} h ago"


class ActivityCard(QFrame):
    """One channel the monitor is watching: how busy, how strong, what it is."""

    listenRequested = QtSignal(object)
    holdRequested = QtSignal(float)
    releaseRequested = QtSignal(float)
    skipRequested = QtSignal(float)
    resumeRequested = QtSignal(float)

    def __init__(self, activity: Activity, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.activity = activity
        self.setObjectName("activity")
        self.setStyleSheet(CARD_STYLE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)
        outer.addLayout(self._row(activity))

        self.about = QLabel("")
        self.about.setObjectName("cardAbout")
        self.about.setWordWrap(True)
        self.about.setVisible(False)

        self.explain = QPushButton("What is this?")
        self.explain.setObjectName("explain")
        self.explain.setCursor(Qt.CursorShape.PointingHandCursor)
        self.explain.clicked.connect(self._toggle)
        outer.addWidget(self.explain)
        outer.addWidget(self.about)
        self.update_activity(activity)

    # -- construction ------------------------------------------------------

    def _row(self, activity: Activity) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self.icon = QLabel(glyph(activity.signal.icon))
        self.icon.setObjectName("cardIcon")
        self.icon.setFixedWidth(34)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.icon)

        text = QVBoxLayout()
        text.setSpacing(1)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self.title = QLabel(activity.label)
        self.title.setObjectName("cardTitle")
        title_row.addWidget(self.title)
        self.sound = QLabel("")
        self.sound.setObjectName("cardSound")
        title_row.addWidget(self.sound)
        self.playing = QLabel("PLAYING")
        self.playing.setObjectName("cardPlaying")
        self.playing.setVisible(False)
        title_row.addWidget(self.playing)
        self.skipped = QLabel("SKIPPED")
        self.skipped.setObjectName("cardSkipped")
        self.skipped.setVisible(False)
        title_row.addWidget(self.skipped)
        title_row.addStretch(1)
        text.addLayout(title_row)

        self.where = QLabel("")
        self.where.setObjectName("cardWhere")
        text.addWidget(self.where)

        self.busy = QLabel("")
        self.busy.setObjectName("cardBusy")
        text.addWidget(self.busy)
        row.addLayout(text, 1)

        self.bars = StrengthBars(activity.signal.strength)
        row.addWidget(self.bars)

        self.hold_button = QPushButton("Hold")
        self.hold_button.setObjectName("minor")
        self.hold_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hold_button.setToolTip(
            "Stay on this channel and keep playing it, instead of going back "
            "to sweeping when it goes quiet."
        )
        self.hold_button.clicked.connect(self._hold_clicked)
        row.addWidget(self.hold_button)

        self.skip_button = QPushButton("Skip")
        self.skip_button.setObjectName("minor")
        self.skip_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_button.setToolTip(
            "Stop stopping on this channel. It stays on the list and keeps "
            "being counted."
        )
        self.skip_button.clicked.connect(self._skip_clicked)
        row.addWidget(self.skip_button)

        listen = QPushButton("Listen")
        listen.setObjectName("listen")
        listen.setCursor(Qt.CursorShape.PointingHandCursor)
        listen.clicked.connect(lambda: self.listenRequested.emit(self.activity.signal))
        row.addWidget(listen)
        return row

    # -- keeping up with the ledger ---------------------------------------

    def update_activity(self, activity: Activity, playing: bool = False) -> None:
        """Write the new numbers into the labels that are already there."""
        self.activity = activity
        signal = activity.signal
        self.icon.setText(glyph(signal.icon))
        self.title.setText(activity.label)
        self.bars.set_strength(signal.strength)

        where = f"{signal.display_frequency}  -  {signal.strength.label}"
        if activity.active:
            where += "  -  up now"
        self.where.setText(where)
        self.busy.setText(
            f"{activity.activity_phrase}  -  last heard "
            f"{heard_phrase(activity.silent_for)}"
        )

        verdict = activity.verdict
        if verdict is None:
            self.sound.setText("")
            self.sound.setVisible(False)
        else:
            colour = SOUND_COLOURS.get(verdict.kind, "#6d7b89")
            mark = "" if verdict.certain else "?"
            self.sound.setVisible(True)
            self.sound.setText(f"{glyph(verdict.icon)} {verdict.label}{mark}")
            self.sound.setStyleSheet(f"color: {colour}; font-size: 11px;")
            self.sound.setToolTip(verdict.explanation)

        self.playing.setVisible(playing)
        self.skipped.setVisible(activity.skipped)
        self.hold_button.setText("Holding" if activity.held else "Hold")
        self.skip_button.setText("Un-skip" if activity.skipped else "Skip")
        # A property rather than a second stylesheet, so the accent border on
        # the channel being played comes and goes without rebuilding anything.
        self.setProperty("playing", "true" if playing else "false")
        self.style().unpolish(self)
        self.style().polish(self)

        self.about.setText(self._explanation(activity))

    @staticmethod
    def _explanation(activity: Activity) -> str:
        """Everything the app knows about this channel, in prose.

        Three separate claims and they are kept separate on purpose: what the
        band plan says it is, what listening to it found, and how often it has
        been up. Running them together would let a confident sentence about
        the allocation carry a hedged one about the sound.
        """
        signal = activity.signal
        parts = [signal.description, "", f"Why we think so: {signal.explanation}"]
        if activity.verdict is not None:
            hedge = "" if activity.verdict.certain else " - though not certainly"
            parts += [
                "",
                f"Listening to it: {activity.verdict.explanation}{hedge}.",
            ]
        parts += [
            "",
            f"Heard in {activity.sightings} of the last {activity.passes} sweeps "
            f"of this band, last "
            f"{heard_phrase(activity.silent_for)}.",
        ]
        return "\n".join(parts)

    # -- the buttons -------------------------------------------------------

    def _hold_clicked(self) -> None:
        if self.activity.held:
            self.releaseRequested.emit(self.activity.frequency_hz)
        else:
            self.holdRequested.emit(self.activity.frequency_hz)

    def _skip_clicked(self) -> None:
        if self.activity.skipped:
            self.resumeRequested.emit(self.activity.frequency_hz)
        else:
            self.skipRequested.emit(self.activity.frequency_hz)

    def _toggle(self) -> None:
        showing = not self.about.isVisible()
        self.about.setVisible(showing)
        self.explain.setText("Hide" if showing else "What is this?")


__all__ = ["SOUND_COLOURS", "ActivityCard", "heard_phrase"]
