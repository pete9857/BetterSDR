"""One aircraft, as a row in the list of what is overhead.

The formatting is the point of this module, and it is deliberately separate
from the widget: a beginner reads "31,000 ft, climbing" and "451 kt heading
west" far faster than they read a bare number, and none of that arithmetic
needs a window to be tested.

Unlike a `SignalCard`, these are *updated* rather than rebuilt. An aircraft
reports twice a second and its altitude and position change continuously, so a
card that was thrown away and recreated on every snapshot would flicker, lose
the user's scroll position, and slam shut anything they were reading.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...decode.adsb import Aircraft
from ...scan.classifier import Strength
from ..levels import Level
from .icons import glyph
from .signalcard import StrengthBars

CARD_STYLE = """
QFrame#planeCard {
    background: #10151c; border: 1px solid #1d232b; border-radius: 6px;
}
QFrame#planeCard:hover { border-color: #2b323b; }
QLabel#planeIcon { font-size: 22px; }
QLabel#planeTitle { color: #e6edf3; font-size: 15px; font-weight: 600; }
QLabel#planeDetail { color: #8b98a5; font-size: 12px; }
QLabel#planeWhere { color: #6d7b89; font-size: 11px; }
QLabel#planeGround {
    color: #0b0e13; background: #8b98a5; border-radius: 3px;
    padding: 0px 5px; font-size: 10px; font-weight: 600;
}
"""

# Eight points is as fine as a heading is worth putting in words. The number
# is there too, for anyone who wants it.
_COMPASS = ("north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west")


def compass(track_deg: float) -> str:
    """The eight-point compass name for a track over the ground."""
    index = int((track_deg % 360.0) / 45.0 + 0.5) % 8
    return _COMPASS[index]


def altitude_text(
    altitude_ft: int | None,
    on_ground: bool = False,
    vertical_rate_fpm: int | None = None,
) -> str:
    """Height, and whether it is changing, in words.

    An aircraft that has not sent an altitude yet is not at zero feet, so it
    gets no altitude line at all - the same rule the classifier follows when
    it says "Unknown signal" rather than guessing.
    """
    if on_ground:
        return "On the ground"
    if altitude_ft is None:
        return ""
    text = f"{altitude_ft:,} ft"
    # 200 ft/min is inside the noise of a level cruise; calling that a climb
    # would have every aircraft on the screen permanently changing altitude.
    if vertical_rate_fpm is not None and abs(vertical_rate_fpm) >= 200:
        text += ", climbing" if vertical_rate_fpm > 0 else ", descending"
    return text


def speed_text(
    ground_speed_kt: float | None, track_deg: float | None = None
) -> str:
    """Speed over the ground and the direction it is going."""
    parts = []
    if ground_speed_kt is not None:
        parts.append(f"{ground_speed_kt:,.0f} kt")
    if track_deg is not None:
        parts.append(f"heading {compass(track_deg)} ({track_deg:.0f}°)")
    return "   ".join(parts)


def position_text(latitude: float | None, longitude: float | None) -> str:
    """Where it is, in the degrees-and-hemisphere form a map site takes."""
    if latitude is None or longitude is None:
        return ""
    ns = "N" if latitude >= 0 else "S"
    ew = "E" if longitude >= 0 else "W"
    return f"{abs(latitude):.4f}° {ns}, {abs(longitude):.4f}° {ew}"


def age_text(age_s: float) -> str:
    """How long ago it was last heard, rounded the way people say it."""
    if age_s < 2.0:
        return "just now"
    if age_s < 60.0:
        return f"{age_s:.0f} s ago"
    return f"{age_s / 60.0:.0f} min ago"


def strength_from_rssi(rssi_dbfs: float) -> Strength:
    """Four bars from the level a message arrived at.

    This is a level, not a signal-to-noise ratio, and the two are not
    interchangeable - which is why it does not go through `Strength.from_snr`.
    Every message on the screen has already passed its checkword, so even one
    bar is a real aircraft and not a maybe.

    The thresholds are spread across what a real sky actually produced: six
    aircraft heard indoors on 2026-08-28 arrived between -3 and -24 dBFS, so
    a scale topping out at -20 would have shown four bars for all of them and
    said nothing about which was overhead and which was leaving.
    """
    if rssi_dbfs >= -12.0:
        return Strength.STRONG
    if rssi_dbfs >= -20.0:
        return Strength.GOOD
    if rssi_dbfs >= -28.0:
        return Strength.FAIR
    return Strength.WEAK


def summary_line(aircraft: Aircraft) -> str:
    """Altitude and speed on one line, with whichever half is known."""
    parts = [
        altitude_text(
            aircraft.altitude_ft, aircraft.on_ground, aircraft.vertical_rate_fpm
        ),
        speed_text(aircraft.ground_speed_kt, aircraft.track_deg),
    ]
    return "   ·   ".join(part for part in parts if part)


def heard_line(aircraft: Aircraft, level: Level) -> str:
    """The provenance line: how long ago, how much of it, and how loud.

    Everything past the first clause is for somebody who wants to know how
    much to trust the row, so it appears from Standard upwards. The ICAO
    address is the aircraft's permanent identity and the only thing on the
    card that will still be true tomorrow, so it comes first once it is
    shown at all.
    """
    parts = [f"Heard {age_text(aircraft.age_s)}"]
    if level >= Level.STANDARD:
        parts.append(f"{aircraft.messages} messages")
    if level >= Level.EXPERT:
        parts.append(f"ICAO {aircraft.address}")
        parts.append(f"{aircraft.rssi_dbfs:.0f} dBFS")
    return "   ·   ".join(parts)


class AircraftCard(QFrame):
    """One aircraft: who it is, where it is, and how well it is being heard."""

    def __init__(
        self,
        aircraft: Aircraft,
        level: Level = Level.SIMPLE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.icao = aircraft.icao
        self.level = level
        self.setObjectName("planeCard")
        self.setStyleSheet(CARD_STYLE)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)

        icon = QLabel(glyph("plane"))
        icon.setObjectName("planeIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        outer.addWidget(icon)

        column = QVBoxLayout()
        column.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.title = QLabel()
        self.title.setObjectName("planeTitle")
        title_row.addWidget(self.title)
        self.ground = QLabel("ON GROUND")
        self.ground.setObjectName("planeGround")
        title_row.addWidget(self.ground)
        title_row.addStretch(1)
        column.addLayout(title_row)

        self.summary = QLabel()
        self.summary.setObjectName("planeDetail")
        column.addWidget(self.summary)
        self.position = QLabel()
        self.position.setObjectName("planeDetail")
        column.addWidget(self.position)
        self.heard = QLabel()
        self.heard.setObjectName("planeWhere")
        column.addWidget(self.heard)
        outer.addLayout(column, 1)

        self.bars = StrengthBars(strength_from_rssi(aircraft.rssi_dbfs))
        outer.addWidget(self.bars, 0, Qt.AlignmentFlag.AlignTop)

        self._latest = aircraft
        self.update_from(aircraft)

    def update_from(self, aircraft: Aircraft) -> None:
        """Repaint this card from a newer snapshot of the same aircraft."""
        if aircraft.icao != self.icao:
            return
        self._latest = aircraft
        # A station that has said its callsign names the row better than its
        # address can - "UAL245" against "A1B2C3" - but the callsign arrives
        # in its own message type, so a new aircraft is the address until one
        # does. Same reasoning as the RDS bookmark name.
        self.title.setText(aircraft.label)
        self.ground.setVisible(aircraft.on_ground)
        summary = summary_line(aircraft)
        self.summary.setText(summary)
        self.summary.setVisible(bool(summary))
        position = position_text(aircraft.latitude, aircraft.longitude)
        self.position.setText(position)
        self.position.setVisible(bool(position))
        self.heard.setText(heard_line(aircraft, self.level))
        self.bars.set_strength(strength_from_rssi(aircraft.rssi_dbfs))

    def set_level(self, level: Level) -> None:
        # Re-rendered here rather than left to the next snapshot: reception
        # may well be stopped, and a control that appears to do nothing until
        # something else happens reads as a broken one.
        self.level = level
        self.update_from(self._latest)


__all__ = [
    "AircraftCard",
    "age_text",
    "altitude_text",
    "compass",
    "heard_line",
    "position_text",
    "speed_text",
    "strength_from_rssi",
    "summary_line",
]
