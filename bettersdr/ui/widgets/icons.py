"""Glyphs for the kinds of signal the app can find.

Emoji rather than an icon set, for one practical reason and one honest one.
The practical one is that they need no asset pipeline, no licensing and no
scaling work, and Windows renders them from a font that is always present.
The honest one is that a beginner reads a picture of an aeroplane faster than
they read the word "airband", and speed of recognition is the entire job of
this column.

Names come from the band plan's `icon:` field, so adding a band with a new
icon is a data change here and nothing else.
"""

from __future__ import annotations

FALLBACK = "\N{SATELLITE ANTENNA}"

GLYPHS = {
    "radio": "\N{RADIO}",
    "globe": "\N{EARTH GLOBE EUROPE-AFRICA}",
    "truck": "\N{DELIVERY TRUCK}",
    "ham": "\N{SATELLITE ANTENNA}",
    "music": "\N{MUSICAL NOTE}",
    "nav": "\N{COMPASS}",
    "plane": "\N{AIRPLANE}",
    "satellite": "\N{SATELLITE}",
    "boat": "\N{SAILBOAT}",
    "cloud": "\N{CLOUD WITH RAIN}",
    "shield": "\N{SHIELD}",
    "key": "\N{KEY}",
    "walkie": "\N{PUBLIC ADDRESS LOUDSPEAKER}",
    "chip": "\N{ANTENNA WITH BARS}",
    "wave": "\N{WAVY DASH}",
    "question": "\N{BLACK QUESTION MARK ORNAMENT}",
}


def glyph(name: str) -> str:
    """The glyph for an icon name, or a neutral one for anything unknown.

    Never raises. A band plan with a typo in it should show a generic aerial,
    not crash the screen that is meant to be the friendly one.
    """
    return GLYPHS.get(name, FALLBACK)


__all__ = ["FALLBACK", "GLYPHS", "glyph"]
