"""Band plan loading.

The band plan is the app's prior knowledge about what lives where. Two very
different consumers share it, which is why it is a data file: the spectrum
ribbon labels the picture with it, and the classifier uses it to turn "a
150 kHz constant-power signal at 94.9 MHz" into "an FM radio station".

Regions are separate YAML files, so supporting Europe later is a data change
rather than a code change.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

BANDPLAN_DIR = Path(__file__).resolve().parent
DEFAULT_REGION = "us"


@dataclass(frozen=True)
class Channel:
    """One named channel inside a band.

    Two names, because a beginner and a regulator call the same thing
    different things and both are worth showing: `name` is what somebody
    would say out loud - "Channel 16" - and `official` is the designation it
    carries in the rule book, which is the phrase to search for and the one
    printed on a chart. `use` is the plain-English half, written to the same
    standard as `Band.description`.
    """

    name: str
    frequency_hz: int
    use: str
    official: str = ""


@dataclass(frozen=True)
class Allocation:
    """What a stretch of dial with nothing to listen to is licensed for.

    Deliberately not a `Band`: nothing tunes from these, nothing scans them
    and the classifier never sees them. They exist so that the listening
    screen can answer "what is this part of the dial?" everywhere, instead of
    only where the app has something to offer.
    """

    name: str
    start_hz: int
    end_hz: int
    use: str

    @property
    def width_hz(self) -> int:
        return self.end_hz - self.start_hz

    def contains(self, hz: float) -> bool:
        return self.start_hz <= hz <= self.end_hz


@dataclass(frozen=True)
class Segment:
    """One selectable stretch of the dial, whatever the app knows about it.

    A `Band` is a promise that the app has something to offer somewhere; an
    `Allocation` only says who a stretch is licensed to. Both are places a
    user at Expert level may legitimately point the receiver, so the
    discovery screen needs one list that covers the *whole* tunable dial
    rather than the handful of bands worth putting in front of a beginner.
    That is what this is: a target, with whichever of the two it came from
    still attached so a sweep can take the band's window preference with it.

    Deliberately not a `Band`. Inventing bands for the space between the
    bands would put "Federal government" on the spectrum ribbon, into the
    classifier and into the mode-on-tuning rule, none of which want it.
    """

    name: str
    start_hz: int
    end_hz: int
    description: str
    band: Band | None = None
    icon: str = ""

    @property
    def key(self) -> str:
        """A stable identity to remember a selection by.

        The span rather than the name: two stretches of dial are both called
        "Federal government" and they are not the same target, and a name is
        the field most likely to be reworded between releases.
        """
        return f"{self.start_hz}-{self.end_hz}"

    @property
    def width_hz(self) -> int:
        return self.end_hz - self.start_hz

    @property
    def sample_rate_hz(self) -> int | None:
        """How wide a window this stretch wants, where it has an opinion.

        Only a band can have one - see `Band.sample_rate_hz` - so a stretch
        the app knows nothing about beyond who it is licensed to gets the
        default, and `frontend.safe_sample_rate` still narrows it where the
        window would reach below 0 Hz.
        """
        return None if self.band is None else self.band.sample_rate_hz

    def contains(self, hz: float) -> bool:
        return self.start_hz <= hz <= self.end_hz

    @classmethod
    def of(cls, band: Band) -> Segment:
        """A band as a thing to point the receiver at."""
        return cls(
            name=band.name,
            start_hz=band.start_hz,
            end_hz=band.end_hz,
            description=band.description,
            band=band,
            icon=band.icon,
        )


@dataclass(frozen=True)
class Band:
    """One allocation, as the user should hear it described."""

    name: str
    start_hz: int
    end_hz: int
    mode: str
    bandwidth_hz: float
    description: str
    colour: str
    icon: str = ""
    raster_hz: float | None = None
    # Centre of the band's first channel. Where the channels actually start is
    # a per-band fact, not something to derive: NOAA weather channels begin at
    # the band edge, US FM broadcast begins half a channel in at 88.1, and US
    # AM begins at 540 kHz with the band edge 10 kHz below it. Guessing one
    # convention for all of them put 162.550 MHz on screen as 162.537.
    raster_base_hz: float | None = None
    # Offered as a one-click scan target on the discovery screen. Not every
    # allocation is worth putting in front of a beginner - a band that is
    # empty, encrypted or silent most of the time teaches them the app does
    # not work - so which ones appear is part of the data, not a UI decision.
    scan: bool = False
    # How wide a window to look through in this band, when the default 2.4 MHz
    # is the wrong answer. AM broadcast is the case that forced it: a 2.4 MHz
    # window at 710 kHz reaches below 0 Hz, where the upconverter's local
    # oscillator leak dominates the ADC, and even once that is avoided the
    # band is crowded enough that a narrow window is worth 30 dB of audio.
    sample_rate_hz: int | None = None
    # True where stations transmit around the clock. It decides what a bare
    # carrier means: AM and FM broadcasters radiate one continuously and their
    # sidebands rise and fall with the programme, so a short dwell between
    # words measures the carrier alone. On the airband a channel is silent
    # unless somebody is speaking, so a steady carrier there is interference.
    # Same measurement, opposite meaning, and the band is what tells them
    # apart.
    continuous: bool = False
    # The named channels inside this band, where it has any. Marine VHF is
    # the case that forced them: "156.800 MHz" and "Channel 16" are the same
    # fact, and only one of them is what anybody on a boat would say. Bands
    # whose channels have no names - AM and FM broadcast - have none here,
    # because a number that is already the name is not worth repeating.
    channels: tuple[Channel, ...] = ()

    @property
    def center_hz(self) -> float:
        return (self.start_hz + self.end_hz) / 2.0

    @property
    def width_hz(self) -> int:
        return self.end_hz - self.start_hz

    def contains(self, hz: float) -> bool:
        return self.start_hz <= hz <= self.end_hz

    def snap(self, hz: float) -> float:
        """Nearest legal channel centre, if this band has a channel raster.

        FM broadcast in the US sits on odd tenths of a megahertz, so snapping
        turns a rough click on the spectrum - or a detector's centroid a few
        kilohertz out - into an actual station rather than something 40 kHz
        off it.
        """
        if not self.raster_hz:
            return hz
        base = (
            self.raster_base_hz if self.raster_base_hz is not None else self.start_hz
        )
        return base + round((hz - base) / self.raster_hz) * self.raster_hz

    def channel(self, hz: float) -> Channel | None:
        """The named channel at `hz`, if this band has one there.

        Half a channel either side, taken from the raster where the band has
        one and from the channel width where it does not - so tuning between
        two marine channels names neither, rather than naming whichever is a
        hertz closer.
        """
        if not self.channels:
            return None
        tolerance = (self.raster_hz or self.bandwidth_hz) / 2.0
        nearest = min(self.channels, key=lambda ch: abs(ch.frequency_hz - hz))
        if abs(nearest.frequency_hz - hz) > tolerance:
            return None
        return nearest


def _prose(text: str) -> str:
    """A folded YAML scalar as one line of prose."""
    return " ".join(str(text).split())


@functools.lru_cache(maxsize=4)
def _read(region: str = DEFAULT_REGION) -> dict:
    path = BANDPLAN_DIR / f"{region}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no band plan for region {region!r} at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _channels(entry: dict) -> tuple[Channel, ...]:
    return tuple(
        sorted(
            (
                Channel(
                    name=channel["name"],
                    frequency_hz=int(channel["hz"]),
                    use=_prose(channel.get("use", "")),
                    official=_prose(channel.get("official", "")),
                )
                for channel in entry.get("channels", [])
            ),
            key=lambda channel: channel.frequency_hz,
        )
    )


@functools.lru_cache(maxsize=4)
def load(region: str = DEFAULT_REGION) -> tuple[Band, ...]:
    """Every band for a region, ordered by frequency."""
    raw = _read(region)
    bands = [
        Band(
            name=entry["name"],
            start_hz=int(entry["start_hz"]),
            end_hz=int(entry["end_hz"]),
            mode=entry.get("mode", "nfm"),
            bandwidth_hz=float(entry.get("bandwidth_hz", 12_500)),
            description=_prose(entry.get("description", "")),
            colour=entry.get("colour", "#7a7a7a"),
            icon=entry.get("icon", ""),
            raster_hz=(
                float(entry["raster_hz"]) if entry.get("raster_hz") else None
            ),
            raster_base_hz=(
                float(entry["raster_base_hz"])
                if entry.get("raster_base_hz")
                else None
            ),
            scan=bool(entry.get("scan", False)),
            sample_rate_hz=(
                int(entry["sample_rate_hz"]) if entry.get("sample_rate_hz") else None
            ),
            continuous=bool(entry.get("continuous", False)),
            channels=_channels(entry),
        )
        for entry in raw.get("bands", [])
    ]
    return tuple(sorted(bands, key=lambda band: band.start_hz))


@functools.lru_cache(maxsize=4)
def allocations(region: str = DEFAULT_REGION) -> tuple[Allocation, ...]:
    """What the gaps between the bands are licensed for, in frequency order."""
    return tuple(
        sorted(
            (
                Allocation(
                    name=entry["name"],
                    start_hz=int(entry["start_hz"]),
                    end_hz=int(entry["end_hz"]),
                    use=_prose(entry.get("use", "")),
                )
                for entry in _read(region).get("allocations", [])
            ),
            key=lambda allocation: allocation.start_hz,
        )
    )


def scannable(region: str = DEFAULT_REGION) -> tuple[Band, ...]:
    """The bands offered as one-click scan targets, in frequency order."""
    return tuple(band for band in load(region) if band.scan)


def _gaps(
    spans: list[tuple[int, int]], low_hz: int, high_hz: int
) -> list[tuple[int, int]]:
    """Whatever `spans` leaves uncovered between two frequencies."""
    holes: list[tuple[int, int]] = []
    edge = low_hz
    for start, end in sorted(spans):
        if start > edge:
            holes.append((edge, min(start, high_hz)))
        edge = max(edge, end)
        if edge >= high_hz:
            break
    if edge < high_hz:
        holes.append((edge, high_hz))
    return [(a, b) for a, b in holes if b > a]


def coverage(
    low_hz: int, high_hz: int, region: str = DEFAULT_REGION
) -> tuple[Segment, ...]:
    """Every stretch of dial between two frequencies, as things to point at.

    The bands first, then the allocations for the space between them, then a
    plain "Unallocated" entry for anything neither speaks for - so the list
    covers the whole span with no silent holes in it. A hole would be the
    worse failure of the two: a user scrolling a list that claims to be the
    whole dial has no way to notice that 700 MHz of it is simply missing.

    Nested bands stay in - Remote Controls sits inside 70 cm Amateur - so the
    list is ordered but not a partition. Somebody who wants only the 433 MHz
    gadget band should be able to ask for it, and a caller sweeping both gets
    them merged rather than swept twice; see `merge_ranges`.
    """
    low_hz, high_hz = int(low_hz), int(high_hz)
    segments = [
        Segment(
            name=band.name,
            start_hz=max(low_hz, band.start_hz),
            end_hz=min(high_hz, band.end_hz),
            description=band.description,
            band=band,
            icon=band.icon,
        )
        for band in load(region)
        if band.end_hz > low_hz and band.start_hz < high_hz
    ]
    spanned = [(band.start_hz, band.end_hz) for band in load(region)]
    for gap_low, gap_high in _gaps(spanned, low_hz, high_hz):
        named = [
            allocation
            for allocation in allocations(region)
            if allocation.end_hz > gap_low and allocation.start_hz < gap_high
        ]
        segments.extend(
            Segment(
                name=allocation.name,
                start_hz=max(gap_low, allocation.start_hz),
                end_hz=min(gap_high, allocation.end_hz),
                description=allocation.use,
            )
            for allocation in named
        )
        # Nothing in the band plan speaks for this stretch at all. It is
        # still tunable, so it is still offered - saying so plainly beats
        # leaving a hole in a list that claims to be the whole dial.
        holes = [(a.start_hz, a.end_hz) for a in named]
        segments.extend(
            Segment(
                name="Unallocated",
                start_hz=start,
                end_hz=end,
                description=(
                    "Nothing in the band plan covers this stretch. The "
                    "receiver can still listen across it."
                ),
            )
            for start, end in _gaps(holes, gap_low, gap_high)
        )
    # By where they start, then widest first, so a band comes before anything
    # nested inside it rather than the order depending on the file.
    return tuple(
        sorted(segments, key=lambda segment: (segment.start_hz, -segment.width_hz))
    )


def merge_ranges(
    spans: Iterable[tuple[float, float]], tolerance_hz: float = 0.0
) -> tuple[tuple[float, float], ...]:
    """Overlapping or touching spans joined into one, in frequency order.

    A selection of several stretches is what the user asked to hear, not a
    plan for how to sweep it. Two that abut - Marine VHF and the Federal
    government slice above it - are one continuous range and must be swept as
    one: stepped separately, the boundary is covered twice and a station
    sitting on it is measured by two tiles that each own half of it.
    """
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + tolerance_hz:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def sweep_ranges(
    segments: Iterable[Segment],
    rate_for: Callable[[Segment], int | None] | None = None,
) -> tuple[tuple[int, int, int | None], ...]:
    """A selection of stretches of dial, as ranges a sweep can be planned from.

    Two things happen here and both matter. Touching or overlapping stretches
    are joined, because stepping them separately covers the boundary twice
    and leaves a station sitting on it measured by two tiles that each own
    half of it. And they are only joined when they want the *same* window:
    the AM broadcast band ends exactly where the fixed links above it begin,
    and merging those two would sweep 1.7-1.8 MHz through a 240 kHz window or
    the whole AM band through a 2.4 MHz one, both of which the app has
    measured to be wrong.

    `rate_for` is how a caller says what window a stretch will *actually* get
    rather than what it asked for, and passing it is what makes the merge
    safe. A merge moves a range's lower edge down, and how wide a window may
    be depends on how close its lower edge comes to 0 Hz - so grouping on the
    stated preference alone could join two stretches that will be given
    different windows, and then sweep both through whichever one the merged
    edge turns out to allow. Grouping on the answer instead cannot: the
    answer only ever narrows as the dial goes down, so every member of a
    group shares the edge the group is planned from.
    """
    resolve = rate_for if rate_for is not None else (lambda s: s.sample_rate_hz)
    by_rate: dict[int | None, list[tuple[float, float]]] = {}
    for segment in segments:
        by_rate.setdefault(resolve(segment), []).append(
            (float(segment.start_hz), float(segment.end_hz))
        )
    planned = [
        (int(low), int(high), rate)
        for rate, spans in by_rate.items()
        for low, high in merge_ranges(spans)
    ]
    return tuple(sorted(planned))


def find(hz: float, region: str = DEFAULT_REGION) -> Band | None:
    """The most specific band containing `hz`.

    Allocations overlap - Weather Radio sits inside the wider public service
    range, and the 433 MHz gadget band inside the 70 cm amateur band. The
    narrower one is always the more useful answer, so it wins.
    """
    matches = [band for band in load(region) if band.contains(hz)]
    if not matches:
        return None
    return min(matches, key=lambda band: band.width_hz)


def official(hz: float, region: str = DEFAULT_REGION) -> Allocation | None:
    """Who owns `hz`, for a frequency no band covers.

    Independent of `find` on purpose: a band already describes itself, and
    this list only speaks for the space between them. Callers ask it when
    `find` came back empty.
    """
    matches = [a for a in allocations(region) if a.contains(hz)]
    if not matches:
        return None
    return min(matches, key=lambda allocation: allocation.width_hz)


def overlapping(
    low_hz: float, high_hz: float, region: str = DEFAULT_REGION
) -> list[Band]:
    """Every band intersecting a span, for drawing the spectrum ribbon."""
    return [
        band
        for band in load(region)
        if band.end_hz >= low_hz and band.start_hz <= high_hz
    ]


def overlapping_allocations(
    low_hz: float, high_hz: float, region: str = DEFAULT_REGION
) -> list[Allocation]:
    """Every allocation intersecting a span, for the ribbon's empty stretches."""
    return [
        allocation
        for allocation in allocations(region)
        if allocation.end_hz >= low_hz and allocation.start_hz <= high_hz
    ]


__all__ = [
    "Allocation",
    "Band",
    "Channel",
    "DEFAULT_REGION",
    "Segment",
    "allocations",
    "coverage",
    "find",
    "load",
    "merge_ranges",
    "official",
    "overlapping",
    "overlapping_allocations",
    "scannable",
    "sweep_ranges",
]
