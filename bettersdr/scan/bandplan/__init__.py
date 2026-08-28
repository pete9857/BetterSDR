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
from dataclasses import dataclass
from pathlib import Path

import yaml

BANDPLAN_DIR = Path(__file__).resolve().parent
DEFAULT_REGION = "us"


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
        turns a rough click on the spectrum into an actual station rather than
        something 40 kHz off it.
        """
        if not self.raster_hz:
            return hz
        base = self.start_hz + self.raster_hz / 2.0
        return base + round((hz - base) / self.raster_hz) * self.raster_hz


@functools.lru_cache(maxsize=4)
def load(region: str = DEFAULT_REGION) -> tuple[Band, ...]:
    """Every band for a region, ordered by frequency."""
    path = BANDPLAN_DIR / f"{region}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no band plan for region {region!r} at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bands = [
        Band(
            name=entry["name"],
            start_hz=int(entry["start_hz"]),
            end_hz=int(entry["end_hz"]),
            mode=entry.get("mode", "nfm"),
            bandwidth_hz=float(entry.get("bandwidth_hz", 12_500)),
            description=" ".join(entry.get("description", "").split()),
            colour=entry.get("colour", "#7a7a7a"),
            icon=entry.get("icon", ""),
            raster_hz=(
                float(entry["raster_hz"]) if entry.get("raster_hz") else None
            ),
        )
        for entry in raw.get("bands", [])
    ]
    return tuple(sorted(bands, key=lambda band: band.start_hz))


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


def overlapping(
    low_hz: float, high_hz: float, region: str = DEFAULT_REGION
) -> list[Band]:
    """Every band intersecting a span, for drawing the spectrum ribbon."""
    return [
        band
        for band in load(region)
        if band.end_hz >= low_hz and band.start_hz <= high_hz
    ]


__all__ = ["Band", "DEFAULT_REGION", "find", "load", "overlapping"]
