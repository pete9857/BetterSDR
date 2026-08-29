"""Persisted user settings.

Small and deliberately dumb: a JSON file of flat keys, read once at startup
and written when something changes. There is no schema and no migration
machinery, because the failure mode we actually care about is not a missing
key - it is a settings file that stops the app opening at all.

So every read has a default, an unreadable or corrupt file is silently
replaced by defaults, and a failed write is swallowed. Losing a remembered
window position is a shrug; refusing to start a radio because a JSON file has
a stray brace in it is not something a beginner can recover from.

Writes are atomic - a temporary file in the same directory, then a replace -
so a crash or a power cut mid-save leaves the previous settings intact rather
than a half-written file that will not parse.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

APP_NAME = "BetterSDR"


def config_dir() -> Path:
    """Where this platform expects an application to keep its settings."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / APP_NAME.lower()


def _open_temporary(path: Path):
    """A scratch file beside `path`, for the write-then-replace dance.

    Deliberately not a context manager at this level: the caller opens it,
    writes into it, closes it, and only then replaces the real file - so the
    replace happens after the handle is definitely flushed. Shared with
    `bookmarks.py`, which saves the same way for the same reason.
    """
    return tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed by the caller
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    )


DEFAULTS: dict[str, Any] = {
    # Progressive disclosure. Remembered per user because someone who has
    # reached Expert should not be met by Simple every morning.
    "level": "standard",
    "frequency_hz": 98_500_000,
    "mode": "wfm",
    "volume": 0.5,
    "audio_device": None,
    "ppm": 0,
    # Costs 2.4% of a core and only runs on broadcast FM, so the
    # default is on; it is remembered because somebody who turned it
    # off did so for a reason.
    "rds": True,
    # The same bargain on a two-way channel: 1.3% of a core, and it only
    # attaches on a channel narrow enough to be a pager one in the first
    # place. Somebody who tunes across a paging transmitter has no way of
    # knowing there was a switch to go and find.
    "pocsag": True,
    "stereo": True,
    # Fading a fringe station towards mono is a judgement about how
    # much hiss is worth how much separation, so it is remembered
    # separately from whether stereo is decoded at all.
    "stereo_blend": True,
    # Off by default: an HD session takes a few seconds to find the digital
    # signal, and a first-time user pressing play should hear something
    # immediately. Remembered because somebody who turned it on wants it on
    # every station that carries it, not on the one they pressed it on.
    "hd": False,
    # Display. These are the SDR# parity controls; all are safe to restore
    # because none of them can put the radio into a state that needs rescuing.
    "fft_size": 4096,
    "fft_window": "hann",
    "fft_smoothing": 0.0,
    "colour_map": "classic",
    "range_floor_db": -90.0,
    "range_ceiling_db": -20.0,
    "peak_hold": True,
    "waterfall_speed": 1,
    "split_ratio": 0.4,
    # Deliberately not remembered as "on": the bias tee puts 4.5 V on the
    # antenna port, and a setting that survives a restart could damage
    # equipment plugged in by someone who never turned it on themselves.
    "bias_tee": False,
    "recording_dir": None,
}


class Settings:
    """A flat key-value store backed by one JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config_dir() / "settings.json"
        self._values: dict[str, Any] = dict(DEFAULTS)
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> Settings:
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing, unreadable or corrupt. Defaults are a fine answer and
            # the next save will replace the file.
            return self
        if isinstance(stored, dict):
            self._values.update(
                {key: value for key, value in stored.items() if key in DEFAULTS}
            )
        return self

    def save(self) -> bool:
        """Write atomically. Returns whether it worked, for callers that care."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = _open_temporary(self.path)
            with temporary:
                json.dump(self._values, temporary, indent=2, sort_keys=True)
            os.replace(temporary.name, self.path)
            return True
        except OSError:
            return False

    # -- access ------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._values.get(key, DEFAULTS.get(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, DEFAULTS.get(key, default))

    def update(self, **values: Any) -> Settings:
        self._values.update(values)
        return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def reset(self) -> Settings:
        self._values = dict(DEFAULTS)
        return self


__all__ = ["APP_NAME", "DEFAULTS", "Settings", "config_dir"]
