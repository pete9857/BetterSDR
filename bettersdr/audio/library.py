"""The saved songs: what to call a file, and when to stop making more.

Recording songs off the radio produces duplicates on purpose. The same track
comes round two or three times a day, and the copies are not equivalent - the
DJ talks over the end of one, the news cuts into another, the third is clean.
Nobody can know which is the good one at the moment it is recorded, so the
rule is to keep a few and let the listener throw away the rest:

* **Up to five copies of a song**, numbered. `Dreams.mp3`, `Dreams (2).mp3`,
  and so on.
* **After five, nothing.** A station in heavy rotation would otherwise fill a
  disk with the same three minutes of music, and the sixth copy is not adding
  a choice anybody wanted.

The count is checked against the *filesystem*, not only against the index.
Deleting four copies and keeping the best one is exactly what this feature
expects somebody to do, and if the count did not notice, that song would
never be recorded again - which is the failure nobody would report, because
it looks like the song simply not being played.

The index lives in the songs folder rather than in the settings directory, and
that is deliberate. It means moving or copying the music folder takes its
memory with it, and deleting the folder resets the feature completely, which
is the behaviour somebody would guess at without being told.

No Qt and no radio in here: a tag and a folder in, a path or a refusal out.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..decode.songtag import SongTag, key_for

# How many recordings of one song to keep before refusing further ones.
MAX_COPIES = 5
INDEX_NAME = ".bettersdr-songs.json"
INDEX_VERSION = 1

# Windows will not accept these in a name, and a song title contains most of
# them sooner or later: `AC/DC`, `Question?`, `9:00`.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# ...and these are device names whatever the extension says.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{n}" for n in range(1, 10)),
    *(f"lpt{n}" for n in range(1, 10)),
}
# Long enough for any real "Artist - Title", short enough that the numbered
# suffix and the extension still fit inside the path limit.
MAX_STEM = 120


def safe_stem(name: str) -> str:
    """A filename Windows will accept, that still reads as the song.

    Illegal characters become a hyphen rather than vanishing, because
    `AC-DC - Back In Black` is recognisable and `ACDC` looks like a typo.
    Trailing dots and spaces are removed outright: Windows silently strips
    them when creating the file and then cannot find it again by the name it
    was asked for, which turns a cosmetic problem into a missing recording.
    """
    cleaned = _ILLEGAL.sub("-", str(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).rstrip(". ")
    if not cleaned:
        return "Unknown"
    if cleaned.split(".")[0].casefold() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:MAX_STEM].rstrip(". ") or "Unknown"


@dataclass(frozen=True)
class Placement:
    """Where a recording of this song should go, or why it should not.

    `path` of None is a decision, not a failure - it is the five-copies rule
    doing its job - so it carries a reason the status line can show.
    """

    path: Path | None
    copy: int = 0
    reason: str = ""

    @property
    def wanted(self) -> bool:
        return self.path is not None


class SongLibrary:
    """The folder of saved songs and its memory of what is already in it."""

    def __init__(
        self,
        folder: str | Path,
        max_copies: int = MAX_COPIES,
        extension: str = "mp3",
    ) -> None:
        self.folder = Path(folder)
        self.max_copies = max(1, int(max_copies))
        self.extension = extension.lstrip(".")
        self._songs: dict[str, dict] = {}
        self._loaded = False

    # -- persistence -------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.folder / INDEX_NAME

    def load(self) -> SongLibrary:
        """Read the index. A corrupt or missing one is an empty one.

        Same rule as `core/settings.py`: a file we cannot read is replaced by
        the default rather than reported. The cost of getting this wrong is
        that a song is recorded a sixth time, which nobody will notice; the
        cost of raising here is an unattended recording session that stops.
        """
        self._loaded = True
        try:
            stored = json.loads(self.index_path.read_text(encoding="utf-8"))
            songs = stored["songs"]
        except Exception:  # noqa: BLE001 - see the docstring
            self._songs = {}
            return self
        if isinstance(songs, dict):
            self._songs = {
                str(key): value
                for key, value in songs.items()
                if isinstance(value, dict)
            }
        return self

    def save(self) -> None:
        """Write the index atomically, and never fatally.

        Losing the index costs a few duplicate recordings. Raising out of here
        would cost the session, and this is called on the DSP thread.
        """
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
                "w",
                encoding="utf-8",
                dir=self.folder,
                prefix=INDEX_NAME,
                suffix=".tmp",
                delete=False,
            )
            with handle:
                json.dump(
                    {"version": INDEX_VERSION, "songs": self._songs},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(handle.name, self.index_path)
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    # -- what is already here ---------------------------------------------

    def copies(self, tag: SongTag) -> tuple[Path, ...]:
        """The files of this song that are still on the disk, in order.

        Filtered against the filesystem every time rather than trusted from
        the index, so pruning a folder by hand is immediately reflected in
        what may be recorded next.
        """
        self._ensure()
        entry = self._songs.get(tag.key)
        if not entry:
            return ()
        here: list[Path] = []
        for name in entry.get("files", []):
            path = self.folder / str(name)
            if path.exists():
                here.append(path)
        return tuple(here)

    def plan(self, tag: SongTag) -> Placement:
        """Where the next recording of this song goes, or why it does not."""
        self._ensure()
        if not tag.artist.strip() or not tag.title.strip():
            return Placement(None, reason="The station did not name the song.")
        held = self.copies(tag)
        if len(held) >= self.max_copies:
            return Placement(
                None,
                copy=len(held),
                reason=(
                    f"Already have {len(held)} recordings of "
                    f"{tag.display} - keeping those."
                ),
            )
        stem = safe_stem(tag.display)
        # The number is found by looking, not by counting: the index can be
        # behind the folder in both directions, and two files with the same
        # name is the one outcome that loses a recording outright.
        for number in range(1, self.max_copies + 1):
            suffix = "" if number == 1 else f" ({number})"
            candidate = self.folder / f"{stem}{suffix}.{self.extension}"
            if not candidate.exists():
                return Placement(
                    candidate,
                    copy=len(held) + 1,
                    reason=f"Copy {len(held) + 1} of {self.max_copies}.",
                )
        return Placement(
            None,
            copy=len(held),
            reason=f"Already have {self.max_copies} recordings of {tag.display}.",
        )

    # -- writing back ------------------------------------------------------

    def remember(self, tag: SongTag, path: str | Path, when: str = "") -> None:
        """Record that a file was written, and persist the index."""
        self._ensure()
        entry = self._songs.setdefault(
            tag.key, {"artist": tag.artist, "title": tag.title, "files": []}
        )
        entry["artist"] = tag.artist
        entry["title"] = tag.title
        name = Path(path).name
        files = entry.setdefault("files", [])
        if name not in files:
            files.append(name)
        if when:
            entry["last"] = when
        self.save()

    def forget_missing(self) -> int:
        """Drop index entries whose files are gone. Returns how many."""
        self._ensure()
        removed = 0
        for key in list(self._songs):
            entry = self._songs[key]
            kept = [
                name
                for name in entry.get("files", [])
                if (self.folder / str(name)).exists()
            ]
            removed += len(entry.get("files", [])) - len(kept)
            if kept:
                entry["files"] = kept
            else:
                del self._songs[key]
        if removed:
            self.save()
        return removed

    # -- reporting ---------------------------------------------------------

    @property
    def song_count(self) -> int:
        self._ensure()
        return len(self._songs)

    def known(self, artist: str, title: str) -> int:
        """How many copies of this song are held. Used by the tests."""
        self._ensure()
        entry = self._songs.get(key_for(artist, title))
        return len(entry.get("files", [])) if entry else 0


__all__ = [
    "INDEX_NAME",
    "MAX_COPIES",
    "Placement",
    "SongLibrary",
    "safe_stem",
]
