"""The saved songs: naming, the five-copies rule, and its memory.

Nothing here touches audio. A `SongLibrary` is a folder, an index and a
decision, and the decision is the one thing in Repro-Radio that has to keep
being right across restarts.
"""

from __future__ import annotations

import json

import pytest

from bettersdr.audio.library import MAX_COPIES, SongLibrary, safe_stem
from bettersdr.decode.songtag import SongTag


def tag(artist: str = "Rush", title: str = "Tom Sawyer") -> SongTag:
    return SongTag(artist=artist, title=title)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("AC/DC - Back In Black", "AC-DC - Back In Black"),
        ('Blue "Da Ba Dee"', "Blue -Da Ba Dee-"),
        ("What?", "What-"),
        ("9:00", "9-00"),
        ("Trailing dots...", "Trailing dots"),
        ("Trailing space ", "Trailing space"),
        ("", "Unknown"),
        ("   ", "Unknown"),
    ],
)
def test_a_name_windows_will_take(name, expected):
    assert safe_stem(name) == expected


def test_a_reserved_device_name_is_escaped():
    """`CON.mp3` is the console, whatever the extension says."""
    assert safe_stem("CON") == "_CON"
    assert safe_stem("con.mp3") == "_con.mp3"


def test_a_very_long_title_is_cut_but_still_ends_cleanly():
    stem = safe_stem("A" * 300)
    assert len(stem) <= 120
    assert not stem.endswith((".", " "))


def test_the_first_copy_is_unnumbered(tmp_path):
    library = SongLibrary(tmp_path)
    placement = library.plan(tag())
    assert placement.wanted
    assert placement.path.name == "Rush - Tom Sawyer.mp3"
    assert placement.copy == 1


def test_copies_are_numbered_and_then_refused(tmp_path):
    library = SongLibrary(tmp_path)
    names = []
    for _ in range(MAX_COPIES):
        placement = library.plan(tag())
        assert placement.wanted, placement.reason
        placement.path.parent.mkdir(parents=True, exist_ok=True)
        placement.path.write_bytes(b"x")
        library.remember(tag(), placement.path)
        names.append(placement.path.name)

    assert names == [
        "Rush - Tom Sawyer.mp3",
        "Rush - Tom Sawyer (2).mp3",
        "Rush - Tom Sawyer (3).mp3",
        "Rush - Tom Sawyer (4).mp3",
        "Rush - Tom Sawyer (5).mp3",
    ]
    refused = library.plan(tag())
    assert not refused.wanted
    assert "5 recordings" in refused.reason


def test_deleting_copies_lets_more_be_recorded(tmp_path):
    """Pruning a folder by hand is exactly what this feature expects.

    Counting from the index alone would mean a song somebody had cut down to
    their favourite take was never recorded again - and it would look like the
    station simply not playing it.
    """
    library = SongLibrary(tmp_path)
    for _ in range(MAX_COPIES):
        placement = library.plan(tag())
        placement.path.parent.mkdir(parents=True, exist_ok=True)
        placement.path.write_bytes(b"x")
        library.remember(tag(), placement.path)
    assert not library.plan(tag()).wanted

    (tmp_path / "Rush - Tom Sawyer (3).mp3").unlink()
    again = library.plan(tag())
    assert again.wanted
    assert again.path.name == "Rush - Tom Sawyer (3).mp3"


def test_the_index_survives_a_restart(tmp_path):
    first = SongLibrary(tmp_path)
    placement = first.plan(tag())
    placement.path.parent.mkdir(parents=True, exist_ok=True)
    placement.path.write_bytes(b"x")
    first.remember(tag(), placement.path)

    second = SongLibrary(tmp_path).load()
    assert second.known("Rush", "Tom Sawyer") == 1
    assert second.plan(tag()).path.name == "Rush - Tom Sawyer (2).mp3"


def test_a_corrupt_index_is_an_empty_one(tmp_path):
    """Same rule as the settings file: replaced, never reported.

    This is called on the DSP thread. Raising here would end an unattended
    session over a metadata problem.
    """
    library = SongLibrary(tmp_path)
    library.index_path.parent.mkdir(parents=True, exist_ok=True)
    library.index_path.write_text("{not json at all", encoding="utf-8")
    assert library.load().song_count == 0
    assert library.plan(tag()).wanted


def test_an_index_naming_files_that_are_gone_is_tidied(tmp_path):
    library = SongLibrary(tmp_path)
    placement = library.plan(tag())
    placement.path.parent.mkdir(parents=True, exist_ok=True)
    placement.path.write_bytes(b"x")
    library.remember(tag(), placement.path)
    placement.path.unlink()

    assert library.forget_missing() == 1
    assert library.song_count == 0
    stored = json.loads(library.index_path.read_text(encoding="utf-8"))
    assert stored["songs"] == {}


def test_a_song_with_no_name_is_never_saved(tmp_path):
    library = SongLibrary(tmp_path)
    refused = library.plan(SongTag(artist="", title="Dreams"))
    assert not refused.wanted
    assert "did not name" in refused.reason


def test_a_file_already_there_is_never_overwritten(tmp_path):
    """The index can be behind the folder in both directions, and two files
    with one name is the outcome that loses a recording outright."""
    library = SongLibrary(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "Rush - Tom Sawyer.mp3").write_bytes(b"already here")
    placement = library.plan(tag())
    assert placement.path.name == "Rush - Tom Sawyer (2).mp3"
