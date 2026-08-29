"""Tests for recently played, favourites and the session trail.

Every test drives the clock by hand. Real time would make the dwell threshold
either a four-second test or a flaky one, and the arithmetic being checked -
what counts as listening and what is just passing through - is exactly the
part that has to be exact.
"""

from __future__ import annotations

import json

from bettersdr.core.bookmarks import Bookmark, BookmarkStore
from bettersdr.core.history import DWELL_SECONDS, MAX_TICK_SECONDS, History

KUOW = 94_900_000
KING = 98_100_000
KNKX = 88_500_000


def listen(history: History, hz: int, seconds: float, start: float = 1000.0) -> float:
    """Tune to `hz` and stay there, ticking the way a view's timer would."""
    history.tune(hz, now=start)
    now = start
    while now < start + seconds:
        now = min(now + 1.0, start + seconds)
        history.update(now)
    return now


# -- what counts as listening ----------------------------------------------


def test_passing_through_is_not_listening():
    """The digit tuner emits a frequency per keystroke. None of them count."""
    history = History()
    now = 0.0
    for hz in range(88_100_000, 89_100_000, 100_000):
        history.tune(hz, now=now)
        now += 0.4
        history.update(now)
    assert history.recent() == ()
    assert history.previous() is None


def test_a_station_listened_to_is_recorded():
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 5.0)
    recent = history.recent()
    assert len(recent) == 1
    assert recent[0].frequency_hz == KUOW
    assert recent[0].plays == 1
    assert recent[0].seconds >= DWELL_SECONDS


def test_the_threshold_is_the_whole_gate():
    """One second short is not recorded; one second past it is."""
    short = History()
    listen(short, KUOW, DWELL_SECONDS - 1.0)
    assert short.recent() == ()

    long = History()
    listen(long, KUOW, DWELL_SECONDS + 1.0)
    assert len(long.recent()) == 1


def test_nudging_the_dial_continues_the_visit():
    """A retune inside the match tolerance is the same station.

    Without this a station could be listened to all evening and never be
    recorded, because every small correction would restart the dwell timer.
    """
    history = History()
    history.tune(KUOW, now=0.0)
    now = 0.0
    while now < DWELL_SECONDS + 2.0:
        now += 1.0
        history.update(now)
        # A click landing a couple of bins off, over and over.
        history.tune(KUOW + 900, now=now)
    assert len(history.recent()) == 1
    assert history.recent()[0].plays == 1


def test_time_away_from_the_screen_is_not_counted():
    """A view stops when its page is hidden; the gap must not be credited."""
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 2.0)
    heard = history.recent()[0].seconds
    # An hour behind the Discover screen, then one more tick.
    history.update(1000.0 + DWELL_SECONDS + 2.0 + 3600.0)
    assert history.recent()[0].seconds - heard < MAX_TICK_SECONDS


# -- names -----------------------------------------------------------------


def test_a_name_arriving_late_still_names_the_entry():
    """RDS takes seconds to confirm a name, which is why the dwell is longer."""
    history = History()
    history.tune(KUOW, now=0.0)
    now = 0.0
    while now < DWELL_SECONDS + 6.0:
        now += 1.0
        history.update(now)
        if now > DWELL_SECONDS + 2.0:
            history.name("KUOW")
    assert history.recent()[0].name == "KUOW"


def test_an_empty_name_never_erases_one():
    """A decoder that has lost the signal reports nothing, not a blank name."""
    history = History()
    history.tune(KUOW, now=0.0)
    history.name("KUOW")
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=0.0)
    history.name("")
    history.name("   ")
    assert history.recent()[0].name == "KUOW"


# -- ordering and the trail ------------------------------------------------


def test_recent_is_ordered_by_when_it_was_last_heard():
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=0.0)
    listen(history, KING, DWELL_SECONDS + 1.0, start=100.0)
    listen(history, KNKX, DWELL_SECONDS + 1.0, start=200.0)
    assert [s.frequency_hz for s in history.recent()] == [KNKX, KING, KUOW]


def test_most_played_is_a_different_order():
    history = History()
    listen(history, KUOW, 120.0, start=0.0)
    listen(history, KING, DWELL_SECONDS + 1.0, start=500.0)
    assert history.recent()[0].frequency_hz == KING
    assert history.most_played()[0].frequency_hz == KUOW


def test_coming_back_counts_as_another_play():
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=0.0)
    listen(history, KING, DWELL_SECONDS + 1.0, start=100.0)
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=200.0)
    entry = history.find(KUOW)
    assert entry is not None
    assert entry.plays == 2
    assert len(history.recent()) == 2


def test_back_goes_where_the_radio_just_was():
    """A, B, A: Back goes to B, and the repeats are kept in the trail."""
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=0.0)
    listen(history, KING, DWELL_SECONDS + 1.0, start=100.0)
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=200.0)
    previous = history.previous()
    assert previous is not None
    assert previous.frequency_hz == KING


def test_back_ignores_stations_only_passed_through():
    history = History()
    listen(history, KUOW, DWELL_SECONDS + 1.0, start=0.0)
    history.tune(KING, now=100.0)
    history.update(100.5)
    listen(history, KNKX, DWELL_SECONDS + 1.0, start=200.0)
    previous = history.previous()
    assert previous is not None
    assert previous.frequency_hz == KUOW


def test_the_list_is_capped_and_drops_the_oldest():
    history = History(max_entries=3)
    for index in range(5):
        listen(
            history,
            88_100_000 + index * 1_000_000,
            DWELL_SECONDS + 1.0,
            start=index * 100.0,
        )
    assert len(history.recent()) == 3
    assert history.find(88_100_000) is None
    assert history.find(92_100_000) is not None


# -- persistence -----------------------------------------------------------


def test_round_trip(tmp_path):
    path = tmp_path / "history.json"
    history = History(path)
    listen(history, KUOW, DWELL_SECONDS + 1.0)
    history.name("KUOW")
    history.leave()
    assert history.save()

    reopened = History.open(path)
    assert len(reopened.recent()) == 1
    assert reopened.recent()[0].name == "KUOW"
    # The trail is this session's, so it does not come back with the file.
    assert reopened.previous() is None


def test_a_corrupt_file_is_an_empty_history(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{ not json", encoding="utf-8")
    assert History.open(path).recent() == ()


def test_a_file_from_a_later_version_loses_only_what_it_gained(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps([{"frequency_hz": KUOW, "name": "KUOW", "warp_drive": True}]),
        encoding="utf-8",
    )
    entries = History.open(path).recent()
    assert len(entries) == 1
    assert entries[0].name == "KUOW"


# -- favourites ------------------------------------------------------------


def test_favourites_survive_a_csv_round_trip(tmp_path):
    store = BookmarkStore(tmp_path / "bookmarks.json")
    store.add(Bookmark("KUOW", KUOW, favourite=True))
    store.add(Bookmark("KING", KING))
    reloaded = BookmarkStore(tmp_path / "other.json")
    reloaded.from_csv(store.to_csv())
    assert [e.name for e in reloaded.favourites] == ["KUOW"]


def test_toggling_a_favourite_replaces_the_entry(tmp_path):
    """`Bookmark` is frozen, so the store must not end up holding both."""
    store = BookmarkStore(tmp_path / "bookmarks.json")
    entry = store.add(Bookmark("KUOW", KUOW))
    store.toggle_favourite(entry)
    assert len(store) == 1
    assert store.favourites[0].name == "KUOW"
    store.toggle_favourite(store.entries[0])
    assert store.favourites == []


def test_an_old_bookmark_file_loads_without_the_field(tmp_path):
    path = tmp_path / "bookmarks.json"
    path.write_text(
        json.dumps([{"name": "KUOW", "frequency_hz": KUOW, "mode": "wfm"}]),
        encoding="utf-8",
    )
    store = BookmarkStore.open(path)
    assert len(store) == 1
    assert store.entries[0].favourite is False
