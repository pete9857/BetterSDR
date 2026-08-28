"""Tests for the waterfall's rolling history.

The whole point of the doubled-buffer ring is that scrolling costs nothing, so
these assert both halves of that claim: the rows come out in the right order
across a wraparound, and the view really is a view rather than a copy. A
history that silently copied would still look correct and would quietly cost
200 MB/s at 30 Hz.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.ui.widgets.waterfall import EMPTY_DB, WaterfallHistory


def rows_of(history: WaterfallHistory) -> list[float]:
    """First column of each row, which identifies the row in these tests."""
    return [float(row[0]) for row in history.image()]


def test_new_history_is_empty_and_correctly_shaped():
    history = WaterfallHistory(rows=8, bins=64)
    image = history.image()

    assert image.shape == (8, 64)
    assert np.all(image == EMPTY_DB)
    assert history.pushed == 0


def test_newest_row_lands_at_the_end():
    """Row order is oldest-first, so the display puts the newest line on top."""
    history = WaterfallHistory(rows=4, bins=2)
    history.push(np.full(2, 7.0))

    assert rows_of(history) == [EMPTY_DB, EMPTY_DB, EMPTY_DB, 7.0]


def test_rows_accumulate_oldest_first():
    history = WaterfallHistory(rows=4, bins=2)
    for value in (1.0, 2.0, 3.0, 4.0):
        history.push(np.full(2, value))

    assert rows_of(history) == [1.0, 2.0, 3.0, 4.0]


def test_wraparound_keeps_the_order_and_drops_the_oldest():
    """The case a naive ring gets wrong: writing past the end of the buffer."""
    history = WaterfallHistory(rows=4, bins=2)
    for value in range(1, 11):
        history.push(np.full(2, float(value)))

    assert rows_of(history) == [7.0, 8.0, 9.0, 10.0]
    assert history.pushed == 10


def test_image_is_a_view_not_a_copy():
    history = WaterfallHistory(rows=16, bins=32)
    history.push(np.zeros(32))
    assert np.shares_memory(history.image(), history._data)


def test_clear_wipes_history_but_keeps_the_shape():
    history = WaterfallHistory(rows=4, bins=2)
    for value in range(5):
        history.push(np.full(2, float(value)))
    history.clear()

    assert np.all(history.image() == EMPTY_DB)
    assert history.image().shape == (4, 2)
    assert history.pushed == 0


def test_a_row_of_a_new_width_resizes_rather_than_raising():
    """FFT size is a live display setting; changing it must not crash."""
    history = WaterfallHistory(rows=4, bins=8)
    history.push(np.full(8, 1.0))
    history.push(np.full(16, 2.0))

    assert history.bins == 16
    assert history.image().shape == (4, 16)
    assert rows_of(history)[-1] == 2.0


def test_resize_to_the_same_shape_keeps_the_history():
    history = WaterfallHistory(rows=4, bins=2)
    history.push(np.full(2, 5.0))
    history.resize(rows=4, bins=2)

    assert rows_of(history)[-1] == 5.0


def test_rejects_a_nonsense_shape():
    with pytest.raises(ValueError, match="must be positive"):
        WaterfallHistory(rows=0, bins=8)
