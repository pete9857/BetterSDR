"""Ordering and filtering the list of found signals.

A sweep of the FM band comes back with a dozen stations and needs neither of
these. A sweep of the airband on an indoor aerial comes back with 83 stable
carriers from switching supplies and the dongle's own clock, with four real
channels somewhere underneath them. Both answers are honest - the app reports
what is actually radiating - but the second one is unreadable, and a list
nobody can read is the same failure as a list that is wrong.

So: an order the user chooses, and a way to put a whole *kind* of thing out of
sight. Hiding is by the classifier's own label, which means the filter is
worded in exactly the same plain English as the cards it is hiding, and the
chip keeps saying how many it is holding back - nothing is ever silently
dropped from a count.

Pure functions, no Qt, for the same reason the colour maps and the map trails
are: a sort that quietly is not stable, or a filter that hides the thing it
just counted, looks entirely normal on screen for the first minute.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..scan.classifier import Signal

# The orders offered, as (what the user reads, what is stored). Worded as an
# answer to "what should be at the top", not as a column name.
SORTS: tuple[tuple[str, str], ...] = (
    ("Strongest first", "strength"),
    ("By frequency", "frequency"),
    ("Grouped by type", "kind"),
)

DEFAULT_SORT = "strength"

SORT_KEYS: frozenset[str] = frozenset(key for _, key in SORTS)


def sort_signals(
    signals: Iterable[Signal], order: str = DEFAULT_SORT
) -> tuple[Signal, ...]:
    """Order the list the way the user asked for.

    An unrecognised order is the strongest-first default rather than an error:
    this is a display preference restored from a settings file, and the same
    bargain applies as everywhere else in `settings.py` - a stale key is a
    shrug, not something a beginner has to recover from.

    Every order is fully determined, with frequency as the final tiebreak, so
    two equally strong signals cannot swap places between passes and slam a
    card's "What is this?" shut under the cursor.
    """
    items = list(signals)
    if order == "frequency":
        return tuple(sorted(items, key=lambda s: s.frequency_hz))
    if order == "kind":
        # Groups ordered by their strongest member, so grouping by type does
        # not bury the real stations under whatever kind happens to be the
        # most numerous - which in the airband is the interference.
        best: dict[str, float] = {}
        for signal in items:
            if signal.snr_db > best.get(signal.label, float("-inf")):
                best[signal.label] = signal.snr_db
        return tuple(
            sorted(
                items,
                key=lambda s: (-best[s.label], s.label, s.frequency_hz),
            )
        )
    # Strongest first, the way a Wi-Fi picker orders networks.
    return tuple(sorted(items, key=lambda s: (-s.snr_db, s.frequency_hz)))


def visible(
    signals: Iterable[Signal], hidden: Iterable[str] = ()
) -> tuple[Signal, ...]:
    """The signals to actually put on screen, in the order given."""
    excluded = frozenset(hidden)
    if not excluded:
        return tuple(signals)
    return tuple(signal for signal in signals if signal.label not in excluded)


@dataclass(frozen=True)
class Kind:
    """One classification present in the results, and how many wear it."""

    label: str
    icon: str
    count: int
    shown: bool

    @property
    def chip(self) -> str:
        """The chip's text: what it is, and how many of it there are.

        The count is on the chip whether the kind is showing or hidden,
        because "Unmodulated carrier (83)" sitting there greyed out is the
        difference between a filter and a sweep that missed something.
        """
        return f"{self.label} ({self.count})"


def summarise(
    signals: Sequence[Signal], hidden: Iterable[str] = ()
) -> tuple[Kind, ...]:
    """What kinds of thing this sweep found, most numerous first.

    Most numerous first because the chip worth reaching for is the one holding
    the pile: it is the eighty-three carriers that make the list unreadable,
    not the four aircraft channels.
    """
    excluded = frozenset(hidden)
    counts: dict[str, int] = {}
    icons: dict[str, str] = {}
    for signal in signals:
        counts[signal.label] = counts.get(signal.label, 0) + 1
        icons.setdefault(signal.label, signal.icon)
    return tuple(
        Kind(
            label=label,
            icon=icons[label],
            count=count,
            shown=label not in excluded,
        )
        # Label as the tiebreak so equal counts hold still between passes.
        for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


# How close the dial has to be to a found signal before the step buttons
# consider themselves to be *on* it. The same 5 kHz the bookmark store uses to
# decide two saved frequencies are the same station: a scan reports the centre
# it measured, and clicking a card leaves the radio a few hundred Hz off it.
MATCH_TOLERANCE_HZ = 5_000.0


def neighbour(
    signals: Sequence[Signal],
    frequency_hz: float,
    delta: int,
    tolerance_hz: float = MATCH_TOLERANCE_HZ,
) -> Signal | None:
    """The next or previous entry of a list, from wherever the dial is now.

    `signals` is the list *as displayed* - already ordered and already
    filtered - because these buttons are a way of walking the screen the user
    is looking at, not the sweep behind it. Hiding a kind therefore skips it
    here too, which is the only reading that could not surprise anybody.

    Wraps, so the end of the list is the start of it again: a car radio's seek
    button does, and a list that stopped dead would need a second control to
    say why. A frequency that is not in the list at all is not an error - the
    dial goes where it likes - so stepping forward from one enters at the top
    and stepping back enters at the bottom.
    """
    if not signals or delta == 0:
        return None
    here = -1
    closest = float("inf")
    for index, signal in enumerate(signals):
        distance = abs(signal.frequency_hz - frequency_hz)
        # Strictly closer, so a tie holds the earlier entry rather than
        # letting two overlapping cards swap which one Next steps off.
        if distance <= tolerance_hz and distance < closest:
            here, closest = index, distance
    if here < 0:
        return signals[0] if delta > 0 else signals[-1]
    return signals[(here + delta) % len(signals)]


def hidden_count(signals: Sequence[Signal], hidden: Iterable[str] = ()) -> int:
    """How many of these the filter is holding back."""
    excluded = frozenset(hidden)
    return sum(1 for signal in signals if signal.label in excluded)


__all__ = [
    "DEFAULT_SORT",
    "MATCH_TOLERANCE_HZ",
    "SORTS",
    "SORT_KEYS",
    "Kind",
    "hidden_count",
    "neighbour",
    "sort_signals",
    "summarise",
    "visible",
]
