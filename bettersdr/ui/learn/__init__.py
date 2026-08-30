"""The Learn tab's content, and the logic for browsing and searching it.

The app explains itself - that is the second of the two principles the whole
project runs on, and until now it was spent entirely on the classifier's
`reasons`. This is the other half: a control that says "Squelch" is only
friendly to somebody who already knows what squelch is, and a beginner-facing
radio that leaves forty such words undefined has quietly capped itself at the
users who did not need it.

So the content is **data**, the same bargain as `scan/bandplan/us.yaml` and
`ui/basemap/us.bsm`: a rewrite for a different audience, or a second language,
is a second file and not a second code path. And the logic around it is pure -
no Qt - for the same reason `ui/results.py` is. A search that ranks badly, or
a cross-reference that points at nothing, looks entirely normal on screen: the
article that should have been first is simply third, and the link that goes
nowhere is a click that does nothing at all. Neither is visible without a test.

Three decisions worth stating, because none of them is the obvious one:

**Nothing here is level-gated.** Every other part of the UI hides what belongs
to a higher level; this part must not. Somebody in Simple mode who has read
the words "RF gain" somewhere and wants to know what they mean is precisely
the reader this exists for, and the fact that they cannot yet *see* that
control is not a reason to withhold the explanation of it. Levels decide what
you can change, never what you can understand.

**A slug is a promise.** Control labels across the app link here by slug, so a
renamed slug is a click that does nothing - the failure mode nobody reports
because it looks like a control that was never clickable. `check()` is the
guard, and `tests/test_learn.py` runs it over the whole file plus every topic
the views name.

**Search has to answer questions, not just match words.** Somebody looking for
"hiss" does not know the article is called "Noise blanker", and somebody who
just read the word "capcode" on the pager panel needs to land on POCSAG. So
every article carries the other names it goes by, and the body is searched as
well as the title - ranked below it, but searched.
"""

from __future__ import annotations

import functools
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LEARN_DIR = Path(__file__).resolve().parent
DEFAULT_EDITION = "glossary"

# `[[slug]]` or `[[slug|the words to show]]`. Deliberately the same spelling as
# a wiki link: it is the one cross-reference syntax a non-programmer editing
# this file is likely to have met before.
LINK = re.compile(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]")

# How search results are ordered. A hit in the title beats a hit in the other
# names it goes by, which beats the one-line summary, which beats the body -
# because the body of a long article mentions a dozen things it is not about.
SCORE_TITLE = 100
SCORE_ALIAS = 60
SCORE_SUMMARY = 30
SCORE_BODY = 10
# Bonuses on top, so "am" finds AM rather than every article containing the
# word "amplifier".
BONUS_EXACT = 200
BONUS_PREFIX = 50


@dataclass(frozen=True)
class Article:
    """One thing the app can explain."""

    slug: str
    title: str
    summary: str
    body: tuple[str, ...] = ()
    # Where the control actually is. A glossary that explains a control
    # without saying where it lives is a crossword clue.
    where: str = ""
    # The other names this goes by. Searched, and shown under the title: the
    # whole problem is that the reader met the word somewhere else.
    also: tuple[str, ...] = ()
    see: tuple[str, ...] = ()
    category: str = ""
    # Everything searchable, lowercased once at load. Search runs on every
    # keystroke and there are seventy-odd of these.
    haystack: tuple[str, str, str, str] = field(
        default=("", "", "", ""), repr=False, compare=False
    )

    @property
    def links(self) -> tuple[str, ...]:
        """Every slug this article points at, from its body and its `see`."""
        found = [match.group(1) for text in self.body for match in LINK.finditer(text)]
        return tuple(dict.fromkeys([*found, *self.see]))


@dataclass(frozen=True)
class Category:
    """A heading on the home page and the articles under it."""

    name: str
    articles: tuple[Article, ...]


def _prose(text: object) -> str:
    """Collapse the whitespace a wrapped YAML block leaves behind."""
    return " ".join(str(text or "").split())


def _haystack(
    title: str, also: tuple[str, ...], summary: str, body: tuple[str, ...]
) -> tuple[str, str, str, str]:
    return (
        title.lower(),
        " ".join(also).lower(),
        summary.lower(),
        # Link markup stripped, so searching for "sideband" does not match an
        # article merely because it links to [[ssb|sideband]] - the display
        # text is kept, the slug is not.
        strip_links(" ".join(body)).lower(),
    )


@functools.lru_cache(maxsize=4)
def load(edition: str = DEFAULT_EDITION) -> tuple[Category, ...]:
    """Every article, in the order the file puts them in.

    File order is the order a beginner meets these things, not the alphabet,
    and that is the whole reason the home page browses by category rather than
    offering an A-Z.
    """
    path = LEARN_DIR / f"{edition}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no learn content named {edition!r} at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories = []
    for entry in raw.get("categories", []):
        name = _prose(entry.get("name", ""))
        articles = []
        for item in entry.get("articles", []):
            title = _prose(item.get("title", ""))
            summary = _prose(item.get("summary", ""))
            also = tuple(_prose(name) for name in item.get("also", []))
            body = tuple(_prose(text) for text in item.get("body", []))
            articles.append(
                Article(
                    slug=str(item["slug"]),
                    title=title,
                    summary=summary,
                    body=body,
                    where=_prose(item.get("where", "")),
                    also=also,
                    see=tuple(str(slug) for slug in item.get("see", [])),
                    category=name,
                    haystack=_haystack(title, also, summary, body),
                )
            )
        categories.append(Category(name=name, articles=tuple(articles)))
    return tuple(categories)


@functools.lru_cache(maxsize=4)
def _index(edition: str = DEFAULT_EDITION) -> dict[str, Article]:
    return {
        article.slug: article
        for category in load(edition)
        for article in category.articles
    }


def articles(edition: str = DEFAULT_EDITION) -> tuple[Article, ...]:
    """Every article, flat, in file order."""
    return tuple(_index(edition).values())


def get(slug: str, edition: str = DEFAULT_EDITION) -> Article | None:
    """The article for a slug, or None.

    Never raises. A control pointing at a topic that has been renamed should
    leave the reader on the home page, not take the window down with it - and
    `check()` plus the test suite is where that mistake is meant to be caught,
    not at the click.
    """
    return _index(edition).get(slug)


def has(slug: str, edition: str = DEFAULT_EDITION) -> bool:
    """Whether anything explains `slug`.

    Asked before a label is made clickable. A control that looks like a link
    and does nothing is worse than one that never offered, so the affordance
    only appears where there is something behind it.
    """
    return slug in _index(edition)


# -- links ----------------------------------------------------------------


def strip_links(text: str) -> str:
    """`[[slug|words]]` -> `words`, for plain-text uses like search."""
    return LINK.sub(lambda m: m.group(2) or m.group(1), text)


def to_html(text: str, edition: str = DEFAULT_EDITION) -> str:
    """One paragraph, as rich text with its cross-references as anchors.

    HTML rather than Qt, so this stays testable without a window - Qt's rich
    text is a subset of HTML and a `QLabel` emits `linkActivated` with the
    href, which is all the view needs.

    A link whose slug does not exist renders as **plain text**, not as a dead
    anchor. That is the same bargain as `has()`: nothing in the app may look
    clickable and then do nothing. It also means a typo in the content file
    degrades to a slightly odd sentence rather than to a trap.
    """
    index = _index(edition)

    def render(match: re.Match[str]) -> str:
        slug, shown = match.group(1), match.group(2)
        article = index.get(slug)
        if article is None:
            return html.escape(shown or slug)
        label = html.escape(shown or article.title)
        return f'<a href="{html.escape(slug)}">{label}</a>'

    # Escaped first, then the anchors put in, so an ampersand in the prose
    # cannot become markup and a `<` cannot eat the rest of the paragraph.
    escaped = html.escape(text)
    # Escaping turned the link markup's own characters into entities only if
    # they were special; the brackets and pipe survive it untouched, so the
    # pattern still matches.
    return LINK.sub(render, escaped)


# -- searching -------------------------------------------------------------


def _score(article: Article, term: str) -> int:
    """How well one article answers one word. 0 means it does not."""
    title, also, summary, body = article.haystack
    score = 0
    if term in title:
        score += SCORE_TITLE
    if term in also:
        score += SCORE_ALIAS
    if term in summary:
        score += SCORE_SUMMARY
    if term in body:
        score += SCORE_BODY
    return score


def _bonus(article: Article, query: str) -> int:
    """Extra for an article that *is* what was typed, rather than mentioning it.

    Measured against the whole query, never against one word of it, and that
    is a bug fixed rather than a preference. Awarding it per word let one
    article's slug being "stereo" beat the article actually called "Stereo
    blend" on the query "stereo blend" - the first collected an exact-match
    bonus for half of what was typed and a passing mention of the other half.
    An exact match is a claim about what somebody asked for, so it has to be
    judged against all of it.
    """
    title, _also, _summary, _body = article.haystack
    names = [name.lower() for name in article.also]
    if article.slug == query or title == query or query in names:
        return BONUS_EXACT
    if title.startswith(query) or any(name.startswith(query) for name in names):
        return BONUS_PREFIX
    return 0


def search(query: str, edition: str = DEFAULT_EDITION) -> tuple[Article, ...]:
    """Articles answering `query`, best first.

    Every word has to match, which is what makes a two-word query narrower
    than a one-word one rather than wider - the alternative ranks a long
    article that mentions one of the words above the short one that is
    actually about both.

    Unless that finds nothing, in which case any word will do. People type
    questions into search boxes - "why is my audio quiet" - and requiring all
    of "why", "is" and "my" to appear answers a perfectly good question with a
    blank page. Falling back is the difference between a search box and a
    lookup table.

    An empty query returns nothing rather than everything: the caller shows
    the browsable home page in that case, and returning all seventy articles
    as a flat "result" would look like a search that had failed to narrow.
    """
    terms = [word for word in query.lower().split() if word]
    if not terms:
        return ()
    # Normalised, so "  Stereo   Blend " is the same question as "stereo
    # blend" as far as the exact-match bonus is concerned.
    whole = " ".join(terms)
    return _ranked(terms, whole, edition, require_all=True) or _ranked(
        terms, whole, edition, require_all=False
    )


def _ranked(
    terms: list[str], whole: str, edition: str, require_all: bool
) -> tuple[Article, ...]:
    scored: list[tuple[int, int, Article]] = []
    for position, article in enumerate(articles(edition)):
        total = 0
        for term in terms:
            hit = _score(article, term)
            if not hit and require_all:
                total = 0
                break
            total += hit
        if total:
            total += _bonus(article, whole)
            # File position as the tiebreak, so equally good answers hold
            # still between keystrokes instead of swapping under the cursor.
            scored.append((-total, position, article))
    return tuple(article for _, _, article in sorted(scored, key=lambda s: s[:2]))


# -- integrity -------------------------------------------------------------


def check(edition: str = DEFAULT_EDITION) -> tuple[str, ...]:
    """Everything wrong with the content file, as readable complaints.

    Run by the test suite rather than at startup. All of these are mistakes
    that look fine on screen - a duplicate slug silently shadows an article, a
    dead cross-reference is a link that renders as plain prose and is simply
    never noticed, a missing summary is a blank line in the browse list.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for category in load(edition):
        if not category.name:
            problems.append("a category has no name")
        for article in category.articles:
            where = f"{article.slug!r}"
            if article.slug in seen:
                problems.append(f"{where}: duplicate slug")
            seen.add(article.slug)
            if not article.title:
                problems.append(f"{where}: no title")
            if not article.summary:
                problems.append(f"{where}: no summary")
            if not article.body:
                problems.append(f"{where}: no body")
            for slug in article.links:
                if not has(slug, edition):
                    problems.append(f"{where}: links to unknown topic {slug!r}")
                elif slug == article.slug:
                    problems.append(f"{where}: links to itself")
    return tuple(problems)


__all__ = [
    "DEFAULT_EDITION",
    "Article",
    "Category",
    "articles",
    "check",
    "get",
    "has",
    "load",
    "search",
    "strip_links",
    "to_html",
]
