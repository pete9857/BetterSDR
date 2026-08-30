"""Tests for the Learn tab's content and the logic around it.

Everything this feature can get wrong is invisible on screen, which is why it
is worth a test file of its own:

- a cross-reference to a slug nobody wrote renders as ordinary prose, so the
  sentence still reads and the link is simply never there;
- a control that names a topic with no article behind it is a caption that
  quietly stopped being clickable, and nobody reports a control for *not*
  offering something;
- a search that ranks badly puts the right answer third instead of first,
  which looks exactly like a glossary that does not contain it.

The topic inventory is checked by reading the view source rather than by
building a `ControlPanel`, which would need a Qt application and an `Engine`.
The string in the source is the thing that has to be right, and scanning for
it catches a typo in a row nobody has clicked since it was written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bettersdr.ui import learn
from bettersdr.ui.learn import Article

UI_DIR = Path(__file__).resolve().parent.parent / "bettersdr" / "ui"
TOPIC_ARGUMENT = re.compile(r'topic="([a-z0-9-]+)"')


# -- the content file ------------------------------------------------------


def test_content_is_internally_consistent():
    """Duplicate slugs, dead links, missing summaries - all of it at once."""
    assert learn.check() == ()


def test_every_article_is_reachable_by_browsing():
    """Nothing may exist only as the target of a link.

    An article that no category lists is one only somebody who already knew
    its name could ever find, which is the opposite of the point.
    """
    browsable = {
        article.slug for category in learn.load() for article in category.articles
    }
    assert browsable == {article.slug for article in learn.articles()}


def test_categories_are_ordered_and_named():
    categories = learn.load()
    assert categories
    assert categories[0].name == "Start here"
    assert all(category.name for category in categories)
    assert all(category.articles for category in categories)


def test_summaries_are_one_sentence_and_short():
    """The summary is the browse list's only content, so it has to fit.

    Not a style rule for its own sake: a summary that runs to three lines
    turns a seventy-entry browse list into something nobody scrolls.
    """
    for article in learn.articles():
        summary = learn.strip_links(article.summary)
        assert len(summary) <= 160, article.slug
        assert summary.endswith("."), article.slug


def test_prose_is_folded_not_wrapped():
    """YAML block folding must leave single-spaced prose behind.

    A stray newline inside a paragraph becomes a line break in rich text, so
    this catches an article indented one space too far - which YAML reads as
    a literal continuation and Qt then renders as a ragged paragraph.
    """
    for article in learn.articles():
        for paragraph in article.body:
            assert "\n" not in paragraph, article.slug
            assert "  " not in paragraph, article.slug


# -- links -----------------------------------------------------------------


def test_link_renders_as_the_target_title_by_default():
    html = learn.to_html("See [[squelch]] for more.")
    assert '<a href="squelch">Squelch</a>' in html


def test_link_can_override_the_words_shown():
    html = learn.to_html("This is [[wfm|wide FM]] and nothing else.")
    assert '<a href="wfm">wide FM</a>' in html
    assert "wfm|" not in html


def test_link_to_a_missing_article_is_plain_text_not_a_dead_anchor():
    """Nothing may look clickable and then do nothing.

    A dead anchor is the worst of the three possible behaviours: it survives
    review because the sentence reads correctly, and it fails only under a
    cursor.
    """
    html = learn.to_html("A [[no-such-topic|mystery]] here.")
    assert "<a" not in html
    assert "mystery" in html


def test_markup_in_the_prose_cannot_become_markup_in_the_output():
    html = learn.to_html("Tags like <b> and & are prose, not markup.")
    assert "<b>" not in html
    assert "&lt;b&gt;" in html
    assert "&amp;" in html


def test_strip_links_leaves_readable_prose():
    assert learn.strip_links("A [[wfm|wide FM]] and a [[squelch]].") == (
        "A wide FM and a squelch."
    )


# -- lookup ----------------------------------------------------------------


def test_get_returns_the_article():
    article = learn.get("rf-gain")
    assert isinstance(article, Article)
    assert article.title == "RF gain"
    assert article.category == "The receiver"


def test_get_and_has_never_raise_on_an_unknown_slug():
    assert learn.get("not-a-topic") is None
    assert not learn.has("not-a-topic")
    assert learn.has("rf-gain")


# -- searching -------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("squelch", "squelch"),
        ("rf gain", "rf-gain"),
        ("waterfall", "waterfall"),
        ("ppm", "ppm"),
        ("stereo blend", "stereo-blend"),
        ("pocsag", "pocsag"),
        ("de-emphasis", "deemphasis"),
    ],
)
def test_the_obvious_query_puts_its_article_first(query, expected):
    found = learn.search(query)
    assert found, query
    assert found[0].slug == expected


def test_a_word_someone_met_elsewhere_finds_the_article():
    """Aliases are the whole reason search is not just a title match.

    Nobody looking up "capcode" knows the article is called POCSAG, and
    nobody who has read "SNR" on a forum knows this app spells it out.
    """
    assert learn.search("capcode")[0].slug == "pocsag"
    assert learn.search("SNR")[0].slug == "snr"
    assert learn.search("dBFS")[0].slug == "decibel"
    assert learn.search("aerial")[0].slug == "antenna"


def test_two_words_narrow_rather_than_widen():
    both = learn.search("noise reduction")
    assert both[0].slug in {"if-noise-reduction", "audio-noise-reduction"}
    assert len(both) < len(learn.search("noise"))


def test_a_typed_question_still_finds_something():
    """The fallback to any-word matching, which is what makes it a search box.

    Requiring every word to match answers "why is my audio quiet" with a blank
    page, and a blank page reads as a glossary that does not cover it.
    """
    assert learn.search("why is my audio quiet")
    assert learn.search("what does squelch mean")[0].slug == "squelch"


def test_an_empty_query_returns_nothing_rather_than_everything():
    assert learn.search("") == ()
    assert learn.search("   ") == ()


def test_results_are_stable_between_identical_queries():
    """Equal scores must not swap places under the cursor."""
    first = [article.slug for article in learn.search("noise")]
    assert first == [article.slug for article in learn.search("noise")]


# -- what the views point at -----------------------------------------------


def _topics_named_in(path: Path) -> set[str]:
    return set(TOPIC_ARGUMENT.findall(path.read_text(encoding="utf-8")))


def test_every_topic_a_control_names_has_an_article():
    """The one failure nobody would ever report.

    A caption whose topic does not exist falls back to a plain label, so the
    control looks completely normal - it simply never offers an explanation
    again. Renaming a slug is what would cause it, and this is the only thing
    standing between that rename and forty silent captions.
    """
    named: set[str] = set()
    for path in UI_DIR.rglob("*.py"):
        named |= _topics_named_in(path)
    assert named, "no control names a topic - the wiring has come undone"
    missing = sorted(topic for topic in named if not learn.has(topic))
    assert not missing, f"controls point at topics with no article: {missing}"


def test_the_listening_screen_explains_the_controls_that_need_it_most():
    """A spot check with teeth, in case the wiring is dropped wholesale.

    These are the four the project's own notes describe as most often set
    wrongly or least self-explanatory, so their absence is not a cosmetic
    regression.
    """
    named = _topics_named_in(UI_DIR / "listen_view.py")
    assert {"rf-gain", "squelch", "bandwidth", "sample-rate"} <= named


def test_the_discover_screen_explains_the_controls_that_need_it_most():
    """Two settings and two whole ideas.

    Monitor and voice detection are not settings with a right value; they are
    mental models - "keep sweeping and stop on anything that talks", "listen
    to a channel to hear whether that is a person or a pager" - and a beginner
    who has not met either has nowhere else to find out what they do.
    """
    named = _topics_named_in(UI_DIR / "discover_view.py")
    assert {
        "sensitivity",
        "sort-order",
        "monitor-mode",
        "voice-detection",
    } <= named
