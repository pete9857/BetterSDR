"""The Learn screen: a home page to browse and search, and one article.

Two ways in, and they are not the same journey.

The first is arriving from a control. Somebody is looking at a row called
"Threshold", clicks the word, and lands on the article about it. They did not
come here to browse; they came with a question, and the screen has to answer
it and then get out of the way - which is why an article opened that way keeps
a Back button that goes to the page they came from, not to this screen's own
home page.

The second is pressing Learn. Nobody arriving that way has a specific
question, so they get the home page: everything the app can explain, grouped
in the order a beginner meets it, with a search box above it. Browsing is
offered before searching because you cannot search for a word you have not
met yet, which is the exact predicament this whole tab exists for.

Like every other page here this is a *view*. It touches no device, owns no
threads and does not even poll - it is the one screen in the app with nothing
live on it, so `start` and `stop` have nothing to do beyond satisfying the
page protocol the window expects.

**Nothing on this screen is level-gated**, and that is a deliberate exception
to the rule the rest of the UI follows. Levels decide what you may change,
never what you may understand; somebody in Simple mode who has read the words
"RF gain" and wants to know what they mean is precisely the reader this was
built for, and the fact that they cannot yet see that control is not a reason
to withhold the explanation of it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import learn
from .learn import Article
from .levels import Level
from .widgets.icons import glyph

BACKGROUND = "#0b0e13"

# How many search results to draw. Beyond this the list has stopped being an
# answer and started being the glossary again, and the right response is a
# better query - which the status line says out loud rather than leaving the
# reader to scroll and wonder.
MAX_RESULTS = 24

VIEW_STYLE = """
QWidget#learn { background: #0b0e13; }
QLabel#heading { color: #e6edf3; font-size: 19px; font-weight: 600; }
QLabel#subheading { color: #8b98a5; font-size: 12px; }
QLabel#status { color: #6d7b89; font-size: 11px; }
QLabel#groupTitle {
    color: #5ad1ff; font-size: 10px; font-weight: 700;
    padding: 14px 0 2px 0;
}
QLineEdit#search {
    background: #10151c; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 15px;
    padding: 7px 14px; font-size: 13px;
}
QLineEdit#search:focus { border-color: #5ad1ff; }
QScrollArea { border: none; background: #0b0e13; }
QWidget#learnList, QWidget#articleBody { background: #0b0e13; }

/* One entry in the browse or search list. A whole card is the target rather
   than a link inside it: the summary is the half that tells you whether you
   want the article, so clicking it has to open the article. */
QFrame#entry {
    background: #10151c; border: 1px solid #1d232b; border-radius: 6px;
}
QFrame#entry:hover { border-color: #5ad1ff; }
QLabel#entryTitle { color: #e6edf3; font-size: 13px; font-weight: 600; }
QLabel#entrySummary { color: #8b98a5; font-size: 12px; }
QLabel#entryAlso { color: #55606d; font-size: 11px; }

QLabel#articleTitle { color: #e6edf3; font-size: 22px; font-weight: 600; }
QLabel#articleAlso { color: #6d7b89; font-size: 11px; }
QLabel#articleSummary { color: #b6c2cf; font-size: 14px; }
QLabel#articleBodyText { color: #b6c2cf; font-size: 13px; }
QLabel#articleWhereTitle { color: #5ad1ff; font-size: 10px; font-weight: 700; }
QLabel#articleWhere { color: #8b98a5; font-size: 12px; }
QLabel#seeAlsoTitle {
    color: #5ad1ff; font-size: 10px; font-weight: 700; padding-top: 6px;
}
QFrame#rule { background: #1d232b; max-height: 1px; border: none; }

QPushButton#back {
    background: #10151c; color: #cbd5e0;
    border: 1px solid #2b323b; border-radius: 4px;
    padding: 5px 14px; font-size: 12px;
}
QPushButton#back:hover { border-color: #5ad1ff; color: #e6edf3; }
QPushButton#seeAlso {
    background: #161b22; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 11px;
    padding: 4px 12px; font-size: 11px;
}
QPushButton#seeAlso:hover { border-color: #5ad1ff; }
"""


def _dark_viewport(area: QScrollArea) -> None:
    """Paint a scroll area's viewport without a stylesheet.

    A stylesheet on a viewport drags every descendant through the stylesheet
    style, and `QLabel` is a `QFrame`, so each one starts painting a frame it
    never asked for - here that would be a box around every paragraph of every
    article. Same rule as the control panel and the Discover list.
    """
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.viewport().setAutoFillBackground(True)
    palette = area.viewport().palette()
    palette.setColor(area.viewport().backgroundRole(), QColor(BACKGROUND))
    area.viewport().setPalette(palette)


class Entry(QFrame):
    """One article in a list: its title, what else it is called, its summary.

    The summary is on the card rather than behind it because the browse list
    has to be readable *as a list*. Seventy titles alone is an index, and an
    index only helps somebody who already knows the vocabulary.
    """

    chosen = QtSignal(str)

    def __init__(self, article: Article, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("entry")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._slug = article.slug

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(8)
        title = QLabel(article.title)
        title.setObjectName("entryTitle")
        top.addWidget(title)
        if article.also:
            # Two at most. The point is recognition - "oh, that is the thing
            # I saw called SNR" - and a full list of synonyms competes with
            # the summary underneath for the same glance.
            also = QLabel(", ".join(article.also[:2]))
            also.setObjectName("entryAlso")
            top.addWidget(also)
        top.addStretch(1)
        layout.addLayout(top)

        summary = QLabel(learn.strip_links(article.summary))
        summary.setObjectName("entrySummary")
        summary.setWordWrap(True)
        layout.addWidget(summary)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.chosen.emit(self._slug)
        super().mouseReleaseEvent(event)


class LearnView(QWidget):
    """Browse and search what the app can explain, and read one entry."""

    # Emitted when the reader presses Back on an article they arrived at from
    # a control. The window decides where that goes, because this screen has
    # no business knowing which page was showing before it.
    backRequested = QtSignal()

    HOME = 0
    ARTICLE = 1

    def __init__(self, level: Level = Level.STANDARD, parent: QWidget | None = None):
        super().__init__(parent)
        self.level = level
        self._entries: list[Entry] = []
        self._body_widgets: list[QWidget] = []
        self._see_buttons: list[QPushButton] = []
        # Where Back goes. A reader who arrived from a control wants the
        # control back; one who walked here from the home page wants the home
        # page. Same button, and the difference is which of the two journeys
        # in the module docstring they are on.
        self._return_to_app = False
        self._slug = ""

        self._build()
        self.show_home()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setObjectName("learn")
        self.setStyleSheet(VIEW_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._home())
        self._stack.addWidget(self._article())
        outer.addWidget(self._stack, 1)

    def _home(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        heading = QLabel("What would you like to understand?")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        subheading = QLabel(
            "Every control in this app is explained here - and you can get "
            "straight to any of them by clicking its name on the screen it "
            "is on."
        )
        subheading.setObjectName("subheading")
        subheading.setWordWrap(True)
        layout.addWidget(subheading)

        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText(
            "Search - try squelch, hiss, waterfall, or why is it so noisy"
        )
        self.search.textChanged.connect(self._search_changed)
        layout.addSpacing(4)
        layout.addWidget(self.search)

        self.status = QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self._list_area = QScrollArea()
        _dark_viewport(self._list_area)
        holder = QWidget()
        holder.setObjectName("learnList")
        self._list = QVBoxLayout(holder)
        self._list.setContentsMargins(0, 0, 8, 0)
        self._list.setSpacing(6)
        self._list.addStretch(1)
        self._list_area.setWidget(holder)
        layout.addWidget(self._list_area, 1)
        return page

    def _article(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(0)

        top = QHBoxLayout()
        self.back = QPushButton(f"{glyph('left')}  Back")
        self.back.setObjectName("back")
        self.back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back.clicked.connect(self._back)
        top.addWidget(self.back)
        top.addStretch(1)
        layout.addLayout(top)
        layout.addSpacing(12)

        area = QScrollArea()
        _dark_viewport(area)
        holder = QWidget()
        holder.setObjectName("articleBody")
        # Narrow, and deliberately so. A paragraph the full width of a
        # 1180 px window is about 180 characters a line, which is roughly
        # twice what anybody reads comfortably.
        holder.setMaximumWidth(720)
        holder.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._body = QVBoxLayout(holder)
        self._body.setContentsMargins(0, 0, 12, 12)
        self._body.setSpacing(0)
        area.setWidget(holder)
        layout.addWidget(area, 1)
        return page

    # -- the home page -----------------------------------------------------

    def show_home(self) -> None:
        """Everything there is, grouped, with the search box cleared."""
        # Cleared rather than left as it was: pressing Learn is arriving, not
        # returning, and meeting a filtered list from twenty minutes ago reads
        # as a glossary that has lost most of its entries. `textChanged` then
        # redraws the browse list for us if there was anything in the box.
        if self.search.text():
            self.search.clear()
        else:
            self._browse()
        self._stack.setCurrentIndex(self.HOME)

    def _search_changed(self, text: str) -> None:
        query = text.strip()
        if not query:
            self._browse()
            return
        found = learn.search(query)
        self._fill(found[:MAX_RESULTS])
        if not found:
            self.status.setText(
                f'Nothing here matches "{query}". '
                f"Clear the box to see everything."
            )
        elif len(found) > MAX_RESULTS:
            self.status.setText(
                f"{MAX_RESULTS} of {len(found)} matches - try a more "
                f"specific word."
            )
        else:
            word = "entry" if len(found) == 1 else "entries"
            self.status.setText(f"{len(found)} {word}")

    def _browse(self) -> None:
        """The whole glossary, under its headings, in the order it is meant.

        Grouped rather than alphabetical, because the ordering *is* content:
        "Start here" before "The receiver" is a claim about what to read
        first, and an A-Z would throw that away in exchange for an index
        nobody needs in a list this size.
        """
        self._clear()
        total = 0
        for category in learn.load():
            self._list.insertWidget(self._list.count() - 1, self._group(category.name))
            for article in category.articles:
                self._add_entry(article)
                total += 1
        self.status.setText(f"{total} entries, grouped by what they are about")

    def _fill(self, found: tuple[Article, ...]) -> None:
        self._clear()
        for article in found:
            self._add_entry(article)

    def _group(self, name: str) -> QLabel:
        label = QLabel(name.upper())
        label.setObjectName("groupTitle")
        return label

    def _add_entry(self, article: Article) -> None:
        entry = Entry(article)
        entry.chosen.connect(self.show_topic)
        self._list.insertWidget(self._list.count() - 1, entry)
        self._entries.append(entry)

    def _clear(self) -> None:
        while self._list.count() > 1:
            item = self._list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._entries.clear()

    # -- one article -------------------------------------------------------

    def show_topic(self, slug: str, from_app: bool = False) -> bool:
        """Open an article. Returns whether there was one to open.

        `from_app` says the reader arrived by clicking a control rather than
        by browsing, which is the only thing that decides where Back goes.

        An unknown slug lands on the home page rather than on an error. It is
        a mistake - `tests/test_learn.py` walks every topic the views name -
        but the reader did not make it, and a glossary is a poor place to
        display a stack trace.
        """
        article = learn.get(slug)
        if article is None:
            self.show_home()
            return False
        self._slug = slug
        self._return_to_app = from_app
        self.back.setText(
            f"{glyph('left')}  Back" if from_app else f"{glyph('left')}  All topics"
        )
        self._render(article)
        self._stack.setCurrentIndex(self.ARTICLE)
        return True

    @property
    def topic(self) -> str:
        """The article on screen, or "" on the home page."""
        return self._slug if self._stack.currentIndex() == self.ARTICLE else ""

    def _render(self, article: Article) -> None:
        self._clear_body()
        self._add_body(self._plain(article.title, "articleTitle"))
        if article.also:
            self._add_body(
                self._plain("Also called " + ", ".join(article.also), "articleAlso"),
                top=2,
            )
        self._add_body(self._plain(article.summary, "articleSummary"), top=10)

        rule = QFrame()
        rule.setObjectName("rule")
        rule.setFixedHeight(1)
        self._add_body(rule, top=12)

        for paragraph in article.body:
            self._add_body(self._rich(paragraph), top=12)

        if article.where:
            self._add_body(self._plain("WHERE TO FIND IT", "articleWhereTitle"), top=20)
            self._add_body(self._plain(article.where, "articleWhere"), top=2)

        if article.see:
            self._add_body(self._plain("SEE ALSO", "seeAlsoTitle"), top=20)
            self._add_body(self._see_also(article), top=4)
        self._body.addStretch(1)

    def _see_also(self, article: Article) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for slug in article.see:
            other = learn.get(slug)
            if other is None:
                # Never rendered as a dead chip. Same rule as an inline link
                # to a missing article: nothing may look clickable and then
                # do nothing.
                continue
            button = QPushButton(other.title)
            button.setObjectName("seeAlso")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(learn.strip_links(other.summary))
            button.clicked.connect(
                lambda _checked=False, target=slug: self._follow(target)
            )
            row.addWidget(button)
            self._see_buttons.append(button)
        row.addStretch(1)
        return holder

    def _plain(self, text: str, name: str) -> QLabel:
        label = QLabel(learn.strip_links(text))
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _rich(self, paragraph: str) -> QLabel:
        """A body paragraph, with its cross-references live.

        Following a link inside the prose is the difference between a glossary
        and a set of index cards - the whole reason the content carries
        `[[slug]]` markup rather than naming other articles in plain words.
        """
        label = QLabel(learn.to_html(paragraph))
        label.setObjectName("articleBodyText")
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.linkActivated.connect(self._follow)
        return label

    def _follow(self, slug: str) -> None:
        """Step to another article from inside this one.

        Keeps whichever journey the reader was on: somebody who came from a
        control and read two links deep still wants Back to return them to the
        control, not to a glossary they never opened.
        """
        self.show_topic(slug, from_app=self._return_to_app)

    def _add_body(self, widget: QWidget, top: int = 0) -> None:
        if top:
            self._body.addSpacing(top)
        self._body.addWidget(widget)
        self._body_widgets.append(widget)

    def _clear_body(self) -> None:
        self._see_buttons.clear()
        self._body_widgets.clear()
        while self._body.count():
            item = self._body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _back(self) -> None:
        if self._return_to_app:
            self.backRequested.emit()
            return
        self.show_home()

    # -- the page protocol -------------------------------------------------

    def set_level(self, level: Level) -> None:
        """Accepted and remembered; nothing here is gated on it.

        See the module docstring: levels decide what you may change, never
        what you may understand. The method exists because the window calls it
        on every page, and a Learn tab that quietly went missing at Simple
        would be the exact opposite of the point.
        """
        self.level = level

    def start(self) -> None:
        """Nothing to start. This is the one screen with nothing live on it."""

    def stop(self) -> None:
        """Nothing to stop, for the same reason."""


__all__ = ["MAX_RESULTS", "Entry", "LearnView"]
