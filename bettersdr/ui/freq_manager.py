"""The frequency manager: named bookmarks, grouped, with import and export.

Every radio since the 1970s has had memory channels, and an SDR# user will
look for them within the first five minutes. The window is deliberately plain
- a grouped list, four buttons - because the interesting decisions are all in
`core/bookmarks.py`, which has no Qt in it so the star button on a signal card
and this window can never disagree about what a bookmark is.

Double-clicking a row tunes to it. That is the only interaction most people
will ever use, so it is the one that needs no explanation.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.bookmarks import Bookmark, BookmarkStore
from ..dsp import demod

DIALOG_STYLE = """
QDialog { background: #0b0e13; }
QTreeWidget {
    background: #10151c; color: #e6edf3;
    border: 1px solid #2b323b; alternate-background-color: #131923;
}
QTreeWidget::item:selected { background: #1f3a4a; }
QHeaderView::section {
    background: #161b22; color: #8b98a5;
    border: none; border-right: 1px solid #2b323b; padding: 4px;
}
QPushButton {
    background: #1b222c; color: #cbd5e0;
    border: 1px solid #2b323b; border-radius: 3px; padding: 5px 12px;
}
QPushButton:hover { background: #232b37; }
QPushButton:disabled { color: #4a5460; border-color: #1d232b; }
"""

COLUMNS = ("Name", "Frequency", "Mode", "Bandwidth", "Notes")


def _mode_label(mode: str) -> str:
    """The friendly name, so the list does not read as a table of acronyms."""
    cls = demod.MODES.get(mode)
    return cls.label if cls is not None else mode.upper()


class FrequencyManager(QDialog):
    """Browse, edit and recall saved frequencies."""

    tuneRequested = Signal(object)

    def __init__(self, store: BookmarkStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Saved frequencies")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(660, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.itemDoubleClicked.connect(self._listen)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self._listen_button = self._button(buttons, "Listen", self._listen)
        self._rename_button = self._button(buttons, "Rename", self._rename)
        self._delete_button = self._button(buttons, "Delete", self._delete)
        buttons.addStretch(1)
        self._button(buttons, "Import...", self._import)
        self._button(buttons, "Export...", self._export)
        layout.addLayout(buttons)

        self.refresh()

    def _button(self, layout: QHBoxLayout, text: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(slot)
        layout.addWidget(button)
        return button

    # -- contents ----------------------------------------------------------

    def refresh(self) -> None:
        self.tree.clear()
        for group in self.store.groups:
            parent = QTreeWidgetItem(self.tree, [group])
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            parent.setFirstColumnSpanned(True)
            for entry in self.store.in_group(group):
                item = QTreeWidgetItem(
                    parent,
                    [
                        entry.name or "(unnamed)",
                        entry.label.split(" - ")[-1],
                        _mode_label(entry.mode),
                        f"{entry.bandwidth_hz / 1000:g} kHz",
                        entry.notes,
                    ],
                )
                item.setData(0, Qt.ItemDataRole.UserRole, entry)
            parent.setExpanded(True)
        self._selection_changed()

    def selected(self) -> Bookmark | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _selection_changed(self) -> None:
        has = self.selected() is not None
        for button in (
            self._listen_button,
            self._rename_button,
            self._delete_button,
        ):
            button.setEnabled(has)

    # -- actions -----------------------------------------------------------

    def _listen(self) -> None:
        entry = self.selected()
        if entry is not None:
            self.tuneRequested.emit(entry)

    def _rename(self) -> None:
        entry = self.selected()
        if entry is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename", "Name for this frequency:", text=entry.name
        )
        if accepted:
            self.store.rename(entry, name.strip())
            self.store.save()
            self.refresh()

    def _delete(self) -> None:
        entry = self.selected()
        if entry is None:
            return
        confirmed = QMessageBox.question(
            self,
            "Delete",
            f"Remove {entry.label} from your saved frequencies?",
        )
        if confirmed is QMessageBox.StandardButton.Yes:
            self.store.remove(entry)
            self.store.save()
            self.refresh()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import frequencies", "", "Frequency lists (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            # utf-8-sig, because a list exported from a spreadsheet
            # arrives with a byte-order mark on the first column name and
            # would otherwise import with no "name" column at all.
            text = Path(path).read_text(encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.warning(self, "Import", f"That file could not be read.\n\n{exc}")
            return
        taken = self.store.from_csv(text)
        self.store.save()
        self.refresh()
        QMessageBox.information(
            self,
            "Import",
            f"Added {taken} frequenc{'y' if taken == 1 else 'ies'} to your list."
            if taken
            else "No usable frequencies were found in that file.",
        )

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export frequencies", "frequencies.csv", "Frequency lists (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(self.store.to_csv())
        except OSError as exc:
            QMessageBox.warning(
                self, "Export", f"That file could not be written.\n\n{exc}"
            )


__all__ = ["FrequencyManager"]
