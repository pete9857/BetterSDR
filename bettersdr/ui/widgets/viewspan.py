"""Zooming into the captured window and panning across it.

The radio hands the display one window - `sample_rate` hertz wide, centred on
the frequency it is tuned to - and until now that window *was* the picture. At
2.4 MS/s a 12.5 kHz marine channel is a fifth of one pixel on a thousand-pixel
pane, so "what is sitting next to this signal?" was a question the app could
only answer by narrowing the window, which changes what the radio is doing
rather than what the screen is showing.

Zoom and pan are therefore a *view* on that window and nothing else: no device
call, no second FFT, no resampling. Two numbers say it -

    zoom    how many times narrower than the window the view is, never < 1
    offset  where the middle of the view sits, as a fraction of the whole
            window away from its middle

Both are fractions of the window rather than hertz, so a retune or a change of
window width does not leave them meaning something else afterwards. The
arithmetic is pure and lives in one place because three stacked panes have to
agree on it exactly - a waterfall showing a slightly different span from the
spectrum above it puts every frequency wrong, which is the fault `AXIS_WIDTH`
exists to prevent one layer further down.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from PySide6.QtCore import Qt

MIN_ZOOM = 1.0
# Enough to fill the pane with one 12.5 kHz channel out of a 2.4 MHz window,
# which is the narrowest thing the app has to show against the widest window
# it opens. Past that the transform runs out of bins before the eye runs out
# of detail: 4096 bins across 2.4 MHz is 586 Hz each, so a 37 kHz view is
# already only 64 of them and zooming further magnifies the FFT rather than
# the radio.
MAX_ZOOM = 64.0
# One wheel notch. 1.2 takes 1x to 64x in 23 notches, about two comfortable
# flicks of a wheel in either direction.
WHEEL_STEP = 1.2
# A press that never travels this far is a click, not a drag. Not zero,
# because a mouse moves a pixel or two under a real finger and a click that
# silently became a one-pixel pan would read as click-to-tune being broken.
DRAG_SLOP_PX = 4.0
# How near an edge of the passband a press has to land to be a grab of that
# edge rather than a click on the spectrum behind it.
EDGE_GRAB_PX = 6.0
# The zoom slider's resolution: the number of equal ratio steps between the
# whole window and `MAX_ZOOM`.
SLIDER_STEPS = 100


class View(NamedTuple):
    """A window on the window. `FULL` is the whole of it, and the default."""

    zoom: float = 1.0
    offset: float = 0.0


FULL = View(1.0, 0.0)


def clamped(zoom: float, offset: float) -> View:
    """The nearest legal view to the one asked for.

    The view may not reach past the edge of the samples the radio actually
    captured, because there is nothing out there to draw: a pane that scrolls
    into blackness looks like a receiver that has gone deaf, and the honest
    answer to "pan further" at the edge of the window is to stop.
    """
    zoom = min(max(float(zoom), MIN_ZOOM), MAX_ZOOM)
    limit = (1.0 - 1.0 / zoom) / 2.0
    return View(zoom, min(max(float(offset), -limit), limit))


def span(center_hz: float, sample_rate: float, view: View) -> tuple[float, float]:
    """The frequencies at the left and right edges of the pane."""
    zoom, offset = clamped(*view)
    width = float(sample_rate) / zoom
    middle = float(center_hz) + offset * float(sample_rate)
    return middle - width / 2.0, middle + width / 2.0


def zoomed(view: View, factor: float, anchor: float = 0.5) -> View:
    """Zoom by `factor`, holding still whatever is at `anchor`.

    `anchor` is a fraction across the visible pane, so a wheel passes the
    pointer's own position and zooms under the cursor, while a slider passes
    the middle and zooms about the centre of the view. Working in fractions of
    the window means this is the same arithmetic whatever the radio is tuned
    to and however wide its window is.
    """
    view = clamped(*view)
    zoom = min(max(view.zoom * float(factor), MIN_ZOOM), MAX_ZOOM)
    held = view.offset + (float(anchor) - 0.5) / view.zoom
    return clamped(zoom, held - (float(anchor) - 0.5) / zoom)


def panned(view: View, delta_hz: float, sample_rate: float) -> View:
    """Slide the view `delta_hz` up the dial, staying inside the window."""
    if sample_rate <= 0.0:
        return clamped(*view)
    return clamped(view.zoom, view.offset + float(delta_hz) / float(sample_rate))


def zoom_for_slider(value: int, steps: int = SLIDER_STEPS) -> float:
    """Slider position to zoom, logarithmically.

    A linear slider would spend its first half between 1x and 32x and its
    second half between 32x and 64x, which is one useful step and a hundred
    useless ones. Each step here is the same *ratio*, so the control behaves
    the same at both ends of its travel.
    """
    return float(MAX_ZOOM ** (min(max(int(value), 0), steps) / steps))


def slider_for_zoom(zoom: float, steps: int = SLIDER_STEPS) -> int:
    """Zoom to slider position - the inverse, so a round trip does not drift."""
    zoom = min(max(float(zoom), MIN_ZOOM), MAX_ZOOM)
    return int(round(steps * math.log(zoom) / math.log(MAX_ZOOM)))


class PanZoom:
    """Wheel zoom, drag pan and click-to-tune for a pane with a frequency axis.

    Mixed into the spectrum and the waterfall so that the two behave
    identically. They are stacked and read as one picture, so a wheel that
    zoomed one of them and not the other would read as a fault rather than as
    a rule.

    The widget mixing this in must declare both signals itself -
    `viewChanged(zoom, offset)` and `tuneRequested(hz)` - because Qt's
    metaclass only sees a signal declared on the QObject subclass, and must
    call `_init_pan_zoom()` from its constructor: a mixin with an `__init__`
    of its own would sit in front of the widget's in the MRO.

    Nothing here decides anything. A pane emits what the gesture asked for and
    is told what to show, so the two panes and the ribbon above them cannot
    end up displaying different spans of the same window.
    """

    # -- state -------------------------------------------------------------

    def _init_pan_zoom(self) -> None:
        self._view = FULL
        self._view_center_hz = 0.0
        self._view_rate_hz = 0.0
        self._press_x: float | None = None
        self._grab_hz = 0.0
        self._dragging = False

    def current_view(self) -> View:
        return self._view

    def set_view(self, zoom: float, offset: float) -> None:
        """Show this view of the window.

        Idempotent, so a pane can be told what it has just asked for without
        the two of them handing it back and forth.
        """
        view = clamped(zoom, offset)
        if view == self._view:
            return
        self._view = view
        self._apply_view()

    def set_window(self, center_hz: float, sample_rate: float) -> None:
        """The window the samples came from, which the view is a fraction of."""
        self._view_center_hz = float(center_hz)
        self._view_rate_hz = float(sample_rate)
        self._apply_view()

    def _apply_view(self) -> None:
        if self._view_rate_hz <= 0.0:
            return
        low, high = span(self._view_center_hz, self._view_rate_hz, self._view)
        self.setXRange(low, high, padding=0.0)

    def _change_view(self, view: View) -> None:
        """Apply a gesture's result here, and tell whoever else is showing it."""
        if view == self._view:
            return
        self.set_view(*view)
        self.viewChanged.emit(view.zoom, view.offset)

    # -- geometry ----------------------------------------------------------

    def _hz_at(self, position) -> float:
        return float(self.getPlotItem().vb.mapSceneToView(position).x())

    def _hz_per_px(self) -> float:
        box = self.getPlotItem().getViewBox()
        width = float(box.width())
        if width <= 1.0:
            return 0.0
        low, high = box.viewRange()[0]
        return (high - low) / width

    def _anchor(self, position) -> float:
        """Where the pointer is across the pane, as a fraction from the left."""
        low, high = self.getPlotItem().getViewBox().viewRange()[0]
        if high <= low:
            return 0.5
        return min(max((self._hz_at(position) - low) / (high - low), 0.0), 1.0)

    def _grabs_an_item(self, hz: float) -> bool:
        """Whether a press at `hz` belongs to something drawn on the pane.

        The spectrum's passband edges are the only such thing, and the
        waterfall has none - so the default is that a press anywhere on the
        pane is the pane's own.
        """
        return False

    # -- gestures ----------------------------------------------------------

    def wheelEvent(self, event) -> None:
        if self._view_rate_hz <= 0.0:
            super().wheelEvent(event)
            return
        notches = event.angleDelta().y() / 120.0
        if notches:
            self._change_view(
                zoomed(self._view, WHEEL_STEP**notches, self._anchor(event.position()))
            )
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Take the press unless something drawn on the pane wants it.

        Not calling `super()` is the point rather than an omission: pyqtgraph's
        scene turns a press it has seen into click and drag events for the
        items under it, so letting it see one that is going to be a pan is how
        a pan ends up dragging the passband with it.
        """
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._view_rate_hz <= 0.0
            or self._grabs_an_item(self._hz_at(event.position()))
        ):
            self._press_x = None
            super().mousePressEvent(event)
            return
        self._press_x = float(event.position().x())
        self._grab_hz = self._hz_at(event.position())
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_x is None:
            # Still forwarded when nothing is held down, or the passband edges
            # stop lighting up under the pointer.
            super().mouseMoveEvent(event)
            return
        if not self._dragging:
            if abs(float(event.position().x()) - self._press_x) < DRAG_SLOP_PX:
                event.accept()
                return
            self._dragging = True
        # Measured against the frequency the drag started on rather than
        # accumulated from the last move, so the point under the pointer stays
        # under it however many times the clamp has stopped the view at an edge
        # on the way there and back.
        self._change_view(
            panned(
                self._view,
                self._grab_hz - self._hz_at(event.position()),
                self._view_rate_hz,
            )
        )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._press_x is None:
            super().mouseReleaseEvent(event)
            return
        self._press_x = None
        if not self._dragging:
            self.tuneRequested.emit(self._hz_at(event.position()))
        self._dragging = False
        event.accept()


__all__ = [
    "DRAG_SLOP_PX",
    "EDGE_GRAB_PX",
    "FULL",
    "MAX_ZOOM",
    "MIN_ZOOM",
    "SLIDER_STEPS",
    "WHEEL_STEP",
    "PanZoom",
    "View",
    "clamped",
    "panned",
    "slider_for_zoom",
    "span",
    "zoom_for_slider",
    "zoomed",
]
