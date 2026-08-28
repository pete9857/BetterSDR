"""The listening screen: frequency, meter, spectrum, waterfall and controls.

This is a *view* onto `core.engine.Engine`. It owns no threads and touches no
device. A 30 Hz timer reads the newest frame out of the engine's mailbox and
paints it; if a frame is missed it is simply never drawn, which is the correct
behaviour for a display and the reason the mailbox is a single slot rather
than a queue.

Controls are registered with the minimum level at which they appear, so
promoting one from Standard to Simple is a one-word change.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.engine import DisplayFrame, Engine
from ..dsp import demod
from ..dsp.features import detect_hd_radio
from ..scan import bandplan
from .levels import Level
from .widgets import colormaps
from .widgets.frequency import FrequencyDisplay
from .widgets.meter import SignalMeter
from .widgets.spectrum import BandRibbon, SpectrumWidget
from .widgets.waterfall import WaterfallWidget

REFRESH_HZ = 30
# HD detection is cheap but not free, and a station does not start or stop
# carrying HD between frames. Once a second is plenty.
HD_CHECK_EVERY = REFRESH_HZ

PANEL_STYLE = """
QWidget#controls { background: #10151c; }
QLabel { color: #8b98a5; }
QLabel#bandName { color: #e6edf3; font-size: 13px; font-weight: 600; }
QLabel#bandInfo { color: #8b98a5; }
QLabel#hdBadge {
    color: #0b0e13; background: #5ad1ff; border-radius: 3px;
    padding: 1px 6px; font-weight: 600;
}
QComboBox, QDoubleSpinBox {
    background: #161b22; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 3px; padding: 2px 6px;
}
QCheckBox { color: #8b98a5; }
"""


class ListenView(QWidget):
    """Tune, listen, and watch the band."""

    def __init__(
        self, engine: Engine, level: Level = Level.STANDARD, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self._levelled: list[tuple[QWidget, Level]] = []
        self._frames = 0
        self._auto_ranged = False
        self._band: bandplan.Band | None = None

        self._build()
        self.set_level(level)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(int(1000 / REFRESH_HZ))

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setStyleSheet(PANEL_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.frequency = FrequencyDisplay(value_hz=self.engine.center_hz)
        self.frequency.valueChanged.connect(self._tune)
        outer.addWidget(self.frequency)

        header = QHBoxLayout()
        header.setContentsMargins(10, 0, 10, 6)
        self.meter = SignalMeter()
        header.addWidget(self.meter, 3)

        band_box = QVBoxLayout()
        name_row = QHBoxLayout()
        self.band_name = QLabel("--")
        self.band_name.setObjectName("bandName")
        self.hd_badge = QLabel("HD")
        self.hd_badge.setObjectName("hdBadge")
        self.hd_badge.setVisible(False)
        name_row.addWidget(self.band_name)
        name_row.addWidget(self.hd_badge)
        name_row.addStretch(1)
        self.band_info = QLabel("")
        self.band_info.setObjectName("bandInfo")
        self.band_info.setWordWrap(True)
        band_box.addLayout(name_row)
        band_box.addWidget(self.band_info)
        header.addLayout(band_box, 4)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        display = QWidget()
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)
        self.ribbon = BandRibbon()
        self.spectrum = SpectrumWidget()
        self.waterfall = WaterfallWidget()
        self.spectrum.tuneRequested.connect(self._tune_from_display)
        self.spectrum.bandwidthChanged.connect(self._set_bandwidth)
        self.waterfall.tuneRequested.connect(self._tune_from_display)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self.spectrum)
        self._splitter.addWidget(self.waterfall)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 6)
        self._splitter.setHandleWidth(2)
        display_layout.addWidget(self.ribbon)
        display_layout.addWidget(self._splitter, 1)
        body.addWidget(display, 1)
        body.addWidget(self._controls())
        outer.addLayout(body, 1)

    def _controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("controls")
        panel.setFixedWidth(232)
        form = QFormLayout(panel)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(7)

        def add(label: str, widget: QWidget, level: Level) -> None:
            caption = QLabel(label)
            form.addRow(caption, widget)
            self._levelled.append((caption, level))
            self._levelled.append((widget, level))

        def separator(level: Level) -> None:
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color: #2b323b;")
            form.addRow(line)
            self._levelled.append((line, level))

        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(self.engine.volume * 100))
        self.volume.valueChanged.connect(lambda v: self.engine.set_volume(v / 100.0))
        add("Volume", self.volume, Level.SIMPLE)

        separator(Level.STANDARD)

        self.mode = QComboBox()
        for info in demod.mode_table():
            self.mode.addItem(info.label, info.mode)
        self.mode.setCurrentIndex(self.mode.findData(self.engine.mode))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        add("Mode", self.mode, Level.STANDARD)

        self.bandwidth = QDoubleSpinBox()
        self.bandwidth.setRange(0.1, 2_500.0)
        self.bandwidth.setSuffix(" kHz")
        self.bandwidth.setDecimals(1)
        self.bandwidth.setValue(200.0)
        self.bandwidth.valueChanged.connect(
            lambda khz: self.engine.set_bandwidth(khz * 1000.0)
        )
        add("Bandwidth", self.bandwidth, Level.STANDARD)

        self.squelch_on = QCheckBox("Squelch")
        self.squelch = QDoubleSpinBox()
        self.squelch.setRange(-90.0, 0.0)
        self.squelch.setSuffix(" dBFS")
        self.squelch.setValue(-40.0)
        self.squelch_on.toggled.connect(self._squelch_changed)
        self.squelch.valueChanged.connect(self._squelch_changed)
        add("", self.squelch_on, Level.STANDARD)
        add("Threshold", self.squelch, Level.STANDARD)

        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 49.6)
        self.gain.setSuffix(" dB")
        self.gain.setValue(
            self.engine.gain.gain_db if self.engine.gain is not None else 20.0
        )
        self.gain.valueChanged.connect(self.engine.set_gain)
        add("RF gain", self.gain, Level.STANDARD)

        separator(Level.STANDARD)

        self.colour_map = QComboBox()
        self.colour_map.addItems(colormaps.NAMES)
        self.colour_map.setCurrentText(colormaps.DEFAULT_NAME)
        self.colour_map.currentTextChanged.connect(self.waterfall.set_colour_map)
        add("Colours", self.colour_map, Level.STANDARD)

        self.range_floor = QDoubleSpinBox()
        self.range_floor.setRange(-140.0, 0.0)
        self.range_floor.setValue(-90.0)
        self.range_floor.setSuffix(" dB")
        self.range_ceiling = QDoubleSpinBox()
        self.range_ceiling.setRange(-140.0, 0.0)
        self.range_ceiling.setValue(-20.0)
        self.range_ceiling.setSuffix(" dB")
        self.range_floor.valueChanged.connect(self._range_changed)
        self.range_ceiling.valueChanged.connect(self._range_changed)
        add("Floor", self.range_floor, Level.STANDARD)
        add("Ceiling", self.range_ceiling, Level.STANDARD)

        self.peak_hold = QCheckBox("Peak hold")
        self.peak_hold.setChecked(True)
        self.peak_hold.toggled.connect(self.spectrum.set_peak_hold)
        add("", self.peak_hold, Level.STANDARD)

        separator(Level.EXPERT)

        self.waterfall_speed = QComboBox()
        for label, frames in (("Fast", 1), ("Medium", 2), ("Slow", 4)):
            self.waterfall_speed.addItem(label, frames)
        self.waterfall_speed.currentIndexChanged.connect(
            lambda: self.waterfall.set_speed(self.waterfall_speed.currentData())
        )
        add("Waterfall", self.waterfall_speed, Level.EXPERT)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self._levelled.append((self.status, Level.EXPERT))
        return panel

    # -- levels ------------------------------------------------------------

    def set_level(self, level: Level) -> None:
        self.level = level
        for widget, minimum in self._levelled:
            widget.setVisible(level >= minimum)

    # -- control handlers --------------------------------------------------

    def _tune(self, hz: int) -> None:
        self.engine.tune(int(hz))
        self.spectrum.reset_peak_hold()
        self._update_band_label(hz)
        self._apply_band_defaults(hz)

    def _apply_band_defaults(self, hz: float, force: bool = False) -> None:
        """Take mode and bandwidth from the band plan when the band changes.

        Without this, tuning from FM broadcast to the airband leaves an AM
        transmission being demodulated as wideband FM, which sounds like
        nothing at all. In Simple mode there is no mode control to fix it
        with, so getting this right is the difference between the app working
        and the app appearing broken.

        Only on a *change* of band, so a deliberate choice - listening to an
        FM station in narrow FM to cut the noise, say - survives retuning
        within that band.
        """
        band = bandplan.find(hz)
        if band is None or (band is self._band and not force):
            return
        self._band = band
        index = self.mode.findData(band.mode)
        if index >= 0:
            self.mode.setCurrentIndex(index)
        self.bandwidth.setValue(band.bandwidth_hz / 1000.0)

    def _tune_from_display(self, hz: float) -> None:
        """Click-to-tune, snapped to the band's channel raster if it has one."""
        band = bandplan.find(hz)
        snapped = band.snap(hz) if band else hz
        self.frequency.set_value(int(round(snapped)))
        self._tune(int(round(snapped)))

    def _mode_changed(self) -> None:
        mode = self.mode.currentData()
        self.engine.set_mode(mode)
        default = demod.MODES[mode].default_bandwidth_hz
        self.bandwidth.blockSignals(True)
        self.bandwidth.setValue(default / 1000.0)
        self.bandwidth.blockSignals(False)

    def _set_bandwidth(self, hz: float) -> None:
        self.bandwidth.setValue(hz / 1000.0)

    def _squelch_changed(self) -> None:
        threshold = self.squelch.value() if self.squelch_on.isChecked() else None
        self.engine.set_squelch(threshold)

    def _range_changed(self) -> None:
        self.waterfall.set_range(self.range_floor.value(), self.range_ceiling.value())

    def _update_band_label(self, hz: float) -> None:
        band = bandplan.find(hz)
        self.band_name.setText(band.name if band else "Unallocated")
        self.band_info.setText(
            band.description if band else "Nothing is normally broadcast here."
        )

    # -- refresh -----------------------------------------------------------

    def start(self) -> None:
        self._update_band_label(self.engine.center_hz)
        self._apply_band_defaults(self.engine.center_hz, force=True)
        if self.engine.gain is not None:
            self.gain.blockSignals(True)
            self.gain.setValue(self.engine.gain.gain_db)
            self.gain.blockSignals(False)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        frame = self.engine.latest()
        if frame is None:
            return
        self._frames += 1

        self.meter.set_level(frame.channel_power_dbfs, frame.squelch_open)
        self.spectrum.update_spectrum(
            frame.spectrum_db, frame.center_hz, frame.sample_rate
        )
        self.spectrum.set_passband(frame.center_hz, frame.bandwidth_hz)
        self.waterfall.push(frame.spectrum_db, frame.center_hz, frame.sample_rate)
        low = frame.center_hz - frame.sample_rate / 2.0
        self.ribbon.set_span(low, low + frame.sample_rate)

        if not self._auto_ranged:
            # One automatic fit on the first real frame, so the display opens
            # showing signal rather than an empty rectangle the user has to
            # calibrate themselves.
            self._auto_ranged = True
            self.spectrum.auto_range(frame.spectrum_db)
            self._fit_waterfall_range(frame.spectrum_db)

        if self._frames % HD_CHECK_EVERY == 0:
            self._update_hd_badge(frame)
            self.status.setText(
                f"buffer {frame.audio_latency_s * 1000:.0f} ms   "
                f"underruns {frame.underruns}   overruns {frame.ring_overruns}"
            )

    def _fit_waterfall_range(self, spectrum_db: np.ndarray) -> None:
        floor = float(np.percentile(spectrum_db, 30)) - 4.0
        ceiling = float(np.max(spectrum_db)) + 4.0
        self.range_floor.blockSignals(True)
        self.range_ceiling.blockSignals(True)
        self.range_floor.setValue(floor)
        self.range_ceiling.setValue(ceiling)
        self.range_floor.blockSignals(False)
        self.range_ceiling.blockSignals(False)
        self.waterfall.set_range(floor, ceiling)

    def _update_hd_badge(self, frame: DisplayFrame) -> None:
        """Flag stations that also carry HD Radio.

        Only meaningful on FM broadcast, and only when the sweep window
        actually reaches past the sidebands - `detect_hd_radio` refuses rather
        than guessing, and a refusal is not a reason to bother the user.
        """
        band = bandplan.find(frame.center_hz)
        if band is None or band.mode != "wfm":
            self.hd_badge.setVisible(False)
            return
        try:
            result = detect_hd_radio(frame.spectrum_db, frame.bin_width_hz)
        except ValueError:
            self.hd_badge.setVisible(False)
            return
        self.hd_badge.setVisible(result.present)
        if result.present:
            self.hd_badge.setToolTip(result.summary)


__all__ = ["ListenView"]
