"""The listening screen: frequency, meter, spectrum, waterfall and controls.

This is a *view* onto `core.engine.Engine`. It owns no threads and touches no
device. A 30 Hz timer reads the newest frame out of the engine's mailbox and
paints it; if a frame is missed it is simply never drawn, which is the correct
behaviour for a display and the reason the mailbox is a single slot rather
than a queue.

Controls are registered with the minimum level at which they appear, so
promoting one from Standard to Simple is a one-word change. Phase 3 takes the
column from eight controls to about forty, which is why they now live in
`widgets/panel.py` under headings that hide themselves when everything under
them belongs to a higher level - a screen of empty headings reads as an app
that has broken rather than one being quiet.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..audio import output
from ..core.bookmarks import Bookmark, BookmarkStore
from ..core.calibrate import calibrate
from ..core.device import DEFAULT_SAMPLE_RATE
from ..core.engine import DisplayFrame, Engine
from ..core.frontend import SUPPORTED_SAMPLE_RATES, GainChoice
from ..core.settings import Settings
from ..decode.rds import RdsState
from ..dsp import demod
from ..dsp.features import detect_hd_radio
from ..dsp.psd import WINDOWS
from ..scan import bandplan
from ..scan.classifier import Signal
from .freq_manager import FrequencyManager
from .levels import Level
from .widgets import colormaps
from .widgets.frequency import FrequencyDisplay
from .widgets.meter import SignalMeter
from .widgets.panel import ControlPanel
from .widgets.spectrum import BandRibbon, SpectrumWidget
from .widgets.waterfall import WaterfallWidget

REFRESH_HZ = 30
# HD detection is cheap but not free, and a station does not start or stop
# carrying HD between frames. Once a second is plenty, and the same tick
# refreshes the status and recording lines.
SLOW_TICK = REFRESH_HZ

FFT_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
WINDOW_LABELS = {
    "hann": "Hann",
    "blackmanharris": "Blackman-Harris",
    "hamming": "Hamming",
    "flattop": "Flat top",
    "boxcar": "None (rectangular)",
}
DEEMPHASIS_CHOICES = (
    ("Automatic", "auto"),
    ("75 microseconds (Americas)", 75.0),
    ("50 microseconds (elsewhere)", 50.0),
    ("Off", None),
)

HEADER_STYLE = """
QLabel { color: #8b98a5; }
QLabel#bandName { color: #e6edf3; font-size: 13px; font-weight: 600; }
QLabel#bandInfo { color: #8b98a5; }
QLabel#stationName { color: #5ad1ff; font-size: 13px; font-weight: 600; }
QLabel#stationText { color: #b6c2cf; }
QLabel#hdBadge {
    color: #0b0e13; background: #5ad1ff; border-radius: 3px;
    padding: 1px 6px; font-weight: 600;
}
QLabel#stereoBadge {
    color: #0b0e13; background: #7ee081; border-radius: 3px;
    padding: 1px 6px; font-weight: 600;
}
"""


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    suffix: str = "",
    decimals: int = 1,
    step: float = 1.0,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


def _button(text: str, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("panelButton")
    button.setCheckable(checkable)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class ListenView(QWidget):
    """Tune, listen, and watch the band."""

    def __init__(
        self,
        engine: Engine,
        level: Level = Level.STANDARD,
        settings: Settings | None = None,
        bookmarks: BookmarkStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self.settings = settings
        self.bookmarks = bookmarks if bookmarks is not None else BookmarkStore()
        self._frames = 0
        self._auto_ranged = False
        # The GainChoice currently shown in the spin box, so the display
        # can follow the engine re-picking without fighting the user.
        self._shown_gain: GainChoice | None = None
        self._started = False
        self._band: bandplan.Band | None = None
        # The last thing the station said about itself, kept so that
        # saving a frequency can name it without reading its own label.
        self._station: RdsState | None = None
        self._manager: FrequencyManager | None = None

        self._build()
        self._restore()
        self.set_level(level)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(int(1000 / REFRESH_HZ))

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setStyleSheet(HEADER_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.frequency = FrequencyDisplay(value_hz=self.engine.center_hz)
        self.frequency.valueChanged.connect(self._tune)
        outer.addWidget(self.frequency)
        outer.addLayout(self._header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._display(), 1)
        body.addWidget(self._controls())
        outer.addLayout(body, 1)

    def _header(self) -> QHBoxLayout:
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
        # Lit from what reached the sound card, not from the pilot: switching
        # on audio noise reduction mixes the two channels together, and a
        # badge still saying STEREO at that point would be a lie the user has
        # no way of catching.
        self.stereo_badge = QLabel("STEREO")
        self.stereo_badge.setObjectName("stereoBadge")
        self.stereo_badge.setVisible(False)
        name_row.addWidget(self.band_name)
        name_row.addWidget(self.hd_badge)
        name_row.addWidget(self.stereo_badge)
        name_row.addStretch(1)
        self.band_info = QLabel("")
        self.band_info.setObjectName("bandInfo")
        self.band_info.setWordWrap(True)
        # What the station says about itself. Hidden rather than blank when
        # there is nothing, so a mode that carries no such data does not leave
        # two empty lines pushing the spectrum down the screen.
        self.station_name = QLabel("")
        self.station_name.setObjectName("stationName")
        self.station_name.setVisible(False)
        self.station_text = QLabel("")
        self.station_text.setObjectName("stationText")
        self.station_text.setWordWrap(True)
        self.station_text.setVisible(False)
        band_box.addLayout(name_row)
        band_box.addWidget(self.band_info)
        band_box.addWidget(self.station_name)
        band_box.addWidget(self.station_text)
        header.addLayout(band_box, 4)
        return header

    def _display(self) -> QWidget:
        display = QWidget()
        layout = QVBoxLayout(display)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
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
        layout.addWidget(self.ribbon)
        layout.addWidget(self._splitter, 1)
        return display

    # -- the control column ------------------------------------------------

    def _controls(self) -> QWidget:
        self.panel = ControlPanel()
        self._build_audio_section()
        self._build_radio_section()
        self._build_recording_section()
        self._build_bookmark_section()
        self._build_display_section()
        self._build_processing_section()
        self._build_correction_section()
        self._build_status_section()
        return self.panel

    def _build_audio_section(self) -> None:
        section = self.panel.section("Audio", Level.SIMPLE)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(self.engine.volume * 100))
        self.volume.valueChanged.connect(lambda v: self.engine.set_volume(v / 100.0))
        section.add("Volume", self.volume)

        self.mute = QCheckBox("Mute")
        self.mute.toggled.connect(self.engine.set_mute)
        section.add_wide(self.mute)

        self.audio_device = QComboBox()
        self.audio_device.addItem("System default", None)
        for index, name in output.output_devices():
            self.audio_device.addItem(name, index)
        self.audio_device.currentIndexChanged.connect(
            lambda: self.engine.set_audio_device(self.audio_device.currentData())
        )
        section.add("Output", self.audio_device, Level.STANDARD)

    def _build_radio_section(self) -> None:
        section = self.panel.section("Radio", Level.STANDARD)

        self.mode = QComboBox()
        for info in demod.mode_table():
            self.mode.addItem(info.label, info.mode)
            self.mode.setItemData(
                self.mode.count() - 1, info.description, Qt.ItemDataRole.ToolTipRole
            )
        self.mode.setCurrentIndex(self.mode.findData(self.engine.mode))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        section.add("Mode", self.mode)

        self.bandwidth = _spin(0.1, 2_500.0, 200.0, " kHz")
        self.bandwidth.valueChanged.connect(
            lambda khz: self.engine.set_bandwidth(khz * 1000.0)
        )
        section.add("Bandwidth", self.bandwidth)

        self.squelch_on = QCheckBox("Squelch")
        self.squelch = _spin(-90.0, 0.0, -40.0, " dBFS")
        self.squelch_on.toggled.connect(self._squelch_changed)
        self.squelch.valueChanged.connect(self._squelch_changed)
        section.add_wide(self.squelch_on)
        section.add("Threshold", self.squelch)

        self.filter_taps = QComboBox()
        for label, taps in (
            ("Normal", 24),
            ("Sharp", 48),
            ("Very sharp", 96),
            ("Soft", 12),
        ):
            self.filter_taps.addItem(label, taps)
        self.filter_taps.currentIndexChanged.connect(
            lambda: self.engine.set_filter_taps(self.filter_taps.currentData())
        )
        self.filter_taps.setToolTip(
            "How steeply the channel filter cuts off at its edges. Sharper "
            "pushes a strong neighbour off a weak channel, at some cost in "
            "processing."
        )
        section.add("Filter edge", self.filter_taps, Level.EXPERT)

        self.gain = _spin(0.0, 49.6, 20.0, " dB")
        if self.engine.gain is not None:
            self.gain.setValue(self.engine.gain.gain_db)
        self.gain.valueChanged.connect(self.engine.set_gain)
        section.add("RF gain", self.gain)

        measure = _button("Measure gain for this band")
        measure.clicked.connect(self.engine.auto_gain)
        measure.setToolTip(
            "Finds the highest gain that does not overload the receiver. The "
            "right setting is about 30 dB apart between the AM band and FM."
        )
        section.add_wide(measure)

        self.sample_rate = QComboBox()
        for rate in SUPPORTED_SAMPLE_RATES:
            label = (
                f"{rate / 1e6:.2f} MS/s"
                if rate >= 1_000_000
                else f"{rate / 1e3:.0f} kS/s"
            )
            self.sample_rate.addItem(label, rate)
        self.sample_rate.setCurrentIndex(self.sample_rate.findData(self.engine.sample_rate))
        self.sample_rate.currentIndexChanged.connect(
            lambda: self.engine.set_sample_rate(self.sample_rate.currentData())
        )
        self.sample_rate.setToolTip(
            "How wide a slice of the radio spectrum to look at. Narrower is "
            "not only a display choice: below about 2 MHz on the dial a wide "
            "window puts the receiver's own oscillator on screen and drowns "
            "out the station."
        )
        section.add("Window", self.sample_rate, Level.EXPERT)

        self.stereo = QCheckBox("Stereo")
        self.stereo.setChecked(self.engine.stereo_enabled)
        self.stereo.toggled.connect(self.engine.set_stereo)
        self.stereo.setToolTip(
            "Broadcast FM carries the difference between the left and right "
            "channels on a second, quieter signal. A weak station may only "
            "manage the mono part of it, and the badge goes out when it does."
        )
        section.add_wide(self.stereo, Level.STANDARD)

        self.rds = QCheckBox("Read station names (RDS)")
        self.rds.setChecked(self.engine.rds_enabled)
        self.rds.toggled.connect(self.engine.set_rds)
        self.rds.setToolTip(
            "Broadcast FM stations send their name and what is playing on a "
            "quiet subcarrier alongside the sound. It takes a few seconds to "
            "arrive and a weak station may never manage it."
        )
        section.add_wide(self.rds, Level.STANDARD)

        self.tuner_agc = QCheckBox("Let the tuner set its own gain")
        self.tuner_agc.toggled.connect(self.engine.set_tuner_agc)
        section.add_wide(self.tuner_agc, Level.EXPERT)

        self.digital_agc = QCheckBox("Receiver digital AGC")
        self.digital_agc.toggled.connect(self.engine.set_digital_agc)
        section.add_wide(self.digital_agc, Level.EXPERT)

        self.ppm = QSpinBox()
        self.ppm.setRange(-200, 200)
        self.ppm.setSuffix(" ppm")
        self.ppm.setValue(self.engine.ppm)
        self.ppm.valueChanged.connect(self.engine.set_ppm)
        section.add("Correction", self.ppm, Level.EXPERT)

        self.calibrate_button = _button("Calibrate on this station")
        self.calibrate_button.clicked.connect(self._calibrate)
        section.add_wide(self.calibrate_button, Level.EXPERT)

        self.bias_tee = QCheckBox("Bias tee (4.5 V on the aerial)")
        self.bias_tee.toggled.connect(self._bias_tee_toggled)
        section.add_wide(self.bias_tee, Level.EXPERT)

    def _build_recording_section(self) -> None:
        section = self.panel.section("Recording", Level.STANDARD)
        self.record_audio = _button("Record audio", checkable=True)
        self.record_audio.toggled.connect(self._record_audio_toggled)
        section.add_wide(self.record_audio)

        self.record_iq = _button("Record raw IQ", checkable=True)
        self.record_iq.toggled.connect(self._record_iq_toggled)
        self.record_iq.setToolTip(
            "Everything the aerial received, for replaying later. Large: "
            "4.8 MB every second at the widest window."
        )
        section.add_wide(self.record_iq, Level.EXPERT)

        self.recording_status = QLabel("")
        self.recording_status.setWordWrap(True)
        section.add_wide(self.recording_status)
        self.recording_status.setVisible(False)

    def _build_bookmark_section(self) -> None:
        section = self.panel.section("Saved frequencies", Level.STANDARD)
        self.save_button = _button("Save this frequency")
        self.save_button.clicked.connect(self._save_current)
        section.add_wide(self.save_button)

        manage = _button("Open my list...")
        manage.clicked.connect(self.open_frequency_manager)
        section.add_wide(manage)

    def _build_display_section(self) -> None:
        section = self.panel.section("Display", Level.STANDARD)

        self.colour_map = QComboBox()
        self.colour_map.addItems(colormaps.NAMES)
        self.colour_map.setCurrentText(colormaps.DEFAULT_NAME)
        self.colour_map.currentTextChanged.connect(self.waterfall.set_colour_map)
        section.add("Colours", self.colour_map)

        self.range_floor = _spin(-140.0, 0.0, -90.0, " dB")
        self.range_ceiling = _spin(-140.0, 0.0, -20.0, " dB")
        self.range_floor.valueChanged.connect(self._range_changed)
        self.range_ceiling.valueChanged.connect(self._range_changed)
        section.add("Floor", self.range_floor)
        section.add("Ceiling", self.range_ceiling)

        fit = _button("Fit to what is on screen")
        fit.clicked.connect(self._fit_now)
        section.add_wide(fit)

        self.peak_hold = QCheckBox("Peak hold")
        self.peak_hold.setChecked(True)
        self.peak_hold.toggled.connect(self.spectrum.set_peak_hold)
        section.add_wide(self.peak_hold)

        self.fft_size = QComboBox()
        for size in FFT_SIZES:
            self.fft_size.addItem(f"{size}", size)
        self.fft_size.setCurrentIndex(self.fft_size.findData(self.engine.fft_size))
        self.fft_size.currentIndexChanged.connect(self._display_changed)
        section.add("Resolution", self.fft_size, Level.EXPERT)

        self.fft_window = QComboBox()
        for name in WINDOWS:
            self.fft_window.addItem(WINDOW_LABELS.get(name, name), name)
        self.fft_window.currentIndexChanged.connect(self._display_changed)
        section.add("Window", self.fft_window, Level.EXPERT)

        self.smoothing = _spin(0.0, 0.95, 0.0, "", decimals=2, step=0.05)
        self.smoothing.valueChanged.connect(self._display_changed)
        section.add("Smoothing", self.smoothing, Level.EXPERT)

        self.waterfall_speed = QComboBox()
        for label, frames in (("Fast", 1), ("Medium", 2), ("Slow", 4)):
            self.waterfall_speed.addItem(label, frames)
        self.waterfall_speed.currentIndexChanged.connect(
            lambda: self.waterfall.set_speed(self.waterfall_speed.currentData())
        )
        section.add("Waterfall", self.waterfall_speed, Level.EXPERT)

        self.split = QSlider(Qt.Orientation.Horizontal)
        self.split.setRange(10, 90)
        self.split.setValue(40)
        self.split.valueChanged.connect(self._split_changed)
        section.add("Split", self.split, Level.EXPERT)

    def _build_processing_section(self) -> None:
        section = self.panel.section("Processing", Level.EXPERT)

        self.deemphasis = QComboBox()
        for label, value in DEEMPHASIS_CHOICES:
            self.deemphasis.addItem(label, value)
        self.deemphasis.currentIndexChanged.connect(
            lambda: self.engine.set_deemphasis(self.deemphasis.currentData())
        )
        self.deemphasis.setToolTip(
            "Broadcast FM boosts the treble before transmitting; this cuts it "
            "back. Skipping it is why a home-made FM receiver sounds thin."
        )
        section.add("De-emphasis", self.deemphasis)

        self.noise_blanker = QCheckBox("Noise blanker")
        self.noise_blanker.setToolTip(
            "Clips short bursts - ignition noise, a thermostat, an electric "
            "fence - before they reach the rest of the receiver."
        )
        self.nb_threshold = _spin(1.5, 20.0, 4.0, "x", decimals=1, step=0.5)
        self.noise_blanker.toggled.connect(self._blanker_changed)
        self.nb_threshold.valueChanged.connect(self._blanker_changed)
        section.add_wide(self.noise_blanker)
        section.add("Above", self.nb_threshold)

        self.if_nr = QCheckBox("Noise reduction (radio)")
        self.if_nr_db = _spin(3.0, 30.0, 12.0, " dB", decimals=0)
        self.if_nr.toggled.connect(self._if_nr_changed)
        self.if_nr_db.valueChanged.connect(self._if_nr_changed)
        section.add_wide(self.if_nr)
        section.add("Depth", self.if_nr_db)

        self.audio_nr = QCheckBox("Noise reduction (audio)")
        self.audio_nr_db = _spin(3.0, 30.0, 12.0, " dB", decimals=0)
        self.audio_nr.toggled.connect(self._audio_nr_changed)
        self.audio_nr_db.valueChanged.connect(self._audio_nr_changed)
        section.add_wide(self.audio_nr)
        section.add("Depth", self.audio_nr_db)

        self.filter_audio = QCheckBox("Filter the audio")
        self.filter_audio.setToolTip(
            "Trims everything outside the range speech lives in - the rumble "
            "below 300 Hz and the hiss above 3 kHz - which is most of what "
            "makes a weak signal tiring to listen to."
        )
        self.filter_audio.toggled.connect(
            lambda on: self.engine.audio.set_audio_filter(on)
        )
        section.add_wide(self.filter_audio)

        self.agc_on = QCheckBox("Automatic volume (AGC)")
        self.agc_on.toggled.connect(self._agc_changed)
        section.add_wide(self.agc_on)

        self.agc_threshold = _spin(-90.0, -10.0, -55.0, " dBFS")
        self.agc_decay = _spin(20.0, 5_000.0, 500.0, " ms", decimals=0, step=50.0)
        self.agc_slope = _spin(0.0, 10.0, 0.0, " dB")
        self.agc_hang = QCheckBox("Hold through pauses")
        self.agc_hang.setChecked(True)
        self.agc_threshold.setToolTip(
            "Below this the gain stops rising, so a silent channel is not "
            "amplified into a roar."
        )
        self.agc_slope.setToolTip(
            "How many dB the output may rise for every 10 dB the input does. "
            "Zero is perfectly flat; a little keeps some of the dynamics."
        )
        for widget in (self.agc_threshold, self.agc_decay, self.agc_slope):
            widget.valueChanged.connect(self._agc_changed)
        self.agc_hang.toggled.connect(self._agc_changed)
        section.add("Threshold", self.agc_threshold)
        section.add("Decay", self.agc_decay)
        section.add("Slope", self.agc_slope)
        section.add_wide(self.agc_hang)

    def _build_correction_section(self) -> None:
        section = self.panel.section("Receiver correction", Level.EXPERT)

        self.dc_removal = QCheckBox("Remove DC offset")
        self.iq_balance = QCheckBox("Correct IQ imbalance")
        self.swap_iq = QCheckBox("Swap I and Q")
        self.iq_balance.setToolTip(
            "Cancels the mirror image the receiver puts on the far side of "
            "centre. Without it a strong station appears twice."
        )
        for box, attribute in (
            (self.dc_removal, "dc_removal"),
            (self.iq_balance, "iq_balance"),
            (self.swap_iq, "swap_iq"),
        ):
            box.toggled.connect(
                lambda on, name=attribute: setattr(self.engine.front, name, on)
            )
            section.add_wide(box)

        self.offset_tuning = _spin(-500.0, 500.0, 0.0, " kHz", decimals=0, step=25.0)
        self.offset_tuning.valueChanged.connect(
            lambda khz: self.engine.set_offset_tuning(khz * 1000.0)
        )
        self.offset_tuning.setToolTip(
            "Parks the tuner to one side and shifts the signal back in "
            "software, so the spike at the middle of the window lands "
            "somewhere harmless instead of on top of what you are hearing."
        )
        section.add("Offset tuning", self.offset_tuning)

        self.imbalance_readout = QLabel("")
        self.imbalance_readout.setWordWrap(True)
        section.add_wide(self.imbalance_readout)

    def _build_status_section(self) -> None:
        section = self.panel.section("Status", Level.EXPERT)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        section.add_wide(self.status)

    # -- levels ------------------------------------------------------------

    def set_level(self, level: Level) -> None:
        self.level = level
        self.panel.set_level(level)
        # The panel shows every row that belongs at the new level, including
        # the ones this view hides for having nothing to say.
        self._refresh_recording()

    # -- remembered settings -----------------------------------------------

    def _restore(self) -> None:
        """Put back the display choices from last time.

        Only the display ones. Nothing here can leave the radio in a state a
        beginner would struggle to get out of, which is exactly why the bias
        tee and the correction switches are not in this list.
        """
        settings = self.settings
        if settings is None:
            return
        for widget, key in (
            (self.colour_map, "colour_map"),
            (self.fft_window, "fft_window"),
        ):
            index = widget.findData(settings[key])
            if index < 0:
                index = widget.findText(str(settings[key]))
            if index >= 0:
                widget.setCurrentIndex(index)
        index = self.fft_size.findData(int(settings["fft_size"]))
        if index >= 0:
            self.fft_size.setCurrentIndex(index)
        index = self.waterfall_speed.findData(int(settings["waterfall_speed"]))
        if index >= 0:
            self.waterfall_speed.setCurrentIndex(index)
        self.smoothing.setValue(float(settings["fft_smoothing"]))
        self.range_floor.setValue(float(settings["range_floor_db"]))
        self.range_ceiling.setValue(float(settings["range_ceiling_db"]))
        self.peak_hold.setChecked(bool(settings["peak_hold"]))
        self.volume.setValue(int(float(settings["volume"]) * 100))
        self.split.setValue(int(float(settings["split_ratio"]) * 100))
        self.ppm.setValue(int(settings["ppm"]))
        self.rds.setChecked(bool(settings["rds"]))
        self.stereo.setChecked(bool(settings["stereo"]))

    def remember(self) -> None:
        """Write the display choices back. Called when the window closes."""
        settings = self.settings
        if settings is None:
            return
        settings.update(
            colour_map=self.colour_map.currentText(),
            fft_size=self.fft_size.currentData(),
            fft_window=self.fft_window.currentData(),
            fft_smoothing=self.smoothing.value(),
            waterfall_speed=self.waterfall_speed.currentData(),
            range_floor_db=self.range_floor.value(),
            range_ceiling_db=self.range_ceiling.value(),
            peak_hold=self.peak_hold.isChecked(),
            volume=self.volume.value() / 100.0,
            split_ratio=self.split.value() / 100.0,
            ppm=self.ppm.value(),
            rds=self.rds.isChecked(),
            stereo=self.stereo.isChecked(),
            frequency_hz=self.frequency.value_hz,
            mode=self.mode.currentData(),
        )
        settings.save()

    # -- control handlers --------------------------------------------------

    def _tune(self, hz: int) -> None:
        self.engine.tune(int(hz))
        self.spectrum.reset_peak_hold()
        self._update_band_label(hz)
        self._apply_band_defaults(hz)
        self._guard_window()
        self._sync_save_button()

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
        # Only on a band change, for the same reason as the mode: a window
        # deliberately widened in Expert mode should survive retuning inside
        # the band it was chosen for.
        self.engine.set_sample_rate(band.sample_rate_hz or DEFAULT_SAMPLE_RATE)
        # And re-measure the gain, because how loud a band is has nothing to
        # do with how loud the last one was. `set_sample_rate` does this too
        # when it actually changes the window, so this covers the other case:
        # FM broadcast to the airband, both 2.4 MS/s and 40 dB apart.
        self.engine.auto_gain()

    def _guard_window(self) -> None:
        """Re-check the window against the frequency we just moved to.

        Unconditional, unlike the band defaults, because this one only ever
        narrows: a window reaching below 0 Hz shows the upconverter's
        oscillator rather than a station, and no user preference makes that
        the right answer.
        """
        self.engine.set_sample_rate(self.engine.sample_rate)
        self._sync_sample_rate()

    def _sync_sample_rate(self) -> None:
        index = self.sample_rate.findData(self.engine.sample_rate)
        if index >= 0 and index != self.sample_rate.currentIndex():
            self.sample_rate.blockSignals(True)
            self.sample_rate.setCurrentIndex(index)
            self.sample_rate.blockSignals(False)

    def show_signal(self, signal: Signal) -> None:
        """Tune to something the scanner found, the way it classified it.

        The classifier already decided what this is and how it should be
        demodulated, so that decision is applied directly rather than
        re-derived from the band plan. It is usually the same answer, but not
        always: an AM signal sitting in an amateur band is what the classifier
        actually saw, and second-guessing it here would be the app disagreeing
        with the reason it just showed the user.
        """
        self.tune_to(
            int(round(signal.frequency_hz)), signal.mode, signal.demod_bandwidth_hz
        )

    def tune_to(self, hz: int, mode: str | None, bandwidth_hz: float | None) -> None:
        """Go to a frequency with a mode and bandwidth already decided.

        Shared by the discovery screen and the frequency manager, which both
        know more about where they are sending the radio than the band plan
        does - a saved marine channel should come back as the mode it was
        saved with, not as whatever the band it sits in usually carries.
        """
        self.frequency.set_value(hz)
        self.engine.tune(hz)
        self.spectrum.reset_peak_hold()
        self._update_band_label(hz)
        band = bandplan.find(hz)
        self._band = band
        self.engine.set_sample_rate(
            (band.sample_rate_hz if band else None) or DEFAULT_SAMPLE_RATE
        )
        self._guard_window()
        # Unconditionally, not only when the window changed: arriving from a
        # scan of the airband onto an airband card never changes the rate, and
        # the gain would still be the one measured on the FM band at startup.
        # De-duplicated inside the engine, so the common case where the rate
        # did change still runs one probe rather than two.
        self.engine.auto_gain()

        if mode is not None:
            index = self.mode.findData(mode)
            if index >= 0:
                self.mode.setCurrentIndex(index)
        if bandwidth_hz is not None:
            self.bandwidth.setValue(bandwidth_hz / 1000.0)
        self._sync_save_button()

    def _tune_from_display(self, hz: float) -> None:
        """Click-to-tune, snapped to the band's channel raster if it has one.

        The frequency the click lands on comes off the plot's x-axis, so it
        can be anywhere the window reaches - and at the bottom of the AM band
        the window legitimately extends below the 500 kHz the dongle can tune
        to. So the readout is set first and the *clamped* value is what the
        radio is asked for, which also keeps the two showing the same number.
        """
        band = bandplan.find(hz)
        snapped = band.snap(hz) if band else hz
        self.frequency.set_value(int(round(snapped)))
        self._tune(self.frequency.value_hz)

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

    def _display_changed(self) -> None:
        self.engine.set_display(
            fft_size=self.fft_size.currentData(),
            window=self.fft_window.currentData(),
            smoothing=self.smoothing.value(),
        )

    def _split_changed(self, value: int) -> None:
        total = max(1, sum(self._splitter.sizes()))
        top = int(total * value / 100)
        self._splitter.setSizes([top, total - top])

    def _fit_now(self) -> None:
        frame = self.engine.latest()
        if frame is not None:
            self.spectrum.auto_range(frame.spectrum_db)
            self._fit_waterfall_range(frame.spectrum_db)

    def _blanker_changed(self) -> None:
        self.engine.front.set_blanker(
            self.noise_blanker.isChecked(), self.nb_threshold.value()
        )

    def _if_nr_changed(self) -> None:
        self.engine.set_if_noise_reduction(
            self.if_nr.isChecked(), self.if_nr_db.value()
        )

    def _audio_nr_changed(self) -> None:
        self.engine.audio.set_noise_reduction(
            self.audio_nr.isChecked(), self.audio_nr_db.value()
        )

    def _agc_changed(self) -> None:
        self.engine.audio.agc_enabled = self.agc_on.isChecked()
        self.engine.audio.configure_agc(
            threshold_dbfs=self.agc_threshold.value(),
            decay_ms=self.agc_decay.value(),
            slope_db=self.agc_slope.value(),
            use_hang=self.agc_hang.isChecked(),
        )

    # -- bias tee ----------------------------------------------------------

    def _bias_tee_toggled(self, enabled: bool) -> None:
        """Guarded, because this one can damage somebody's equipment.

        4.5 V goes down the coax to whatever is plugged in. That is what a
        powered antenna amplifier wants and what a plain antenna, a splitter
        or a signal generator very much does not, so it is the one control in
        the app that asks before it acts.
        """
        if not enabled:
            self.engine.set_bias_tee(False)
            return
        answer = QMessageBox.warning(
            self,
            "Turn on the bias tee?",
            "This puts 4.5 volts down the aerial cable to power a mast-head "
            "amplifier.\n\nIf anything else is connected - a plain aerial, a "
            "splitter, another radio - that voltage can damage it. Only turn "
            "this on if you know the equipment expects it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer is not QMessageBox.StandardButton.Yes:
            self.bias_tee.blockSignals(True)
            self.bias_tee.setChecked(False)
            self.bias_tee.blockSignals(False)
            return
        self.engine.set_bias_tee(True)

    # -- calibration -------------------------------------------------------

    def _calibrate(self) -> None:
        """Measure the crystal error against the station currently tuned.

        The whole method is a conversation rather than a button that silently
        changes a number: it says what it needs before measuring, says what it
        found afterwards, and asks before applying it.
        """
        if self.engine.scanning:
            QMessageBox.information(
                self, "Calibrate", "Wait for the scan to finish, then try again."
            )
            return
        ready = QMessageBox.question(
            self,
            "Calibrate the receiver",
            "Tune to a strong local broadcast station first - its frequency "
            "is far more accurate than this receiver's.\n\nBetterSDR will "
            "listen for half a second and measure how far off the station "
            "lands.\n\nMeasure now?",
        )
        if ready is not QMessageBox.StandardButton.Yes:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            iq = self.engine.capture(1.0)
        finally:
            QApplication.restoreOverrideCursor()

        if iq is None or iq.size < 16_384:
            QMessageBox.warning(
                self, "Calibrate", "The receiver did not deliver enough samples."
            )
            return

        result = calibrate(
            iq, self.engine.sample_rate, self.engine.center_hz, self.engine.ppm
        )
        if not (result.steady and result.trustworthy):
            QMessageBox.information(self, "Calibrate", result.summary)
            return
        apply = QMessageBox.question(
            self,
            "Calibrate",
            f"{result.summary}\n\nSet the correction to {result.ppm} ppm?",
        )
        if apply is QMessageBox.StandardButton.Yes:
            self.ppm.setValue(result.ppm)

    # -- recording ---------------------------------------------------------

    def _record_audio_toggled(self, on: bool) -> None:
        if on:
            self.engine.start_recording(audio=True)
        else:
            self.engine.stop_recording(audio=True, iq=False)

    def _record_iq_toggled(self, on: bool) -> None:
        if on:
            self.engine.start_recording(iq=True)
        else:
            self.engine.stop_recording(audio=False, iq=True)

    def _refresh_recording(self) -> None:
        """Follow the engine rather than assume the buttons are the truth.

        A recording stops itself when it reaches its size limit or the disk
        runs low, and a button left pressed after that would be the app
        claiming to be recording when it is not.
        """
        status = self.engine.recording
        for button, active in (
            (self.record_audio, status.audio_path is not None),
            (self.record_iq, status.iq_path is not None),
        ):
            if button.isChecked() != active:
                button.blockSignals(True)
                button.setChecked(active)
                button.blockSignals(False)

        if status.message:
            text = status.message
        elif status.active:
            parts = []
            if status.audio_path is not None:
                parts.append(f"audio {status.audio_seconds:.0f} s")
            if status.iq_path is not None:
                parts.append(f"IQ {status.iq_seconds:.0f} s")
            text = "Recording: " + ", ".join(parts)
        else:
            text = ""
        self.recording_status.setText(text)
        # Hidden rather than blank: an empty word-wrapping label still takes a
        # row, and a gap under the buttons reads as something missing.
        self.recording_status.setVisible(bool(text) and self.level >= Level.STANDARD)

    # -- bookmarks ---------------------------------------------------------

    def _save_current(self) -> None:
        existing = self.bookmarks.find(self.engine.center_hz)
        if existing is not None:
            self.bookmarks.remove(existing)
            self.bookmarks.save()
            self._sync_save_button()
            return
        band = bandplan.find(self.engine.center_hz)
        # A station that has told us who it is names the bookmark better than
        # the band plan can: "KUOW" against "FM Radio". The callsign comes
        # first because it is the part that will still be true tomorrow - the
        # name field was reading "BBC News" when this was written.
        rds = self._station
        station = "" if rds is None else (rds.callsign or rds.name)
        self.bookmarks.add(
            Bookmark(
                name=station or (self.band_name.text() if band else ""),
                frequency_hz=int(self.engine.center_hz),
                mode=self.mode.currentData(),
                bandwidth_hz=self.bandwidth.value() * 1000.0,
                group=band.name if band else "General",
            )
        )
        self.bookmarks.save()
        self._sync_save_button()
        if self._manager is not None:
            self._manager.refresh()

    def _sync_save_button(self) -> None:
        saved = self.bookmarks.find(self.engine.center_hz) is not None
        self.save_button.setText(
            "Remove from my list" if saved else "Save this frequency"
        )

    def open_frequency_manager(self) -> None:
        if self._manager is None:
            self._manager = FrequencyManager(self.bookmarks, self)
            self._manager.tuneRequested.connect(self._tune_to_bookmark)
        self._manager.refresh()
        self._manager.show()
        self._manager.raise_()

    def _tune_to_bookmark(self, entry: Bookmark) -> None:
        self.tune_to(entry.frequency_hz, entry.mode, entry.bandwidth_hz)

    def _update_band_label(self, hz: float) -> None:
        band = bandplan.find(hz)
        self.band_name.setText(band.name if band else "Unallocated")
        self.band_info.setText(
            band.description if band else "Nothing is normally broadcast here."
        )

    # -- refresh -----------------------------------------------------------

    def start(self) -> None:
        self._update_band_label(self.engine.center_hz)
        if not self._started:
            # Only on the very first start. This runs again every time the
            # user comes back from the discovery screen, and forcing the band
            # defaults each time would quietly throw away both a deliberate
            # mode choice and the one the classifier just made.
            self._started = True
            self._apply_band_defaults(self.engine.center_hz, force=True)
        self._sync_gain()
        self._sync_save_button()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _sync_gain(self) -> None:
        """Show what the engine actually picked, if it has picked again.

        Compared by identity rather than value: `choose_gain` replaces the
        whole `GainChoice`, so this fires exactly when a new measurement lands
        and never while the user is turning the control themselves.
        """
        choice = self.engine.gain
        if choice is None or choice is self._shown_gain:
            return
        self._shown_gain = choice
        self.gain.blockSignals(True)
        self.gain.setValue(choice.gain_db)
        self.gain.blockSignals(False)

    def _tick(self) -> None:
        frame = self.engine.latest()
        if frame is None:
            return
        self._frames += 1
        self._sync_gain()

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

        if self._frames % SLOW_TICK == 0:
            self._update_hd_badge(frame)
            self.stereo_badge.setVisible(frame.stereo)
            self._refresh_recording()
            self._update_status(frame)

    def _update_status(self, frame: DisplayFrame) -> None:
        parts = [
            f"buffer {frame.audio_latency_s * 1000:.0f} ms",
            f"underruns {frame.underruns}",
            f"overruns {frame.ring_overruns}",
        ]
        if self.engine.audio.agc_enabled:
            parts.append(f"AGC {frame.agc_gain_db:+.0f} dB")
        if self.engine.front.noise_blanker:
            parts.append(f"blanked {self.engine.front.blanked_samples}")
        self.status.setText("   ".join(parts))
        self._update_station(frame)
        self.imbalance_readout.setText(
            f"Measured imbalance: {self.engine.front.imbalance.summary}"
            if self.engine.front.iq_balance
            else ""
        )

    def _update_station(self, frame: DisplayFrame) -> None:
        """Show the station's own name, once enough of it has arrived twice.

        Nothing is shown until there is something to show. A station that is
        simply too weak to decode looks exactly like one that sends nothing,
        and inventing a placeholder for either would be claiming knowledge the
        receiver does not have.
        """
        rds = self._station = frame.rds
        name = "" if rds is None else rds.name
        if name:
            traffic = "Traffic" if rds.traffic_announcement else ""
            detail = " - ".join(
                part for part in (rds.pty_name, traffic) if part
            )
            self.station_name.setText(f"{name}   {detail}" if detail else name)
        self.station_name.setVisible(bool(name))
        text = "" if rds is None else rds.text
        self.station_text.setText(text)
        self.station_text.setVisible(bool(text))

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
