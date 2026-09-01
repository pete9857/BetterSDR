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

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import Signal as QtSignal
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

from ..audio import output, repro
from ..core.bookmarks import Bookmark, BookmarkStore
from ..core.calibrate import calibrate
from ..core.device import DEFAULT_SAMPLE_RATE
from ..core.engine import DisplayFrame, Engine
from ..core.frontend import SUPPORTED_SAMPLE_RATES, GainChoice
from ..core.history import History, Visit
from ..core.settings import Settings
from ..decode import hdradio
from ..decode.hdradio import HdProgram, HdState
from ..decode.rds import RdsState
from ..dsp import demod
from ..dsp.features import detect_hd_radio
from ..dsp.psd import WINDOWS
from ..scan import bandplan
from ..scan.classifier import Signal
from . import results
from .freq_manager import FrequencyManager
from .levels import Level
from .widgets import colormaps, viewspan
from .widgets.frequency import FrequencyDisplay
from .widgets.icons import glyph
from .widgets.meter import SignalMeter
from .widgets.pagerlog import PagerLog
from .widgets.panel import ControlPanel
from .widgets.spectrum import BandRibbon, SpectrumWidget
from .widgets.waterfall import WaterfallWidget

REFRESH_HZ = 30
# HD detection is cheap but not free, and a station does not start or stop
# carrying HD between frames. Once a second is plenty, and the same tick
# refreshes the status and recording lines.
SLOW_TICK = REFRESH_HZ

# How many recently played stations the panel offers. The history keeps more;
# a drop-down longer than this is a list to search rather than a shortcut.
RECENT_SHOWN = 12

# Both step buttons are this wide, whatever their label, so that the readout
# between them stays centred in the window.
STEP_BUTTON_WIDTH = 108

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
QLabel#hdStatus { color: #8b98a5; }
QLabel#stereoBadge {
    color: #0b0e13; background: #7ee081; border-radius: 3px;
    padding: 1px 6px; font-weight: 600;
}
QPushButton#stepButton {
    background: #10151c; color: #cbd5e0;
    border: 1px solid #2b323b; border-radius: 4px;
    padding: 8px 6px; font-size: 12px;
}
QPushButton#stepButton:hover { border-color: #5ad1ff; color: #e6edf3; }
QPushButton#stepButton:disabled { color: #3d4650; border-color: #1a2028; }
"""


def band_headline(hz: float, level: Level) -> tuple[str, str]:
    """What the header says about a frequency: the band, and the prose.

    A plain function, like the colour maps and the digit arithmetic, because
    the interesting part is the wording and the level gating rather than the
    labels it ends up in.

    The names themselves - "Channel 16", "Television 7 to 13" - are drawn on
    the ribbon, over the piece of spectrum they describe, where they say
    something a line of text cannot: how wide the channel is and where its
    edges fall. What is left up here is the half that has nowhere else to go,
    which is the explanation. From Standard upwards that also carries the
    regulator's own phrase for the channel, the one somebody would search for
    or find printed on a chart.
    """
    band = bandplan.find(hz)
    if band is None:
        # At Simple, a stretch of dial with nothing to listen to is better
        # left quiet than explained: "licensed to mobile phone networks" is a
        # true answer to a question a beginner did not ask.
        allocation = bandplan.official(hz) if level >= Level.STANDARD else None
        if allocation is None:
            return "Unallocated", "Nothing is normally broadcast here."
        return "Unallocated", allocation.use
    channel = band.channel(hz)
    if channel is None:
        return band.name, band.description
    info = channel.use
    if channel.official and level >= Level.STANDARD:
        info = f"{info} Officially: {channel.official}."
    return band.name, info


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


# What the two Repro-Radio caps offer, in minutes. Fixed choices rather than a
# pair of hours-and-minutes spin boxes: two numbers per cap is four fields in a
# 272 px column, and nobody has ever wanted a recording that runs for 1 h 47 m.
_REPRO_CLIP_LIMITS: tuple[tuple[str, int], ...] = (
    ("5 minutes", 5),
    ("15 minutes", 15),
    ("30 minutes", 30),
    ("1 hour", 60),
    ("2 hours", 120),
    ("4 hours", 240),
)
_REPRO_SESSION_LIMITS: tuple[tuple[str, int], ...] = (
    ("30 minutes", 30),
    ("1 hour", 60),
    ("2 hours", 120),
    ("4 hours", 240),
    ("8 hours", 480),
    ("12 hours", 720),
)


def _duration(seconds: float) -> str:
    """A length of time as somebody would say it, not as a clock shows it."""
    whole = int(max(0.0, seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _button(text: str, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("panelButton")
    button.setCheckable(checkable)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class ListenView(QWidget):
    """Tune, listen, and watch the band."""

    # Somebody clicked the name of a control wanting to know what it means.
    # Passed straight up from the panel and out again without being acted on:
    # this view has no idea the Learn screen exists, the same as it has no
    # idea the Discover screen does.
    helpRequested = QtSignal(str)

    def __init__(
        self,
        engine: Engine,
        level: Level = Level.STANDARD,
        settings: Settings | None = None,
        bookmarks: BookmarkStore | None = None,
        history: History | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self.settings = settings
        self.bookmarks = bookmarks if bookmarks is not None else BookmarkStore()
        self.history = history if history is not None else History()
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
        # The best name anything has given for what is tuned - a callsign
        # from RDS, or the station name off the digital signal - so saving a
        # frequency can use it whichever decoder happened to find it.
        self._station_name = ""
        self._manager: FrequencyManager | None = None
        # Whether this build has an HD Radio decoder at all. Asked once: it
        # is a look at the filesystem, and the answer cannot change while the
        # app is running.
        self._hd_available = hdradio.available()
        # The programme list is kept here rather than read straight off each
        # frame, because restarting the decoder to change subchannel empties
        # it for a few seconds - and a list that vanished the moment it was
        # used would be unusable.
        self._hd_programs: tuple[HdProgram, ...] = ()
        self._hd_labels: list[str] = []
        # What the Discover screen is showing, handed over by the window. The
        # step buttons walk this and nothing else: they are a way of moving
        # through the list the user was just looking at, so they must skip
        # what that list is hiding and follow the order it is in.
        self._results: tuple[Signal, ...] = ()
        # How much of the captured window the three panes are showing, and
        # which frequency they were showing it around. One copy, here,
        # because the spectrum, the waterfall and the ribbon have to agree on
        # it exactly and none of them knows the others exist.
        self._view = viewspan.FULL
        self._view_center_hz = 0.0

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

        outer.addLayout(self._tuner())
        outer.addLayout(self._header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._display(), 1)
        body.addWidget(self._controls())
        outer.addLayout(body, 1)

    def _tuner(self) -> QHBoxLayout:
        """The readout, with a step through the Discover list either side.

        Available at every level, including Simple, and that is the same
        argument as the Recently played section: Simple has no mode control
        and no bandwidth, so somebody who has just been shown a list of
        stations needs a way to walk it that does not involve going back to
        the other screen and clicking Listen eleven times.

        Both buttons are the same fixed width so the digits stay centred in
        the window rather than shifting when one of them changes its label.
        """
        row = QHBoxLayout()
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(8)

        self.previous_found = _button(f"{glyph('left')}  Previous")
        self.previous_found.setObjectName("stepButton")
        self.previous_found.setFixedWidth(STEP_BUTTON_WIDTH)
        self.previous_found.clicked.connect(lambda: self._step_found(-1))

        self.next_found = _button(f"Next  {glyph('right')}")
        self.next_found.setObjectName("stepButton")
        self.next_found.setFixedWidth(STEP_BUTTON_WIDTH)
        self.next_found.clicked.connect(lambda: self._step_found(1))

        self.frequency = FrequencyDisplay(value_hz=self.engine.center_hz)
        self.frequency.valueChanged.connect(self._tune)

        row.addWidget(self.previous_found)
        row.addWidget(self.frequency, 1)
        row.addWidget(self.next_found)
        self._refresh_step_buttons()
        return row

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
        self.ribbon = BandRibbon(level=self.level)
        self.spectrum = SpectrumWidget()
        self.waterfall = WaterfallWidget()
        self.spectrum.tuneRequested.connect(self._tune_from_display)
        self.spectrum.bandwidthChanged.connect(self._set_bandwidth)
        self.waterfall.tuneRequested.connect(self._tune_from_display)
        # Either pane can be the one under the pointer, and both report what
        # the gesture asked for rather than acting on it alone.
        self.spectrum.viewChanged.connect(self._view_changed)
        self.waterfall.viewChanged.connect(self._view_changed)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self.spectrum)
        self._splitter.addWidget(self.waterfall)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 6)
        self._splitter.setHandleWidth(2)
        # Under the splitter rather than in it: the split slider divides the
        # spectrum from the waterfall, and a third pane in there would make
        # that control mean something different every time a pager turned up.
        self.pager = PagerLog(self.level)

        layout.addWidget(self.ribbon)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self.pager)
        return display

    # -- the control column ------------------------------------------------

    def _controls(self) -> QWidget:
        self.panel = ControlPanel()
        # One connection for forty rows. Which of them offer an explanation is
        # declared at each row's own call site, so this never has to be kept
        # in step with a list somewhere else.
        self.panel.helpRequested.connect(self.helpRequested.emit)
        self._build_audio_section()
        self._build_radio_section()
        self._build_recording_section()
        self._build_repro_section()
        self._build_bookmark_section()
        self._build_history_section()
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
        section.add("Volume", self.volume, topic="volume")

        self.mute = QCheckBox("Mute")
        self.mute.toggled.connect(self.engine.set_mute)
        section.add_wide(self.mute)

        # HD Radio lives in Audio rather than in Radio, because that is what
        # it is from where the listener sits: a choice about what comes out
        # of the speakers. Everything it does to the receiver - a window
        # nothing else uses, its own gain, no demodulator at all - is the
        # engine's business, not theirs.
        self.hd = QCheckBox("HD Radio")
        self.hd.setToolTip(
            "Many FM stations also broadcast a digital copy of themselves, "
            "and often one or two extra stations alongside it. It sounds "
            "clearer and takes a few seconds to find. Stations that do not "
            "carry it simply keep playing normally."
        )
        self.hd.toggled.connect(self._hd_toggled)
        section.add_wide(self.hd, Level.SIMPLE, topic="hd-radio")

        self.hd_program = QComboBox()
        self.hd_program.setToolTip(
            "The extra stations this one carries. Switching takes the same "
            "few seconds as turning HD Radio on, because the digital signal "
            "has to be found again."
        )
        self.hd_program.currentIndexChanged.connect(self._hd_program_changed)
        section.add_wide(self.hd_program, Level.SIMPLE, topic="hd-program")

        self.hd_status = QLabel("")
        self.hd_status.setObjectName("hdStatus")
        self.hd_status.setWordWrap(True)
        section.add_wide(self.hd_status, Level.SIMPLE)

        self.audio_device = QComboBox()
        self.audio_device.addItem("System default", None)
        for index, name in output.output_devices():
            self.audio_device.addItem(name, index)
        self.audio_device.currentIndexChanged.connect(
            lambda: self.engine.set_audio_device(self.audio_device.currentData())
        )
        section.add("Output", self.audio_device, Level.STANDARD, topic="audio-device")

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
        section.add("Mode", self.mode, topic="mode")

        self.bandwidth = _spin(0.1, 2_500.0, 200.0, " kHz")
        self.bandwidth.valueChanged.connect(
            lambda khz: self.engine.set_bandwidth(khz * 1000.0)
        )
        section.add("Bandwidth", self.bandwidth, topic="bandwidth")

        self.squelch_on = QCheckBox("Squelch")
        self.squelch = _spin(-90.0, 0.0, -40.0, " dBFS")
        self.squelch_on.toggled.connect(self._squelch_changed)
        self.squelch.valueChanged.connect(self._squelch_changed)
        section.add_wide(self.squelch_on, topic="squelch")
        section.add("Threshold", self.squelch, topic="squelch")

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
        section.add(
            "Filter edge", self.filter_taps, Level.EXPERT, topic="filter-edge"
        )

        self.gain = _spin(0.0, 49.6, 20.0, " dB")
        if self.engine.gain is not None:
            self.gain.setValue(self.engine.gain.gain_db)
        self.gain.valueChanged.connect(self.engine.set_gain)
        section.add("RF gain", self.gain, topic="rf-gain")

        measure = _button("Measure gain for this band")
        measure.clicked.connect(self.engine.auto_gain)
        measure.setToolTip(
            "Finds the highest gain that does not overload the receiver. The "
            "right setting is about 30 dB apart between the AM band and FM."
        )
        section.add_wide(measure, topic="rf-gain")

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
        section.add("Window", self.sample_rate, Level.EXPERT, topic="sample-rate")

        self.stereo = QCheckBox("Stereo")
        self.stereo.setChecked(self.engine.stereo_enabled)
        self.stereo.toggled.connect(self.engine.set_stereo)
        self.stereo.setToolTip(
            "Broadcast FM carries the difference between the left and right "
            "channels on a second, quieter signal. A weak station may only "
            "manage the mono part of it, and the badge goes out when it does."
        )
        section.add_wide(self.stereo, Level.STANDARD, topic="stereo")

        self.stereo_blend = QCheckBox("Fade to mono on weak stations")
        self.stereo_blend.setChecked(self.engine.stereo_blend)
        self.stereo_blend.toggled.connect(self.engine.set_stereo_blend)
        self.stereo_blend.setToolTip(
            "The left-minus-right signal sits in the noisiest part of an FM "
            "broadcast, so on a distant station stereo is mostly hiss. This "
            "fades it out as the station weakens, which trades the stereo "
            "effect for a quieter, steadier sound."
        )
        section.add_wide(self.stereo_blend, Level.EXPERT, topic="stereo-blend")

        self.rds = QCheckBox("Read station names (RDS)")
        self.rds.setChecked(self.engine.rds_enabled)
        self.rds.toggled.connect(self.engine.set_rds)
        self.rds.setToolTip(
            "Broadcast FM stations send their name and what is playing on a "
            "quiet subcarrier alongside the sound. It takes a few seconds to "
            "arrive and a weak station may never manage it."
        )
        section.add_wide(self.rds, Level.STANDARD, topic="rds")

        self.pocsag = QCheckBox("Read pager messages (POCSAG)")
        self.pocsag.setChecked(self.engine.pocsag_enabled)
        self.pocsag.toggled.connect(self.engine.set_pocsag)
        self.pocsag.setToolTip(
            "Pagers are still used by hospitals and factories, and they send "
            "their messages as plain text. When a channel carries them, they "
            "appear in a panel under the waterfall."
        )
        section.add_wide(self.pocsag, Level.STANDARD, topic="pocsag")

        self.tuner_agc = QCheckBox("Let the tuner set its own gain")
        self.tuner_agc.toggled.connect(self.engine.set_tuner_agc)
        section.add_wide(self.tuner_agc, Level.EXPERT, topic="tuner-agc")

        self.digital_agc = QCheckBox("Receiver digital AGC")
        self.digital_agc.toggled.connect(self.engine.set_digital_agc)
        section.add_wide(self.digital_agc, Level.EXPERT, topic="digital-agc")

        self.ppm = QSpinBox()
        self.ppm.setRange(-200, 200)
        self.ppm.setSuffix(" ppm")
        self.ppm.setValue(self.engine.ppm)
        self.ppm.valueChanged.connect(self.engine.set_ppm)
        section.add("Correction", self.ppm, Level.EXPERT, topic="ppm")

        self.calibrate_button = _button("Calibrate on this station")
        self.calibrate_button.clicked.connect(self._calibrate)
        section.add_wide(self.calibrate_button, Level.EXPERT, topic="ppm")

        self.bias_tee = QCheckBox("Bias tee (4.5 V on the aerial)")
        self.bias_tee.toggled.connect(self._bias_tee_toggled)
        section.add_wide(self.bias_tee, Level.EXPERT, topic="bias-tee")

    def _build_recording_section(self) -> None:
        section = self.panel.section("Recording", Level.STANDARD)
        self.record_audio = _button("Record audio", checkable=True)
        self.record_audio.toggled.connect(self._record_audio_toggled)
        section.add_wide(self.record_audio, topic="audio-recording")

        self.record_iq = _button("Record raw IQ", checkable=True)
        self.record_iq.toggled.connect(self._record_iq_toggled)
        self.record_iq.setToolTip(
            "Everything the aerial received, for replaying later. Large: "
            "4.8 MB every second at the widest window."
        )
        section.add_wide(self.record_iq, Level.EXPERT, topic="iq-recording")

        self.recording_status = QLabel("")
        self.recording_status.setWordWrap(True)
        section.add_wide(self.recording_status)
        self.recording_status.setVisible(False)

    def _build_repro_section(self) -> None:
        """Unattended recording, and songs off a broadcast station.

        A section of its own rather than two more rows under Recording,
        because the two buttons there are things you press and let go of and
        this is a thing you leave running. The caps come first and the switch
        last, in the order somebody actually decides them: how long am I
        leaving this, and then, right, go.
        """
        section = self.panel.section("Repro-Radio", Level.STANDARD)

        self.repro_clip_limit = QComboBox()
        for caption, minutes in _REPRO_CLIP_LIMITS:
            self.repro_clip_limit.addItem(caption, minutes)
        self.repro_clip_limit.setToolTip(
            "How long one file may get before the next one is started. A "
            "single overnight file is one nothing will seek inside."
        )
        self.repro_clip_limit.activated.connect(self._repro_settings_changed)
        section.add(
            "Maximum recording time",
            self.repro_clip_limit,
            topic="repro-radio",
        )

        self.repro_session_limit = QComboBox()
        for caption, minutes in _REPRO_SESSION_LIMITS:
            self.repro_session_limit.addItem(caption, minutes)
        self.repro_session_limit.setToolTip(
            "How long the whole session runs before Repro-Radio switches "
            "itself off."
        )
        self.repro_session_limit.activated.connect(self._repro_settings_changed)
        section.add("Stop after", self.repro_session_limit, topic="repro-radio")

        self.repro_hang = QDoubleSpinBox()
        self.repro_hang.setRange(0.0, 30.0)
        self.repro_hang.setSingleStep(0.5)
        self.repro_hang.setDecimals(1)
        self.repro_hang.setSuffix(" s")
        self.repro_hang.setValue(repro.DEFAULT_HANG_S)
        self.repro_hang.setToolTip(
            "How long to keep recording after the signal stops, so a pause "
            "in a conversation does not end up as two files."
        )
        self.repro_hang.valueChanged.connect(self._repro_settings_changed)
        section.add("Hang time", self.repro_hang, Level.EXPERT, topic="repro-radio")

        self.repro_songs = QCheckBox("Save songs separately")
        self.repro_songs.setToolTip(
            "On an FM station that sends song information, save each song as "
            "its own file, named and tagged from what the station transmits."
        )
        self.repro_songs.toggled.connect(self._repro_settings_changed)
        section.add_wide(self.repro_songs, topic="repro-songs")

        self.repro_button = _button("Record everything on this frequency", True)
        self.repro_button.toggled.connect(self._repro_toggled)
        section.add_wide(self.repro_button, topic="repro-radio")

        self.repro_status = QLabel("")
        self.repro_status.setWordWrap(True)
        section.add_wide(self.repro_status)
        self.repro_status.setVisible(False)

    def _build_bookmark_section(self) -> None:
        section = self.panel.section("Saved frequencies", Level.STANDARD)
        self.save_button = _button("Save this frequency")
        self.save_button.clicked.connect(self._save_current)
        section.add_wide(self.save_button, topic="bookmarks")

        # A favourite implies a saved entry, so this saves first where there
        # is nothing to star yet. Two presses to get a station onto the
        # landing screen would be one too many for the thing that is supposed
        # to be the shortcut.
        self.favourite_button = _button("Add to my favourites")
        self.favourite_button.setToolTip(
            "Favourites appear on the Discover screen, so you can go straight "
            "back to them without scanning."
        )
        self.favourite_button.clicked.connect(self._toggle_favourite)
        section.add_wide(self.favourite_button, topic="favourites")

        manage = _button("Open my list...")
        manage.clicked.connect(self.open_frequency_manager)
        section.add_wide(manage, topic="bookmarks")

    def _build_history_section(self) -> None:
        """Where the radio has been - at Simple too, where nothing else is.

        Simple has no mode control, no bandwidth and no gain, so a beginner
        who tunes away from something they were enjoying has no way back to
        it. This is that way back, and it is the one panel section that is
        more use the less of the app somebody understands.
        """
        section = self.panel.section("Recently played", Level.SIMPLE)

        self.back_button = _button("Back to the last station")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self._go_back)
        section.add_wide(self.back_button, topic="recently-played")

        self.recent_list = QComboBox()
        self.recent_list.setToolTip(
            "Everything you have listened to for more than a few seconds."
        )
        self.recent_list.activated.connect(self._recent_chosen)
        section.add_wide(self.recent_list, topic="recently-played")
        self._recent_shown: tuple[tuple[int, str], ...] = ()
        self._refresh_history_controls()

    def _build_display_section(self) -> None:
        section = self.panel.section("Display", Level.STANDARD)

        # The one row in this section that appears at Simple, and the reason
        # is the wheel: somebody who scrolls over the spectrum by accident has
        # zoomed in, and at Simple there would otherwise be nothing on screen
        # to say what happened or how to undo it. It is also plain English and
        # cannot affect what the radio is doing - only what is drawn.
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(0, viewspan.SLIDER_STEPS)
        self.zoom.setValue(0)
        self.zoom.setToolTip(
            "Look at a narrower slice of what the radio is receiving. "
            "You can also scroll the wheel over the spectrum to zoom, drag "
            "sideways to move along it, and click to tune to what you see."
        )
        self.zoom.valueChanged.connect(self._zoom_changed)
        section.add("Zoom", self.zoom, Level.SIMPLE, topic="zoom")

        self.colour_map = QComboBox()
        self.colour_map.addItems(colormaps.NAMES)
        self.colour_map.setCurrentText(colormaps.DEFAULT_NAME)
        self.colour_map.currentTextChanged.connect(self.waterfall.set_colour_map)
        section.add("Colours", self.colour_map, topic="colour-map")

        self.range_floor = _spin(-140.0, 0.0, -90.0, " dB")
        self.range_ceiling = _spin(-140.0, 0.0, -20.0, " dB")
        self.range_floor.valueChanged.connect(self._range_changed)
        self.range_ceiling.valueChanged.connect(self._range_changed)
        section.add("Floor", self.range_floor, topic="display-range")
        section.add("Ceiling", self.range_ceiling, topic="display-range")

        fit = _button("Fit to what is on screen")
        fit.clicked.connect(self._fit_now)
        section.add_wide(fit, topic="display-range")

        self.peak_hold = QCheckBox("Peak hold")
        self.peak_hold.setChecked(True)
        self.peak_hold.toggled.connect(self.spectrum.set_peak_hold)
        section.add_wide(self.peak_hold, topic="peak-hold")

        self.fft_size = QComboBox()
        for size in FFT_SIZES:
            self.fft_size.addItem(f"{size}", size)
        self.fft_size.setCurrentIndex(self.fft_size.findData(self.engine.fft_size))
        self.fft_size.currentIndexChanged.connect(self._display_changed)
        section.add("Resolution", self.fft_size, Level.EXPERT, topic="fft-size")

        self.fft_window = QComboBox()
        for name in WINDOWS:
            self.fft_window.addItem(WINDOW_LABELS.get(name, name), name)
        self.fft_window.currentIndexChanged.connect(self._display_changed)
        section.add("Window", self.fft_window, Level.EXPERT, topic="fft-window")

        self.smoothing = _spin(0.0, 0.95, 0.0, "", decimals=2, step=0.05)
        self.smoothing.valueChanged.connect(self._display_changed)
        section.add("Smoothing", self.smoothing, Level.EXPERT, topic="smoothing")

        self.waterfall_speed = QComboBox()
        for label, frames in (("Fast", 1), ("Medium", 2), ("Slow", 4)):
            self.waterfall_speed.addItem(label, frames)
        self.waterfall_speed.currentIndexChanged.connect(
            lambda: self.waterfall.set_speed(self.waterfall_speed.currentData())
        )
        section.add(
            "Waterfall", self.waterfall_speed, Level.EXPERT, topic="waterfall-speed"
        )

        self.split = QSlider(Qt.Orientation.Horizontal)
        self.split.setRange(10, 90)
        self.split.setValue(40)
        self.split.valueChanged.connect(self._split_changed)
        section.add("Split", self.split, Level.EXPERT, topic="split")

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
        section.add("De-emphasis", self.deemphasis, topic="deemphasis")

        self.noise_blanker = QCheckBox("Noise blanker")
        self.noise_blanker.setToolTip(
            "Clips short bursts - ignition noise, a thermostat, an electric "
            "fence - before they reach the rest of the receiver."
        )
        self.nb_threshold = _spin(1.5, 20.0, 4.0, "x", decimals=1, step=0.5)
        self.noise_blanker.toggled.connect(self._blanker_changed)
        self.nb_threshold.valueChanged.connect(self._blanker_changed)
        section.add_wide(self.noise_blanker, topic="noise-blanker")
        section.add("Above", self.nb_threshold, topic="noise-blanker")

        self.if_nr = QCheckBox("Noise reduction (radio)")
        self.if_nr_db = _spin(3.0, 30.0, 12.0, " dB", decimals=0)
        self.if_nr.toggled.connect(self._if_nr_changed)
        self.if_nr_db.valueChanged.connect(self._if_nr_changed)
        section.add_wide(self.if_nr, topic="if-noise-reduction")
        section.add("Depth", self.if_nr_db, topic="if-noise-reduction")

        self.audio_nr = QCheckBox("Noise reduction (audio)")
        self.audio_nr_db = _spin(3.0, 30.0, 12.0, " dB", decimals=0)
        self.audio_nr.toggled.connect(self._audio_nr_changed)
        self.audio_nr_db.valueChanged.connect(self._audio_nr_changed)
        section.add_wide(self.audio_nr, topic="audio-noise-reduction")
        section.add("Depth", self.audio_nr_db, topic="audio-noise-reduction")

        self.filter_audio = QCheckBox("Filter the audio")
        self.filter_audio.setToolTip(
            "Trims everything outside the range speech lives in - the rumble "
            "below 300 Hz and the hiss above 3 kHz - which is most of what "
            "makes a weak signal tiring to listen to."
        )
        self.filter_audio.toggled.connect(
            lambda on: self.engine.audio.set_audio_filter(on)
        )
        section.add_wide(self.filter_audio, topic="audio-filter")

        self.agc_on = QCheckBox("Automatic volume (AGC)")
        self.agc_on.toggled.connect(self._agc_changed)
        section.add_wide(self.agc_on, topic="audio-agc")

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
        section.add("Threshold", self.agc_threshold, topic="agc-threshold")
        section.add("Decay", self.agc_decay, topic="agc-decay")
        section.add("Slope", self.agc_slope, topic="agc-slope")
        section.add_wide(self.agc_hang, topic="agc-hang")

    def _build_correction_section(self) -> None:
        section = self.panel.section("Receiver correction", Level.EXPERT)

        self.dc_removal = QCheckBox("Remove DC offset")
        self.iq_balance = QCheckBox("Correct IQ imbalance")
        self.swap_iq = QCheckBox("Swap I and Q")
        self.iq_balance.setToolTip(
            "Cancels the mirror image the receiver puts on the far side of "
            "centre. Without it a strong station appears twice."
        )
        for box, attribute, topic in (
            (self.dc_removal, "dc_removal", "dc-removal"),
            (self.iq_balance, "iq_balance", "iq-imbalance"),
            (self.swap_iq, "swap_iq", "swap-iq"),
        ):
            box.toggled.connect(
                lambda on, name=attribute: setattr(self.engine.front, name, on)
            )
            section.add_wide(box, topic=topic)

        self.offset_tuning = _spin(-500.0, 500.0, 0.0, " kHz", decimals=0, step=25.0)
        self.offset_tuning.valueChanged.connect(
            lambda khz: self.engine.set_offset_tuning(khz * 1000.0)
        )
        self.offset_tuning.setToolTip(
            "Parks the tuner to one side and shifts the signal back in "
            "software, so the spike at the middle of the window lands "
            "somewhere harmless instead of on top of what you are hearing."
        )
        section.add("Offset tuning", self.offset_tuning, topic="offset-tuning")

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
        self.pager.set_level(level)
        # Both say more from Standard upwards - the ribbon names the
        # stretches no band covers, the header adds what a channel is
        # officially called - so they are redrawn here rather than waiting
        # for a retune to reword them.
        self.ribbon.set_level(level)
        self._update_band_label(self.engine.center_hz)
        # The panel shows every row that belongs at the new level, including
        # the ones this view hides for having nothing to say.
        self._refresh_recording()
        self._refresh_repro()
        self._refresh_hd(self.engine.latest())

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
        self.pocsag.setChecked(bool(settings["pocsag"]))
        self.stereo.setChecked(bool(settings["stereo"]))
        self.stereo_blend.setChecked(bool(settings["stereo_blend"]))
        self.hd.setChecked(bool(settings["hd"]) and self._hd_available)
        # Repro-Radio's own switch is deliberately absent: see `settings.py`.
        # What it should do once it is switched on is remembered; whether it
        # is running is not.
        self.repro_songs.setChecked(bool(settings["repro_songs"]))
        self.repro_hang.setValue(float(settings["repro_hang_s"]))
        for widget, key in (
            (self.repro_clip_limit, "repro_max_clip_minutes"),
            (self.repro_session_limit, "repro_max_session_minutes"),
        ):
            index = widget.findData(int(settings[key]))
            if index >= 0:
                widget.setCurrentIndex(index)
        self.engine.set_repro_settings(self._repro_settings())

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
            pocsag=self.pocsag.isChecked(),
            stereo=self.stereo.isChecked(),
            stereo_blend=self.stereo_blend.isChecked(),
            hd=self.hd.isChecked(),
            repro_songs=self.repro_songs.isChecked(),
            repro_hang_s=self.repro_hang.value(),
            repro_max_clip_minutes=self.repro_clip_limit.currentData(),
            repro_max_session_minutes=self.repro_session_limit.currentData(),
            frequency_hz=self.frequency.value_hz,
            mode=self.mode.currentData(),
        )
        settings.save()

    # -- control handlers --------------------------------------------------

    def _tune(self, hz: int) -> None:
        self._forget_station()
        self.engine.tune(int(hz))
        self.spectrum.reset_peak_hold()
        self._update_band_label(hz)
        self._apply_band_defaults(hz)
        self._guard_window()
        self._sync_save_button()
        # After the band defaults, not before: they are what decides the mode
        # and bandwidth this visit will be remembered with.
        self._record_visit(hz)
        self._refresh_step_buttons()

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
        self._forget_station()
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
        self._record_visit(hz)
        self._refresh_step_buttons()

    # -- walking the Discover list -----------------------------------------

    def set_results(self, signals: Sequence[Signal]) -> None:
        """Adopt what the Discover screen is showing, cards and order alike.

        Pushed in by the window rather than pulled out of the other view, so
        this screen keeps knowing nothing about that one. An empty list is a
        perfectly ordinary state - nobody has scanned yet - and the buttons
        say so rather than disappearing, because a control that comes and
        goes is harder to find the second time than one that is greyed out.
        """
        self._results = tuple(signals)
        self._refresh_step_buttons()

    def _step_found(self, delta: int) -> None:
        target = results.neighbour(self._results, self.frequency.value_hz, delta)
        if target is not None:
            self.show_signal(target)

    def _refresh_step_buttons(self) -> None:
        """Say where each button goes, from wherever the dial is right now."""
        for button, delta, word in (
            (self.previous_found, -1, "Previous"),
            (self.next_found, 1, "Next"),
        ):
            target = results.neighbour(self._results, self.frequency.value_hz, delta)
            button.setEnabled(target is not None)
            if target is None:
                button.setToolTip(
                    "Scan a band on the Discover screen and these step "
                    "through what it found."
                )
            else:
                button.setToolTip(
                    f"{word} of what Discover found: {target.headline}"
                )

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
            visible = self._visible(frame)
            self.spectrum.auto_range(visible)
            self._fit_waterfall_range(visible)

    def _visible(self, frame: DisplayFrame) -> np.ndarray:
        """The part of the transform that is actually on screen.

        The button says "Fit to what is on screen" and the automatic fit is
        the same measurement, so both have to mean the view rather than the
        window once the two can differ. Zoomed into a quiet corner beside a
        broadcast station, fitting to the whole window sets the ceiling from
        a signal the user cannot see and flattens everything they can.
        """
        spectrum = frame.spectrum_db
        bins = spectrum.size
        if self._view.zoom <= 1.0 or bins == 0 or frame.sample_rate <= 0:
            return spectrum
        low, high = viewspan.span(frame.center_hz, frame.sample_rate, self._view)
        scale = bins / frame.sample_rate
        first = int(np.floor((low - frame.center_hz) * scale)) + bins // 2
        last = int(np.ceil((high - frame.center_hz) * scale)) + bins // 2
        first = min(max(first, 0), bins - 1)
        last = min(max(last, first + 1), bins)
        return spectrum[first:last]

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

    # -- HD Radio ----------------------------------------------------------

    def _hd_toggled(self, on: bool) -> None:
        self.engine.set_hd(on)
        self._refresh_hd(self.engine.latest())

    def _hd_program_changed(self) -> None:
        index = self.hd_program.currentData()
        if index is not None:
            self.engine.set_hd_program(int(index))

    def _forget_station(self) -> None:
        """Drop what the last station said about itself, before leaving it.

        The subchannel list is the part that matters: HD2 on this station has
        nothing to do with HD2 on the next one, and a stale list would offer
        programmes that are not there - which costs twelve seconds of silence
        to find out. The name goes with it because a bookmark saved on the
        new frequency must not be named after the old one, and during an HD
        session the RDS receiver is not running to correct it.
        """
        self._hd_programs = ()
        self._hd_labels = []
        self._station = None
        self._station_name = ""
        # Pager messages belong to a channel, not to the app. Carrying them
        # to the next frequency would put somebody else's traffic under a
        # heading that says it was heard here.
        self.pager.clear()

    def _hd_offered(self) -> bool:
        """Whether the switch is worth showing at all right now.

        Two separate questions. Whether this build has a decoder is answered
        once, at startup. Whether the band could carry HD is answered every
        time the radio moves: NRSC-5 is an FM broadcast standard, so a switch
        offered on the airband would be a control that can only ever fail,
        and offering one of those is worse than not having the feature.
        """
        if not self._hd_available:
            return False
        if self.mode.currentData() != "wfm":
            return False
        band = bandplan.find(self.engine.center_hz)
        return band is not None and band.mode == "wfm"

    def _refresh_hd(self, frame: DisplayFrame | None) -> None:
        """Follow the engine rather than assume the switch is the truth.

        A session ends itself when the station turns out not to carry HD, and
        a switch left on over that would be the app claiming to be playing a
        digital broadcast that was never there.
        """
        offered = self._hd_offered()
        self.hd.setVisible(offered)
        if self.hd.isChecked() != self.engine.hd_enabled:
            self.hd.blockSignals(True)
            self.hd.setChecked(self.engine.hd_enabled)
            self.hd.blockSignals(False)

        state = None if frame is None else frame.hd
        if state is not None and state.programs:
            # Only ever replaced by a longer answer. A restart empties the
            # list for a few seconds and the user is most likely reaching for
            # it exactly then.
            self._hd_programs = state.programs
        self._sync_hd_programs(state)
        # Gated on the switch, not on there being a session: it has to stay
        # put across the restart that changing subchannel costs, and it has
        # to go away when HD is off, where picking an entry would do nothing
        # anybody could see.
        self.hd_program.setVisible(
            offered and self.engine.hd_enabled and len(self._hd_programs) > 1
        )

        self.hd_status.setText(self._hd_status_text(state))
        # Hidden rather than blank: an empty word-wrapping label still takes
        # a row, and a gap under the switch reads as something missing.
        self.hd_status.setVisible(offered and bool(self.hd_status.text()))

    def _sync_hd_programs(self, state: HdState | None) -> None:
        """Rebuild the subchannel list, but only when it has actually changed.

        Rebuilt on every frame it would fight the user, who is trying to open
        it thirty times a second.
        """
        labels = [self._hd_program_label(program) for program in self._hd_programs]
        wanted = self.engine.hd_program
        if labels != self._hd_labels:
            self._hd_labels = labels
            self.hd_program.blockSignals(True)
            self.hd_program.clear()
            for program, label in zip(self._hd_programs, labels, strict=True):
                self.hd_program.addItem(label, program.index)
            self.hd_program.blockSignals(False)
        index = self.hd_program.findData(wanted)
        if index >= 0 and index != self.hd_program.currentIndex():
            self.hd_program.blockSignals(True)
            self.hd_program.setCurrentIndex(index)
            self.hd_program.blockSignals(False)

    @staticmethod
    def _hd_program_label(program: HdProgram) -> str:
        """What a car radio would show, plus whatever the station added.

        nrsc5 prints "None" for a programme that declared no type, which is a
        name for the absence of one rather than a genre - so a programme with
        nothing to say about itself is just "HD2".
        """
        parts = [program.label]
        if program.kind:
            parts.append(program.kind)
        if program.restricted:
            # Subscription-only. Shown rather than hidden, because a listener
            # who picks it and gets silence deserves to have been told why.
            parts.append("subscription only")
        return " - ".join(parts)

    def _hd_status_text(self, state: HdState | None) -> str:
        message = self.engine.hd_message
        if message:
            return message
        if state is None or not state.running:
            return ""
        if state.playing:
            return ""
        return "Finding the HD Radio signal..."

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

    def _repro_settings(self) -> repro.ReproSettings:
        """What the controls currently say, as the engine's own settings."""
        return repro.ReproSettings(
            enabled=self.repro_button.isChecked(),
            songs=self.repro_songs.isChecked(),
            hang_s=float(self.repro_hang.value()),
            max_clip_s=float(self.repro_clip_limit.currentData()) * 60.0,
            max_session_s=float(self.repro_session_limit.currentData()) * 60.0,
        )

    def _repro_settings_changed(self, *_: object) -> None:
        self.engine.set_repro_settings(self._repro_settings())

    def _repro_toggled(self, on: bool) -> None:
        self.engine.set_repro_settings(self._repro_settings())
        if on:
            self.engine.start_repro()
        else:
            self.engine.stop_repro()

    def _refresh_repro(self) -> None:
        """Say what it is doing, and take the button back up if it stopped.

        Repro-Radio ends itself when it reaches its session limit, fills the
        disk or cannot open a file, so the button is driven from what the
        engine reports rather than from what was last clicked - the same rule
        as the Record audio button, and for the same reason: a control
        claiming to be recording when it is not is worse than no control.
        """
        status = self.engine.repro
        if self.repro_button.isChecked() != status.enabled:
            self.repro_button.blockSignals(True)
            self.repro_button.setChecked(status.enabled)
            self.repro_button.blockSignals(False)

        parts: list[str] = []
        if status.enabled:
            if status.recording:
                parts.append(f"Recording {_duration(status.clip_seconds)}")
            else:
                parts.append("Waiting for a signal")
            if status.session_remaining is not None:
                parts.append(f"{_duration(status.session_remaining)} left")
        if status.clips:
            parts.append(f"{status.clips} files")
        if status.songs_enabled:
            if status.song_title:
                parts.append(f"Song: {status.song_title}")
            elif status.enabled:
                parts.append("No song information yet")
            if status.songs_saved:
                parts.append(f"{status.songs_saved} songs kept")
        # A middle dot rather than a hyphen: every song title on this line
        # already contains " - ", and joining with the same thing makes the
        # artist and the count either side of it read as one phrase.
        text = " · ".join(parts)
        if status.message:
            text = f"{text}\n{status.message}" if text else status.message
        self.repro_status.setText(text)
        self.repro_status.setVisible(bool(text) and self.level >= Level.STANDARD)

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
        station = "" if rds is None else rds.callsign
        station = station or self._station_name
        # And where the station has not said, a named channel beats the band
        # it sits in: "Channel 16" against "Marine VHF", which is the group
        # this is about to be filed under anyway.
        channel = band.channel(self.engine.center_hz) if band else None
        default = channel.name if channel is not None else (band.name if band else "")
        self.bookmarks.add(
            Bookmark(
                name=station or default,
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

    def _toggle_favourite(self) -> None:
        """Star this frequency, saving it first if it is not saved yet."""
        existing = self.bookmarks.find(self.engine.center_hz)
        if existing is None:
            self._save_current()
            existing = self.bookmarks.find(self.engine.center_hz)
            if existing is None:
                return
            self.bookmarks.set_favourite(existing, True)
        else:
            self.bookmarks.toggle_favourite(existing)
        self.bookmarks.save()
        self._sync_save_button()
        if self._manager is not None:
            self._manager.refresh()

    def _sync_save_button(self) -> None:
        entry = self.bookmarks.find(self.engine.center_hz)
        self.save_button.setText(
            "Remove from my list" if entry is not None else "Save this frequency"
        )
        self.favourite_button.setText(
            "Remove from my favourites"
            if entry is not None and entry.favourite
            else "Add to my favourites"
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

    # -- recently played ---------------------------------------------------

    def _record_visit(self, hz: float) -> None:
        """Tell the history where the radio has gone.

        Called from every path that tunes, with the mode and bandwidth read
        back off the controls rather than passed in - by this point they have
        been through the band plan, the classifier or a bookmark, and what is
        on screen is what the user will actually hear.
        """
        band = bandplan.find(hz)
        self.history.tune(
            hz,
            mode=self.mode.currentData(),
            bandwidth_hz=self.bandwidth.value() * 1000.0,
            group=band.name if band else "",
        )
        self._refresh_history_controls()

    def _go_back(self) -> None:
        previous = self.history.previous()
        if previous is not None:
            self._tune_to_visit(previous)

    def _recent_chosen(self, index: int) -> None:
        station = self.recent_list.itemData(index)
        if station is not None:
            self.tune_to(station.frequency_hz, station.mode, station.bandwidth_hz)
        # Back to the prompt whatever happened. Leaving the chosen station
        # showing would make the box read as "this is what is playing", which
        # it stops being the moment anyone touches the dial.
        self.recent_list.setCurrentIndex(0)

    def _tune_to_visit(self, visit: Visit) -> None:
        self.tune_to(visit.frequency_hz, visit.mode, visit.bandwidth_hz)

    def _refresh_history_controls(self) -> None:
        previous = self.history.previous()
        self.back_button.setEnabled(previous is not None)
        self.back_button.setText(
            f"Back to {previous.label}" if previous else "Back to the last station"
        )

        recent = self.history.recent(RECENT_SHOWN)
        signature = tuple((s.frequency_hz, s.label) for s in recent)
        if signature == self._recent_shown:
            return
        # Never while the list is open. Rebuilding a combo box under an open
        # popup closes it, and the history changes on its own timer - so this
        # would otherwise snatch the list shut as somebody read it.
        if self.recent_list.view().isVisible():
            return
        self._recent_shown = signature
        self.recent_list.blockSignals(True)
        self.recent_list.clear()
        self.recent_list.addItem(
            "Recently played..." if recent else "Nothing played yet", None
        )
        for station in recent:
            self.recent_list.addItem(station.label, station)
        self.recent_list.setCurrentIndex(0)
        self.recent_list.blockSignals(False)
        self.recent_list.setEnabled(bool(recent))

    def _update_band_label(self, hz: float) -> None:
        name, info = band_headline(hz, self.level)
        self.band_name.setText(name)
        self.band_info.setText(info)

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
        # The radio has been on this frequency all along - the user has only
        # come back to watching it - so this opens a visit rather than
        # continuing one, and a station listened to across two trips to the
        # Discover screen is counted once by `History.tune`'s tolerance test.
        self._record_visit(self.engine.center_hz)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        # Accrue what this stretch was worth, but do not close the visit: the
        # radio is still playing and the user may be back in two seconds.
        # `MAX_TICK_SECONDS` is what stops the time spent away being counted.
        self.history.update()

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
        self.waterfall.set_tuned(frame.center_hz)
        self._sync_view(frame)

        if not self._auto_ranged:
            # One automatic fit on the first real frame, so the display opens
            # showing signal rather than an empty rectangle the user has to
            # calibrate themselves.
            self._auto_ranged = True
            visible = self._visible(frame)
            self.spectrum.auto_range(visible)
            self._fit_waterfall_range(visible)

        if self._frames % SLOW_TICK == 0:
            self._refresh_hd(frame)
            self._update_hd_badge(frame)
            self.stereo_badge.setVisible(frame.stereo)
            self._refresh_recording()
            self._refresh_repro()
            self._update_status(frame)
            self.pager.update_state(frame.pocsag)
            # After `_update_status`, which is where `_station_name` is set
            # from whichever decoder found it. A name read off the air is
            # what turns "94.9 MHz" in the recent list into "KUOW".
            self.history.name(self._station_name)
            self.history.update()
            self._refresh_history_controls()

    def _sync_view(self, frame: DisplayFrame) -> None:
        """Keep the ribbon over the same stretch of dial as the two panes.

        Also re-centres the pan whenever the radio moves. A view offset is a
        fraction of the window, so it survives a retune arithmetically - and
        would then be pointing a zoomed pane several channels away from the
        station the user has just asked to listen to, because the window has
        moved under it. Zoom is a standing preference and stays; where the
        user had panned to was about the old window and does not.
        """
        if frame.center_hz != self._view_center_hz:
            self._view_center_hz = frame.center_hz
            if self._view.offset:
                self._set_view(viewspan.View(self._view.zoom, 0.0))
        low, high = viewspan.span(frame.center_hz, frame.sample_rate, self._view)
        self.ribbon.set_span(low, high, frame.center_hz)

    def _view_changed(self, zoom: float, offset: float) -> None:
        """A pane was dragged or scrolled. Put the others where it is."""
        self._set_view(viewspan.View(zoom, offset))

    def _set_view(self, view: viewspan.View) -> None:
        self._view = viewspan.clamped(*view)
        self.spectrum.set_view(*self._view)
        self.waterfall.set_view(*self._view)
        self._sync_zoom_control()

    def _sync_zoom_control(self) -> None:
        """Show the wheel's answer on the slider, without answering it back."""
        value = viewspan.slider_for_zoom(self._view.zoom)
        if value == self.zoom.value():
            return
        self.zoom.blockSignals(True)
        self.zoom.setValue(value)
        self.zoom.blockSignals(False)

    def _zoom_changed(self) -> None:
        """The slider zooms about the middle of what is on screen.

        Not about the tuned frequency: at 8x on a panned view the user is
        looking somewhere else deliberately, and a slider that dragged the
        view back to the middle of the window would undo the pan every time
        it was touched.
        """
        zoom = viewspan.zoom_for_slider(self.zoom.value())
        self._set_view(viewspan.zoomed(self._view, zoom / self._view.zoom))

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
        if frame.stereo and frame.stereo_blend < 0.99:
            # Only while it is doing something. A permanent "stereo 100%" is
            # a number nobody reads, and its absence is then no longer the
            # signal that something changed.
            parts.append(f"stereo {frame.stereo_blend * 100:.0f}%")
        hd = frame.hd
        if hd is not None:
            # MER and BER are for the carrier as a whole, not for the
            # programme being listened to - nrsc5 filters the title and the
            # bit rate but not these. Below about 9 dB of MER the audio
            # starts dropping out, which makes it the number worth watching.
            parts.append("HD sync" if hd.synced else "HD searching")
            if hd.mer_db is not None:
                parts.append(f"MER {hd.mer_db:.0f} dB")
            if hd.ber is not None:
                parts.append(f"BER {hd.ber:.1e}")
            if hd.bit_rate_kbps is not None:
                parts.append(f"{hd.bit_rate_kbps:.0f} kbps")
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

        The digital signal wins where there is one. It is the same station
        saying the same things over a channel with a checkword on it, and
        during a session the analog receiver is not running at all - so
        anything RDS had to say is minutes old.
        """
        hd = frame.hd
        if hd is not None and hd.running:
            self._show_station(hd.name, hd.label, hd.track or hd.slogan)
            return
        rds = self._station = frame.rds
        name = "" if rds is None else rds.name
        detail = ""
        if rds is not None:
            traffic = "Traffic" if rds.traffic_announcement else ""
            detail = " - ".join(part for part in (rds.pty_name, traffic) if part)
        self._show_station(name, detail, "" if rds is None else rds.text)

    def _show_station(self, name: str, detail: str, text: str) -> None:
        if name:
            self.station_name.setText(f"{name}   {detail}" if detail else name)
            self._station_name = name
        self.station_name.setVisible(bool(name))
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
        hd = frame.hd
        if hd is not None and hd.running:
            # A running session is better evidence than any measurement of
            # the picture: the sidebands are not merely present, they are
            # being decoded. The badge names the subchannel while it is,
            # which is the only place on screen that says which of a
            # station's programmes is playing.
            self.hd_badge.setText(hd.label if hd.playing else "HD")
            self.hd_badge.setToolTip(
                f"Playing the digital {hd.label} programme."
                if hd.playing
                else "Looking for the digital signal."
            )
            self.hd_badge.setVisible(True)
            return
        self.hd_badge.setText("HD")
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
