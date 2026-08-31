"""Application entry point.

Starting the radio can fail for reasons that are the user's to fix - no dongle
plugged in, or one still bound to the Windows TV driver - so failure here
opens the window on an explanation rather than printing a traceback and
exiting. A beginner staring at a closed app has no way forward; a beginner
looking at numbered steps does.

Every command-line option defaults to "whatever was remembered", not to a
constant. That is what lets the app open where it was left without making the
flags useless: an option given explicitly always wins, and one left out falls
back to the settings file, which falls back to a sensible default of its own.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .core import native
from .core.bookmarks import BookmarkStore
from .core.device import device_count
from .core.engine import Engine
from .core.history import History
from .core.settings import Settings
from .ui.assets import declare_app_id, icon_path
from .ui.levels import Level
from .ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bettersdr", description="A beginner-friendly SDR receiver."
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="starting frequency in MHz, or in Hz if large",
    )
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--fft", type=int, default=None)
    parser.add_argument(
        "--level",
        default=None,
        choices=[level.name.lower() for level in Level],
        help="how much of the radio to show",
    )
    parser.add_argument("--audio-device", default=None, help="index or name substring")
    parser.add_argument("--ppm", type=int, default=None, help="crystal correction")
    parser.add_argument(
        "--no-settings",
        action="store_true",
        help="ignore and do not write the saved settings",
    )
    return parser


def parse_frequency(text: str) -> int:
    """Accept 98.5 as MHz and 98500000 as Hz, the way people actually type."""
    value = float(text)
    return int(value if value > 10_000 else value * 1e6)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.no_settings:
        settings.reset()

    level_name = args.level or str(settings["level"])
    frequency = (
        parse_frequency(args.freq)
        if args.freq is not None
        else int(settings["frequency_hz"])
    )
    audio_device: int | str | None = (
        args.audio_device if args.audio_device is not None else settings["audio_device"]
    )
    if isinstance(audio_device, str) and audio_device.isdigit():
        audio_device = int(audio_device)

    declare_app_id()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("BetterSDR")
    app.setApplicationDisplayName("BetterSDR")
    icon = icon_path()
    if icon is not None:
        # On the application rather than the window, so every dialog and the
        # taskbar button get it too.
        app.setWindowIcon(QIcon(str(icon)))

    engine: Engine | None = None
    # A driver that will not load at all raises before any dongle can be
    # counted. That is a reason to open the window on an explanation, the
    # same as every other setup failure here - not to exit with a traceback
    # a beginner has no way forward from.
    try:
        present = device_count() > 0
    except native.DriverNotFoundError:
        present = False
    if present:
        engine = Engine(
            sample_rate=args.rate or 2_400_000,
            fft_size=args.fft or int(settings["fft_size"]),
            audio_device=audio_device,
        )
        engine.ppm = args.ppm if args.ppm is not None else int(settings["ppm"])
        engine.volume = float(settings["volume"])
        engine.audio.volume = engine.volume
        try:
            engine.start(frequency)
        except Exception:  # noqa: BLE001 - any failure here is a setup problem
            engine.stop()
            engine = None

    window = MainWindow(
        engine,
        level=Level[level_name.upper()],
        settings=None if args.no_settings else settings,
        bookmarks=BookmarkStore.open(),
        # Opened whatever `--no-settings` says, the same as the bookmarks:
        # that flag is about the remembered *preferences*, and what was
        # listened to is the user's own list rather than a setting.
        history=History.open(),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
