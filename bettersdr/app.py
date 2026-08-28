"""Application entry point.

Starting the radio can fail for reasons that are the user's to fix - no dongle
plugged in, or one still bound to the Windows TV driver - so failure here
opens the window on an explanation rather than printing a traceback and
exiting. A beginner staring at a closed app has no way forward; a beginner
looking at numbered steps does.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .core.device import DEFAULT_SAMPLE_RATE, device_count
from .core.engine import Engine
from .dsp.psd import DEFAULT_FFT_SIZE
from .ui.levels import Level
from .ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bettersdr", description="A beginner-friendly SDR receiver."
    )
    parser.add_argument(
        "--freq", default="98.5", help="starting frequency in MHz, or in Hz if large"
    )
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--fft", type=int, default=DEFAULT_FFT_SIZE)
    parser.add_argument(
        "--level",
        default="standard",
        choices=[level.name.lower() for level in Level],
        help="how much of the radio to show",
    )
    parser.add_argument("--audio-device", default=None, help="index or name substring")
    return parser


def parse_frequency(text: str) -> int:
    """Accept 98.5 as MHz and 98500000 as Hz, the way people actually type."""
    value = float(text)
    return int(value if value > 10_000 else value * 1e6)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio_device: int | str | None = args.audio_device
    if isinstance(audio_device, str) and audio_device.isdigit():
        audio_device = int(audio_device)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("BetterSDR")

    engine: Engine | None = None
    if device_count() > 0:
        engine = Engine(
            sample_rate=args.rate, fft_size=args.fft, audio_device=audio_device
        )
        try:
            engine.start(parse_frequency(args.freq))
        except Exception:  # noqa: BLE001 - any failure here is a setup problem
            engine.stop()
            engine = None

    window = MainWindow(engine, level=Level[args.level.upper()])
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
