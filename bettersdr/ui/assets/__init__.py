"""The application's own artwork, and how a running build finds it.

One file so far - the Windows icon, compiled from `BetterSDRLogo.svg` by
`tools/make_icon.py` and committed, the same bargain as the basemap and the
band plan: pay the size once, and the running program parses no SVG and
depends on nothing it cannot see.

`Path(__file__).parent` is deliberate rather than incidental. A frozen build
puts the package under its own root, so the same expression finds the file
in a checkout and in the packaged application - which is why the icon is
package data and not a loose file beside the executable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent

ICON_FILE = ASSETS_DIR / "bettersdr.ico"

# What Windows groups taskbar buttons by. Without one, a Python process is
# grouped under python.exe and shows its icon rather than ours - which in a
# packaged build is nobody's icon at all.
APP_ID = "StoneMerchant.BetterSDR"


def icon_path() -> Path | None:
    """The application icon, or None if this build has none."""
    return ICON_FILE if ICON_FILE.is_file() else None


def declare_app_id(app_id: str = APP_ID) -> None:
    """Tell Windows this process is its own application, not Python's.

    Best effort by design: it is cosmetic, it is Windows-only, and a shell
    that declines to answer is not a reason for the radio not to start.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:  # noqa: BLE001 - cosmetic; never worth failing over
        pass
