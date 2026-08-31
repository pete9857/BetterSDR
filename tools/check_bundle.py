"""Run the packaged application's own code, out of the packaged bundle.

    .venv/Scripts/python.exe tools/check_bundle.py

This exists because of a fact about this machine that is also a fact about
most new Windows 11 machines: **Smart App Control blocks the executable.**
It is on by default on a clean Windows 11 install, it refuses any binary
that is neither signed by a publisher it recognises nor vouched for by
Microsoft's reputation service, and an unsigned PyInstaller build is
neither. There is no "run anyway" - the process never starts, and the
refusal is written to the CodeIntegrity event log rather than shown as an
error anybody would connect to the application. So `dist/BetterSDR.exe`
cannot be launched here at all, and the packaging cannot be verified the
obvious way.

Everything except the bootloader can still be verified, and this is that.
The bundle's module archive is unpacked beside its data, and the
application is imported and started out of *that* tree by a bare
interpreter with the site directory switched off - so nothing can quietly
come from the developer's virtual environment instead. What it proves:

- every module the application imports is in the bundle, including the ones
  imported lazily, which static analysis cannot see and which fail only when
  a user reaches the feature;
- every compiled dependency loads from the bundle - NumPy, SciPy, Qt,
  PortAudio - rather than from anywhere else on the machine;
- the band plan, the basemap, the glossary, the icon, the driver and the HD
  Radio decoder are all found by the application's own lookup code, from
  where the packaging actually put them;
- the window opens.

What it does not prove: that the PyInstaller bootloader unpacks and starts
correctly, and that Windows will let it. Those need a machine where the
build can run - see `docs/PLAN.md`.
"""

from __future__ import annotations

import argparse
import importlib.util
import marshal
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "BetterSDR"
UNPACKED = ROOT / "build" / "bundle-check"

# PYZ entry kinds, from PyInstaller's archive format. 2 is a retired
# data kind that no longer appears; a namespace package is 3.
MODULE, PACKAGE, NAMESPACE = 0, 1, 3


def unpack(bundle: Path, target: Path) -> int:
    """Lay the bundle's module archive over a copy of its data folder.

    A frozen build keeps its Python modules in an archive inside the
    executable and its data in `_internal`, and the running program sees
    them as one tree. Recreating that tree on disk is what lets an ordinary
    interpreter import out of it - and it has to be a copy, because the
    modules land beside the data files those same modules read with
    `Path(__file__).parent`.
    """
    from PyInstaller.archive.readers import CArchiveReader

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(bundle / "_internal", target)

    archive = CArchiveReader(str(bundle / "BetterSDR.exe"))
    name = next(entry for entry in archive.toc if entry.endswith(".pyz"))
    modules = archive.open_embedded_archive(name)

    written = 0
    for module, (kind, _, _) in modules.toc.items():
        parts = module.split(".")
        if kind == NAMESPACE:
            # A namespace package is a directory and the absence of a file.
            target.joinpath(*parts).mkdir(parents=True, exist_ok=True)
            continue
        if kind == PACKAGE:
            path = target.joinpath(*parts, "__init__.pyc")
        else:
            path = target.joinpath(*parts[:-1], parts[-1] + ".pyc")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        # A .pyc is the interpreter's magic number, three unused words in
        # this position, and the marshalled code object.
        path.write_bytes(
            importlib.util.MAGIC_NUMBER
            + struct.pack("<III", 0, 0, 0)
            + marshal.dumps(modules.extract(module))
        )
        written += 1
    return written


BOOTSTRAP = '''
"""Started by tools/check_bundle.py inside the unpacked bundle."""
import os, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.frozen = True
sys._MEIPASS = str(root)
# The bundle's own standard library first, then the bundle, and nothing
# else: `-S` kept site-packages off the path, so anything that imports had
# to come from here.
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "base_library.zip"))
os.add_dll_directory(str(root))

failures = []


def check(label, function):
    try:
        detail = function()
    except Exception as error:  # noqa: BLE001 - reporting is the whole job
        failures.append(f"{label}: {type(error).__name__}: {error}")
        print(f"  FAIL  {label}: {type(error).__name__}: {error}")
    else:
        print(f"  ok    {label}" + (f": {detail}" if detail else ""))


def every_module():
    """Import the whole application, one module at a time."""
    import importlib, pkgutil
    import bettersdr

    if not Path(bettersdr.__file__).is_relative_to(root):
        raise RuntimeError(f"imported from outside the bundle: {bettersdr.__file__}")
    names = [
        name
        for _, name, _ in pkgutil.walk_packages(bettersdr.__path__, "bettersdr.")
    ]
    for name in names:
        importlib.import_module(name)
    return f"{len(names) + 1} modules"


def dependencies():
    """Every compiled dependency, and where it came from."""
    import numpy, pyqtgraph, PySide6, scipy, sounddevice, yaml

    outside = [
        module.__name__
        for module in (numpy, scipy, PySide6, pyqtgraph, sounddevice, yaml)
        if not Path(module.__file__).is_relative_to(root)
    ]
    if outside:
        raise RuntimeError(f"came from outside the bundle: {', '.join(outside)}")
    return (
        f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
        f"Qt {PySide6.__version__}"
    )


def driver():
    from bettersdr.core import native

    path = native.driver_dir() / "rtlsdr.dll"
    if not path.is_file():
        raise RuntimeError(f"no driver at {path}")
    return str(path.relative_to(root))


def decoder():
    from bettersdr.decode import hdradio

    path = hdradio.executable()
    if path is None:
        raise RuntimeError("the HD Radio decoder is not in the bundle")
    return str(Path(path).relative_to(root))


def band_plan():
    from bettersdr.scan import bandplan

    bands = bandplan.load()
    return f"{len(bands)} bands, {len(bandplan.allocations())} allocations"


def base_map():
    from bettersdr.ui import basemap

    drawn = basemap.load()
    lines = sum(len(layer.lines) for layer in drawn.layers.values())
    return f"{len(drawn.layers)} layers, {lines} lines, {len(drawn.cities)} places"


def glossary():
    from bettersdr.ui import learn

    categories = learn.load()
    problems = learn.check()
    if problems:
        raise RuntimeError(problems[0])
    articles = sum(len(category.articles) for category in categories)
    return f"{len(categories)} categories, {articles} articles"


def icon():
    from bettersdr.ui import assets

    path = assets.icon_path()
    if path is None:
        raise RuntimeError("no application icon in the bundle")
    return f"{path.stat().st_size:,} bytes"


def audio():
    """PortAudio has to load, or the radio is silent and says nothing."""
    import sounddevice

    return f"{len(sounddevice.query_devices())} audio devices"


def window():
    """Build the real window, with no radio behind it.

    Which is the state a first-time user with no dongle plugged in sees,
    and the one that most needs to be an explanation rather than a crash.
    """
    from PySide6.QtWidgets import QApplication

    from bettersdr.core.bookmarks import BookmarkStore
    from bettersdr.core.history import History
    from bettersdr.ui.assets import icon_path
    from bettersdr.ui.levels import Level
    from bettersdr.ui.main_window import MainWindow

    application = QApplication(sys.argv[:1])
    from PySide6.QtGui import QIcon

    application.setWindowIcon(QIcon(str(icon_path())))
    main = MainWindow(
        None,
        level=Level.STANDARD,
        settings=None,
        bookmarks=BookmarkStore.open(),
        history=History.open(),
    )
    main.show()
    application.processEvents()
    size = main.size()
    wearing = "with its icon" if not application.windowIcon().isNull() else "NO ICON"
    main.close()
    return f"{size.width()} x {size.height()}, {wearing}"


for label, function in (
    ("the whole application imports", every_module),
    ("dependencies load from the bundle", dependencies),
    ("the RTL-SDR driver is found", driver),
    ("the HD Radio decoder is found", decoder),
    ("the band plan loads", band_plan),
    ("the basemap loads", base_map),
    ("the glossary loads", glossary),
    ("the application icon is found", icon),
    ("the sound card is reachable", audio),
    ("the window opens", window),
):
    check(label, function)

print()
sys.exit(1 if failures else 0)
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the unpacked bundle in build/ for poking at",
    )
    args = parser.parse_args(argv)

    if not (BUNDLE / "BetterSDR.exe").is_file():
        raise SystemExit(f"no bundle at {BUNDLE}; run tools/build_app.py first")

    print(f"unpacking {BUNDLE.name} into {UNPACKED.relative_to(ROOT)} ...")
    written = unpack(BUNDLE, UNPACKED)
    print(f"  {written:,} modules laid over the bundle's data\n")

    script = UNPACKED.parent / "bundle_check_bootstrap.py"
    script.write_text(BOOTSTRAP, encoding="utf-8")

    # A settings directory of its own: a check must not touch what the user
    # has remembered, and `Settings`, the bookmarks and the history all take
    # their location from APPDATA.
    environment = dict(os.environ)
    scratch = tempfile.mkdtemp(prefix="bettersdr-check-")
    environment["APPDATA"] = scratch
    environment["QT_QPA_PLATFORM"] = "offscreen"

    print("running the application out of the bundle:")
    result = subprocess.run(
        # -S so site-packages is not on the path and nothing can come from
        # the developer's environment; -E so the environment cannot put it
        # back. The interpreter is the only thing borrowed from outside.
        [sys.executable, "-S", "-E", str(script), str(UNPACKED)],
        env=environment,
        cwd=UNPACKED,
    )

    shutil.rmtree(scratch, ignore_errors=True)
    if not args.keep:
        shutil.rmtree(UNPACKED, ignore_errors=True)
        script.unlink(missing_ok=True)

    if result.returncode == 0:
        print("The bundle is complete and the application runs out of it.")
        print(
            "Not covered: the PyInstaller bootloader, and whether Windows will\n"
            "let an unsigned executable start at all - see this file's notes on\n"
            "Smart App Control."
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
