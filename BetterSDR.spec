# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the packaged BetterSDR.

Run it through `tools/build_app.py`, which checks the result rather than
just producing one:

    .venv/Scripts/python.exe tools/build_app.py

One folder, not one file. `--onefile` unpacks the whole bundle to a
temporary directory on every launch, which costs seconds on a build this
size, and a self-extracting executable is what most antivirus heuristics are
looking for. It also matters legally: Qt is here under LGPL-3, and shipping
it as ordinary DLLs beside the program is what keeps a user's right to
replace them straightforward.

Two executables share one bundle. `BetterSDR.exe` is windowed, because a
console flashing up behind a radio looks broken. `BetterSDR-Tools.exe` is
the same bundle with a console attached, and it exists because a windowed
build has nowhere to print a failure that happens before the first window -
see `bettersdr/diagnose.py`.
"""

import sys
from pathlib import Path

from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT = Path(SPECPATH).resolve()
sys.path.insert(0, str(ROOT))

# The console tools import their commands by name at the moment one is
# asked for, which keeps Qt out of a driver check but also puts those
# imports beyond anything static analysis can follow. Reading the table
# rather than restating it is what stops a new command shipping as an
# executable that cannot find it - which is exactly what happened to
# `listen`, and which nothing warned about.
from bettersdr.diagnose import COMMANDS  # noqa: E402

HIDDEN_IMPORTS = sorted(module for module, _ in COMMANDS.values())

NAME = "BetterSDR"
VERSION = "0.1.0"
VERSION_TUPLE = (0, 1, 0, 0)
COMPANY = "Stone Merchant, LLC"
COPYRIGHT = "Copyright (C) 2026 Stone Merchant, LLC. GPL-3.0-or-later."

ICON = ROOT / "bettersdr" / "ui" / "assets" / "bettersdr.ico"

# What the running program reads off disk. Package data keeps its position
# inside the package, because every one of these modules finds its files
# with `Path(__file__).parent` and that expression has to keep working; the
# driver, the decoder and the licences sit at the top of the bundle, which
# is what `native.driver_dir()` and `hdradio.executable()` look for.
DATAS = [
    (str(ROOT / "bettersdr" / "scan" / "bandplan" / "us.yaml"), "bettersdr/scan/bandplan"),
    (str(ROOT / "bettersdr" / "ui" / "basemap" / "us.bsm"), "bettersdr/ui/basemap"),
    (str(ROOT / "bettersdr" / "ui" / "learn" / "glossary.yaml"), "bettersdr/ui/learn"),
    (str(ROOT / "bettersdr" / "ui" / "assets" / "bettersdr.ico"), "bettersdr/ui/assets"),
    (str(ROOT / "drivers" / "win-x64"), "drivers/win-x64"),
    (str(ROOT / "vendor" / "nrsc5"), "vendor/nrsc5"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD-PARTY.md"), "."),
    (str(ROOT / "README.md"), "."),
]

# Nothing here is imported by BetterSDR, and every one of them is pulled in
# by something that is. Qt alone is most of the download, so the modules the
# app has no use for are worth naming: the app imports QtCore, QtGui and
# QtWidgets and nothing else, and pyqtgraph reaches for the rest
# opportunistically.
EXCLUDES = [
    # Qt subsystems this application does not use.
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtHelp",
    "PySide6.QtHttpServer",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    # Plotting and notebook stacks pyqtgraph will use if it finds them.
    "matplotlib",
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "cupy",
    "h5py",
    "pyopengl",
    "OpenGL",
    # Development-time only.
    "pytest",
    "ruff",
    "PyInstaller",
    "tkinter",
    "unittest",
    "pydoc_data",
]


# Both entry points in one Analysis, so the two executables share a single
# module archive and a single copy of Qt, NumPy and SciPy rather than
# doubling the download for a console front end that is a hundred lines
# long. Each EXE below then takes every bootstrap script and its own.
analysis = Analysis(
    [
        str(ROOT / "packaging" / "bettersdr_app.py"),
        str(ROOT / "packaging" / "bettersdr_tools.py"),
    ],
    pathex=[str(ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)


def scripts_for(entry: str):
    """Every bootstrap script, plus the one entry point this build runs."""
    other = {"bettersdr_app", "bettersdr_tools"} - {entry}
    chosen = [item for item in analysis.scripts if item[0] not in other]
    if len(chosen) != len(analysis.scripts) - 1:
        raise SystemExit(f"entry point {entry!r} not found in the analysis")
    return chosen


def version_resource(description: str, filename: str) -> VSVersionInfo:
    """What the Properties dialog and SmartScreen read off the executable.

    An unsigned binary with no version information at all is the shape
    SmartScreen is most suspicious of, and it is what a user checks when
    they are deciding whether to trust the download.
    """
    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=VERSION_TUPLE,
            prodvers=VERSION_TUPLE,
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
        ),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", COMPANY),
                            StringStruct("FileDescription", description),
                            StringStruct("FileVersion", VERSION),
                            StringStruct("InternalName", filename),
                            StringStruct("LegalCopyright", COPYRIGHT),
                            StringStruct("OriginalFilename", f"{filename}.exe"),
                            StringStruct("ProductName", NAME),
                            StringStruct("ProductVersion", VERSION),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )


exe = EXE(
    pyz,
    scripts_for("bettersdr_app"),
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX is the other thing antivirus heuristics look for.
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON),
    version=version_resource("BetterSDR - a beginner-friendly SDR receiver", NAME),
)

tools_exe = EXE(
    pyz,
    scripts_for("bettersdr_tools"),
    [],
    exclude_binaries=True,
    name=f"{NAME}-Tools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(ICON),
    version=version_resource("BetterSDR command-line tools", f"{NAME}-Tools"),
)

COLLECT(
    exe,
    tools_exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NAME,
)
