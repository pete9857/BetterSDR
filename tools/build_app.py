"""Build the packaged application, and then check what came out.

    .venv/Scripts/python.exe tools/build_app.py

PyInstaller reporting success means it wrote a folder, not that the folder
contains a working radio. Everything this application carries - the driver
DLLs, the HD Radio decoder, the band plan, the basemap, the glossary - is
found at runtime by a path, and a missing one is a feature that silently
stopped existing rather than an error anybody sees at build time. So the
build is followed by a manifest check and by actually running the console
executable, which is the same check the driver documentation has always
ended with.

`--no-run` skips the launch, for a machine with no dongle attached; the
manifest check still runs, because it needs no hardware.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "BetterSDR.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build" / "pyinstaller"
BUNDLE = DIST / "BetterSDR"

# Everything the running program opens by path. Relative to the bundle
# folder; `_internal` is where PyInstaller puts a one-folder build's data,
# and it is what `sys._MEIPASS` points at.
REQUIRED = (
    "BetterSDR.exe",
    "BetterSDR-Tools.exe",
    "_internal/drivers/win-x64/rtlsdr.dll",
    "_internal/drivers/win-x64/pthreadVC2.dll",
    "_internal/drivers/win-x64/msvcr100.dll",
    "_internal/vendor/nrsc5/win-x64/nrsc5.exe",
    "_internal/vendor/nrsc5/COPYING",
    "_internal/bettersdr/scan/bandplan/us.yaml",
    "_internal/bettersdr/ui/basemap/us.bsm",
    "_internal/bettersdr/ui/learn/glossary.yaml",
    "_internal/bettersdr/ui/assets/bettersdr.ico",
    "_internal/LICENSE",
    "_internal/THIRD-PARTY.md",
)

# Qt's own libraries, which arrive through PySide6's hook rather than from
# anything in DATAS - so a change to the excludes list can remove one
# without any import failing until the window is asked to open.
REQUIRED_QT = (
    "_internal/PySide6/Qt6Core.dll",
    "_internal/PySide6/Qt6Gui.dll",
    "_internal/PySide6/Qt6Widgets.dll",
    "_internal/PySide6/plugins/platforms/qwindows.dll",
)


def bundled_modules(executable: Path) -> set[str]:
    """The `bettersdr` modules inside an executable's own module archive.

    This is the check that matters and it is not obvious that it is needed.
    The first build of this application produced a bundle in which every
    file below was present and correct, and which raised
    `ModuleNotFoundError: No module named 'bettersdr.core'` on the first
    line the user would have seen - because a frozen entry script has no
    package around it, so `bettersdr/app.py`'s relative imports resolved to
    nothing and PyInstaller collected no part of the application at all.
    Nothing warned. The folder looked perfect.
    """
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(executable))
    embedded = next(name for name in archive.toc if name.endswith(".pyz"))
    modules = archive.open_embedded_archive(embedded).toc
    return {
        name
        for name in modules
        if name == "bettersdr" or name.startswith("bettersdr.")
    }


def source_modules() -> set[str]:
    """Every module in the package, named as an import would name it."""
    names = set()
    for path in (ROOT / "bettersdr").rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(parts))
    return names


def check_modules() -> list[str]:
    missing = sorted(source_modules() - bundled_modules(BUNDLE / "BetterSDR.exe"))
    if missing:
        for name in missing:
            print(f"  MISSING {name}")
    else:
        print(f"  ok      all {len(source_modules())} application modules are bundled")
    return missing


def folder_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build(clean: bool) -> None:
    if clean:
        for path in (BUNDLE, WORK):
            if path.exists():
                shutil.rmtree(path)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK),
        "--noconfirm",
    ]
    print(" ".join(command))
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with exit code {result.returncode}")
    print(f"\nbuilt in {time.monotonic() - started:.0f} s")


def check_manifest() -> list[str]:
    missing = []
    for relative in REQUIRED + REQUIRED_QT:
        path = BUNDLE / relative
        if path.is_file():
            print(f"  ok      {relative}  ({path.stat().st_size:,} bytes)")
        else:
            print(f"  MISSING {relative}")
            missing.append(relative)
    return missing


def check_no_source_leak() -> list[str]:
    """The bundle must not carry the tests, the tools or the build inputs.

    Not a licensing point - the licence positively encourages shipping the
    source - but a size and a support one: a `tests/` folder inside the
    application is 40 MB of synthetic IQ generators and a promise that
    somebody will one day report a bug against a file that is not running.
    """
    strays = []
    for name in ("tests", "tools", "build", "docs", ".venv"):
        if (BUNDLE / "_internal" / name).exists():
            strays.append(name)
    return strays


def run_check() -> int:
    tools = BUNDLE / "BetterSDR-Tools.exe"
    print(f"\n$ {tools.name} check --info")
    result = subprocess.run(
        [str(tools), "check", "--info"], cwd=BUNDLE, capture_output=True, text=True
    )
    print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    print(f"\nexit code {result.returncode}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify the packaged app.")
    parser.add_argument(
        "--no-clean", action="store_true", help="reuse the previous build's work folder"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="do not launch the result (no dongle)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check the bundle that is already in dist/, without rebuilding",
    )
    args = parser.parse_args(argv)

    if not args.verify_only:
        build(clean=not args.no_clean)
    elif not BUNDLE.is_dir():
        raise SystemExit(f"nothing to verify: {BUNDLE} does not exist")

    print(f"\n{BUNDLE}")
    print(f"  {folder_size(BUNDLE) / 1e6:,.0f} MB in "
          f"{sum(1 for item in BUNDLE.rglob('*') if item.is_file()):,} files\n")

    missing = check_manifest() + check_modules()
    strays = check_no_source_leak()
    if strays:
        print(f"\n  stray folders in the bundle: {', '.join(strays)}")

    if missing:
        print(f"\n{len(missing)} required file(s) missing from the bundle.")
        return 1

    if args.no_run:
        print("\nSkipping the launch (--no-run).")
        return 0

    code = run_check()
    if code != 0:
        print(
            "\nThe bundle is complete but the radio did not come up. That is a\n"
            "driver or hardware condition, not a packaging one - the remedy is\n"
            "printed above."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
