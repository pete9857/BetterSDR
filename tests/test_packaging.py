"""Tests for what the packaged build carries and how it finds it.

Packaging fails in a way none of the other test files can see. Everything
here runs correctly from a checkout by construction - the repository has all
of it in the right place - and the question is whether it is still true
after PyInstaller has moved it. That question has two halves, and both are
worth a test because both have already been got wrong once:

- **Is the file in the bundle at all?** The spec names them one at a time,
  so adding a data file to the package and forgetting the spec ships an
  application whose band plan, glossary or basemap is simply absent. There
  is no import to fail; the feature is just gone.
- **Does the code look where the bundle put it?** Every one of these files
  is found with `Path(__file__).parent` or a root above the package, and a
  frozen build moves the package under a directory of its own. A lookup
  that reaches for the launcher's folder instead works in a checkout and
  finds nothing once frozen.

The build itself is checked by `tools/build_app.py`, which needs PyInstaller
and a few minutes. Nothing here needs either.
"""

from __future__ import annotations

import re
import struct
import sys
from importlib import import_module
from pathlib import Path

import pytest

from bettersdr import diagnose
from bettersdr.core import native
from bettersdr.decode import hdradio
from bettersdr.scan import bandplan
from bettersdr.ui import assets, basemap, learn

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "BetterSDR.spec"
PACKAGE = ROOT / "bettersdr"

# Matches one line of the spec's DATAS list: the source path's parts, then
# the folder inside the bundle it is collected into.
DATA_ENTRY = re.compile(r"\(str\(ROOT((?: / \"[^\"]+\")+)\), \"([^\"]+)\"\)")
QUOTED = re.compile(r"\"([^\"]+)\"")

# What counts as a file the running program opens rather than imports.
DATA_SUFFIXES = (".yaml", ".bsm", ".ico")


def spec_datas() -> list[tuple[Path, str]]:
    """The spec's DATAS list, as (source path, destination folder)."""
    entries = []
    for parts, destination in DATA_ENTRY.findall(SPEC.read_text(encoding="utf-8")):
        entries.append((ROOT.joinpath(*QUOTED.findall(parts)), destination))
    return entries


# -- the spec ---------------------------------------------------------------


def test_the_spec_names_files_that_exist():
    """A renamed data file leaves the spec pointing at nothing.

    PyInstaller does raise on a missing source, so this only moves the
    failure from a build somebody runs occasionally to a test run that
    happens every time - which is the whole point.
    """
    for source, _ in spec_datas():
        assert source.exists(), f"{SPEC.name} points at a missing {source}"


def test_every_package_data_file_is_bundled():
    """The band plan, the basemap, the glossary and the icon, exhaustively.

    Discovered from the package rather than listed here, because a list in
    a test is one more place to forget the new file.
    """
    found = {
        path
        for path in PACKAGE.rglob("*")
        if path.suffix in DATA_SUFFIXES and path.is_file()
    }
    bundled = {source for source, _ in spec_datas()}
    missing = sorted(str(path.relative_to(ROOT)) for path in found - bundled)
    assert not missing, f"not in {SPEC.name}: {missing}"


def test_package_data_keeps_its_place_inside_the_package():
    """Destination has to be the file's own folder, spelled the same way.

    Every one of these is found at runtime with `Path(__file__).parent`, so
    a file collected to the top of the bundle - or into a folder with a
    different name - is a file the code will never look at.
    """
    for source, destination in spec_datas():
        if PACKAGE not in source.parents:
            continue
        expected = source.parent.relative_to(ROOT).as_posix()
        assert destination == expected, f"{source.name} should go to {expected}"


def test_the_licences_travel_with_the_build():
    """GPL-3 section 4 wants the licence with the program, not near it."""
    destinations = {source.name: dest for source, dest in spec_datas()}
    assert destinations.get("LICENSE") == "."
    assert destinations.get("THIRD-PARTY.md") == "."


def test_the_driver_and_the_decoder_are_bundled_where_they_are_looked_for():
    folders = {destination for _, destination in spec_datas()}
    assert "drivers/win-x64" in folders
    assert "vendor/nrsc5" in folders


# -- finding it again, once frozen ------------------------------------------


def test_the_driver_is_looked_for_inside_the_bundle(monkeypatch, tmp_path):
    """`sys._MEIPASS` is where a frozen build's data actually is.

    Falling back to the executable's own folder is right for a build that
    lays the driver beside the program, and wrong for the one-folder build
    this project ships, where everything is a level down in `_internal`.
    """
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert native.driver_dir() == tmp_path / "drivers" / "win-x64"


def test_the_driver_is_looked_for_beside_the_source_tree():
    """Unfrozen, it is the repository - which is where the DLLs are."""
    assert native.driver_dir() == ROOT / "drivers" / "win-x64"
    assert (native.driver_dir() / "rtlsdr.dll").is_file()


def test_the_decoder_is_looked_for_above_the_package_first():
    """`parents[2]` is the repository here and the bundle once frozen.

    The same expression, because PyInstaller keeps the package's shape
    under its own root - which is the reason the decoder is collected into
    the bundle rather than laid beside the executable.
    """
    roots = hdradio._roots()
    assert len(roots) == 2
    assert roots[0] == ROOT
    assert (roots[0] / "vendor" / "nrsc5").is_dir()
    assert hdradio.executable() is not None


def test_the_decoder_root_follows_the_frozen_executable(monkeypatch, tmp_path):
    """Frozen, the launcher is `sys.executable`; from source it is argv[0].

    `sys.argv[0]` in a frozen build is usually the executable too - but only
    usually. It is whatever the shell passed, which for a shortcut can be a
    bare name. `sys.executable` is the one that is always a real path.
    """
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "BetterSDR.exe"))
    monkeypatch.setattr(sys, "argv", ["BetterSDR"])
    assert hdradio._roots()[1] == tmp_path


@pytest.mark.parametrize(
    ("directory", "name"),
    [
        (bandplan.BANDPLAN_DIR, "us.yaml"),
        (basemap.BASEMAP_DIR, "us.bsm"),
        (learn.LEARN_DIR, "glossary.yaml"),
        (assets.ASSETS_DIR, "bettersdr.ico"),
    ],
)
def test_data_modules_look_inside_their_own_folder(directory, name):
    """The four data directories, and the assumption they all share.

    `Path(__file__).parent` is what makes a frozen build work without any
    frozen-specific code in these modules - so what is being asserted is
    that the file sits beside the module that reads it, rather than
    anywhere else that happens to work from a checkout.
    """
    assert (directory / name).is_file()
    assert directory.is_relative_to(PACKAGE)


# -- the two entry points ---------------------------------------------------


def test_the_entry_shims_import_the_package_rather_than_being_it():
    """A frozen entry script has no package, so it cannot use one.

    `bettersdr/app.py` imports its siblings relatively. Run as a frozen
    `__main__` those resolve to nothing - and, the part worth a test,
    PyInstaller's analysis follows them without complaint and produces an
    executable that fails on the first line the user ever sees.
    """
    for name, target in (
        ("bettersdr_app.py", "from bettersdr.app import main"),
        ("bettersdr_tools.py", "from bettersdr.diagnose import main"),
    ):
        source = (ROOT / "packaging" / name).read_text(encoding="utf-8")
        assert target in source
        assert "from ." not in source


def test_the_spec_builds_from_the_shims_and_not_from_the_modules():
    spec = SPEC.read_text(encoding="utf-8")
    assert 'ROOT / "packaging" / "bettersdr_app.py"' in spec
    assert 'ROOT / "packaging" / "bettersdr_tools.py"' in spec


@pytest.mark.parametrize("command", sorted(diagnose.COMMANDS))
def test_every_console_command_names_something_runnable(command):
    """The tools import their command at the moment it is asked for.

    So a typo in the table is not a failure at start-up or at build time.
    It is a failure the first time somebody in trouble types the command.
    """
    module_name, description = diagnose.COMMANDS[command]
    assert callable(import_module(module_name).main)
    assert description


def test_the_spec_bundles_every_command_the_tools_can_run():
    """A lazily imported command is invisible to PyInstaller's analysis.

    `listen` shipped missing from the first build for exactly this reason,
    and nothing warned: the executable was complete, and one of its three
    commands raised `ModuleNotFoundError`. The spec reads the same table
    this module does rather than restating it.
    """
    spec = SPEC.read_text(encoding="utf-8")
    assert "from bettersdr.diagnose import COMMANDS" in spec
    assert "hiddenimports=HIDDEN_IMPORTS" in spec


# -- the icon ---------------------------------------------------------------


def test_the_icon_exists_and_is_an_icon():
    """Header, directory, and every entry inside the file it claims.

    Read here rather than through Qt, so that the failure names what is
    wrong when the file is truncated or an offset is bad - which is what a
    hand-written ICO writer gets wrong.
    """
    path = assets.icon_path()
    assert path is not None and path.is_file()
    data = path.read_bytes()

    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind) == (0, 1), "not an ICO header"
    assert count >= 4, "too few sizes for the Windows shell to choose from"

    sizes = []
    for index in range(count):
        width, _, _, _, _, _, length, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + 16 * index
        )
        assert offset + length <= len(data), "an entry runs off the end of the file"
        sizes.append(width or 256)

    # 16 for the title bar, 32 for the desktop, 256 for the preview pane.
    assert {16, 32, 256} <= set(sizes)
    assert sizes == sorted(sizes)


def test_the_icon_is_declared_as_package_data():
    """Otherwise a `pip install` of the project ships without it."""
    assert "ui/assets/*.ico" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
