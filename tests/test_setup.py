"""Tests for the first run, the second run, and the runs after a failure.

Two things are covered here, and they are the same story told from either
end. `tools/setup.py` is what a user runs, and the first thing anybody does
after it fails is run it again - so every state a failure can leave behind
has to be recognised and repaired rather than tripped over. And when the
failure is Windows refusing to load the driver, what step 3 prints has to be
a remedy rather than a traceback, or "the steps above are the fix" is the
setup script pointing at a stack trace.

The blocked-driver path is not hypothetical. `rtlsdr.dll` and
`pthreadVC2.dll` are unsigned, Smart App Control is on by default on a clean
Windows 11 machine, and it refuses both with WinError 4551. It was found on
a second machine rather than this one, which is why the fault is simulated
here: the committed driver loads perfectly on the machine these tests run
on, and the last test in the first half is there to keep it that way.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from bettersdr.core import native

ROOT = Path(__file__).resolve().parent.parent


def _load_setup():
    """Import `tools/setup.py` by path; it is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "bettersdr_setup_script", ROOT / "tools" / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = _load_setup()


# --------------------------------------------------------------------------
# A driver Windows will not load
# --------------------------------------------------------------------------


def _refuse_with_policy(_name):
    error = OSError("[WinError 4551] An application control policy has blocked")
    error.winerror = native._POLICY_BLOCKED
    raise error


def test_a_policy_block_is_named_rather_than_raised_as_an_oserror(monkeypatch):
    monkeypatch.setattr(native.ctypes, "CDLL", _refuse_with_policy)
    with pytest.raises(native.DriverBlockedError) as caught:
        native._open(native.driver_dir() / "rtlsdr.dll")
    assert "rtlsdr.dll" in str(caught.value)


def test_a_blocked_driver_is_still_a_driver_problem_to_every_older_caller():
    """`--info` and the GUI catch `DriverNotFoundError` and must keep working."""
    assert issubclass(native.DriverBlockedError, native.DriverNotFoundError)


def test_any_other_refusal_says_which_file_and_what_windows_said(monkeypatch):
    def refuse(_name):
        raise OSError("[WinError 193] not a valid Win32 application")

    monkeypatch.setattr(native.ctypes, "CDLL", refuse)
    with pytest.raises(native.DriverNotFoundError) as caught:
        native._open(native.driver_dir() / "pthreadVC2.dll")
    message = str(caught.value)
    assert "pthreadVC2.dll" in message
    assert "WinError 193" in message
    assert not isinstance(caught.value, native.DriverBlockedError)


def test_a_failed_load_is_not_cached(monkeypatch):
    """Somebody who clears the mark and runs again must get a real attempt."""
    monkeypatch.setattr(native, "_library", None)
    monkeypatch.setattr(native.ctypes, "CDLL", _refuse_with_policy)
    with pytest.raises(native.DriverNotFoundError):
        native.load()
    assert native._library is None


def test_the_remedy_never_suggests_running_as_administrator():
    """It does not help: code integrity applies to administrators too."""
    remedy = native._blocked_remedy(native.driver_dir() / "rtlsdr.dll")
    assert "administrator does not help" in remedy


def test_a_downloaded_copy_is_told_to_unblock_and_a_cloned_one_is_not(monkeypatch):
    """The two machines differ by the mark, so the two remedies differ too."""
    path = native.driver_dir() / "rtlsdr.dll"

    monkeypatch.setattr(native, "_marked_from_the_web", lambda _p: True)
    assert "Unblock-File" in native._blocked_remedy(path)

    monkeypatch.setattr(native, "_marked_from_the_web", lambda _p: False)
    remedy = native._blocked_remedy(path)
    assert "Unblock-File" not in remedy
    assert "Smart App Control" in remedy


@pytest.mark.skipif(sys.platform != "win32", reason="alternate streams are NTFS")
def test_the_mark_of_the_web_is_read_off_the_file_itself(tmp_path):
    plain = tmp_path / "clone.dll"
    plain.write_bytes(bytes(1))
    assert native._marked_from_the_web(plain) is False

    downloaded = tmp_path / "download.dll"
    downloaded.write_bytes(bytes(1))
    with open(f"{downloaded}:Zone.Identifier", "w", encoding="utf-8") as stream:
        stream.write("[ZoneTransfer]" + chr(10) + "ZoneId=3" + chr(10))
    assert native._marked_from_the_web(downloaded) is True


def test_the_committed_driver_still_loads_here():
    """The guard above must not have broken the path that already worked."""
    assert native.load().path.name == "rtlsdr.dll"


# --------------------------------------------------------------------------
# Running setup again
# --------------------------------------------------------------------------


class _FakeBuilder:
    """Stands in for `venv.EnvBuilder`; making a real one takes seconds."""

    created: list[Path] = []

    def __init__(self, **_kwargs) -> None:
        pass

    def create(self, path) -> None:
        _FakeBuilder.created.append(Path(path))
        scripts = Path(path) / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "python.exe").write_bytes(bytes(1))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """`setup` pointed at a throwaway `.venv`, with nothing really built."""
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(setup, "VENV", venv_dir)
    monkeypatch.setattr(
        setup, "venv_python", lambda root=venv_dir: root / "Scripts" / "python.exe"
    )
    _FakeBuilder.created = []
    monkeypatch.setattr(setup.venv, "EnvBuilder", _FakeBuilder)
    return venv_dir


def _make_usable(venv_dir: Path) -> None:
    (venv_dir / "Scripts").mkdir(parents=True)
    (venv_dir / "Scripts" / "python.exe").write_bytes(bytes(1))


def test_removing_an_environment_that_is_not_there_is_not_an_error(sandbox):
    assert setup.remove_environment() is None


def test_a_read_only_file_does_not_stop_the_rebuild(sandbox):
    """pip leaves them behind; it is a mode bit, not a reason to give up."""
    sandbox.mkdir(parents=True)
    stubborn = sandbox / "locked.txt"
    stubborn.write_text("x", encoding="utf-8")
    stubborn.chmod(0o444)

    assert setup.remove_environment() is None
    assert not sandbox.exists()


def test_a_delete_that_cannot_happen_is_a_remedy_not_a_traceback(
    sandbox, monkeypatch, capsys
):
    sandbox.mkdir(parents=True)

    def refuse(*_args, **_kwargs):
        raise OSError("[WinError 32] The process cannot access the file")

    monkeypatch.setattr(setup.shutil, "rmtree", refuse)
    assert setup.remove_environment() == 1
    assert "Close BetterSDR" in capsys.readouterr().out


def test_an_unfinished_environment_is_thrown_away_rather_than_built_over(sandbox):
    """An interrupted first run leaves a directory with no interpreter in it."""
    (sandbox / "Lib").mkdir(parents=True)
    (sandbox / "pyvenv.cfg").write_text("half a run", encoding="utf-8")

    assert setup.build_environment(recreate=False) is None
    assert _FakeBuilder.created == [sandbox]
    assert not (sandbox / "Lib").exists()


def test_an_environment_below_the_version_floor_is_rebuilt(
    sandbox, monkeypatch, capsys
):
    """It answers the probe, so only the version says it cannot be used."""
    _make_usable(sandbox)
    monkeypatch.setattr(setup, "environment_version", lambda: (3, 11, 9))

    assert setup.build_environment(recreate=False) is None
    assert _FakeBuilder.created == [sandbox]
    assert "3.11.9" in capsys.readouterr().out


def test_an_environment_that_cannot_speak_at_all_is_rebuilt(sandbox, monkeypatch):
    _make_usable(sandbox)
    monkeypatch.setattr(setup, "environment_version", tuple)

    assert setup.build_environment(recreate=False) is None
    assert _FakeBuilder.created == [sandbox]


def test_a_usable_environment_is_left_exactly_where_it_is(sandbox, monkeypatch):
    """The second run has to take about a second, which means building nothing."""
    _make_usable(sandbox)
    monkeypatch.setattr(setup, "environment_version", lambda: (3, 14, 5))

    assert setup.build_environment(recreate=False) is None
    assert _FakeBuilder.created == []


def test_the_version_probe_answers_for_a_real_interpreter(monkeypatch):
    monkeypatch.setattr(setup, "venv_python", lambda: Path(sys.executable))
    assert setup.environment_version() >= setup.MINIMUM_PYTHON


def test_the_version_probe_refuses_a_file_that_is_not_an_interpreter(
    monkeypatch, tmp_path
):
    """`is_file()` is true, and running it raises OSError rather than failing."""
    impostor = tmp_path / "python.exe"
    impostor.write_bytes(b"not an executable")
    monkeypatch.setattr(setup, "venv_python", lambda: impostor)
    assert setup.environment_version() == ()


def test_the_version_probe_asks_for_pip_as_well_as_the_version(monkeypatch):
    """An interrupted install leaves one that runs and cannot install."""
    asked = []

    def record(command, **_kwargs):
        asked.append(command)
        return subprocess.CompletedProcess(command, 1, "", "No module named pip")

    monkeypatch.setattr(setup, "venv_python", lambda: Path(sys.executable))
    monkeypatch.setattr(setup.subprocess, "run", record)
    assert setup.environment_version() == ()
    assert "import pip" in " ".join(asked[0])


def test_the_setup_script_imports_only_the_standard_library():
    """It is what runs before anything is installed. Not a style preference."""
    source = (ROOT / "tools" / "setup.py").read_text(encoding="utf-8")
    imported = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    }
    assert imported <= set(sys.stdlib_module_names)


def test_every_step_survives_being_run_twice(sandbox, monkeypatch, capsys):
    """The whole script, twice over, with only the slow parts stood in for."""
    monkeypatch.setattr(setup, "environment_version", lambda: (3, 14, 5))
    monkeypatch.setattr(setup, "already_installed", lambda: True)
    monkeypatch.setattr(setup, "check_radio", lambda: 0)

    assert setup.main(["--check"]) == 0
    assert setup.main(["--check"]) == 0
    assert _FakeBuilder.created == [sandbox]
    assert "already installed" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The warm path: after the first run this is a launcher, not an installer
# --------------------------------------------------------------------------


@pytest.fixture
def warm(sandbox, monkeypatch):
    """An environment that is already installed, and a launch we can see."""
    _make_usable(sandbox)
    monkeypatch.setattr(setup, "already_installed", lambda: True)
    launches: list[int] = []
    monkeypatch.setattr(setup, "launch", lambda: launches.append(1) or 0)
    return launches


def test_an_installed_environment_goes_straight_to_the_app(warm, capsys):
    """Double-clicking BetterSDR.cmd has to open the radio, not report news."""
    assert setup.main([]) == 0
    assert warm == [1]
    assert _FakeBuilder.created == []

    printed = capsys.readouterr().out
    assert "Starting BetterSDR" in printed
    assert "BetterSDR setup" not in printed


def test_the_warm_path_still_answers_check_and_starts_nothing(warm, monkeypatch):
    monkeypatch.setattr(setup, "check_radio", lambda: 1)
    assert setup.main(["--check"]) == 1
    assert warm == []


def test_asking_to_rebuild_takes_the_long_way_round(warm, monkeypatch, capsys):
    """--update and friends are about the environment, so they must see it."""
    monkeypatch.setattr(setup, "environment_version", lambda: (3, 14, 5))
    monkeypatch.setattr(setup, "install", lambda dev: None)
    monkeypatch.setattr(setup, "check_radio", lambda: 0)

    assert setup.main(["--update", "--no-launch"]) == 0
    assert warm == []
    assert "BetterSDR setup" in capsys.readouterr().out


def test_a_half_installed_environment_is_not_taken_for_a_warm_one(
    sandbox, monkeypatch
):
    """The probe is a real import, so an interrupted install falls through."""
    _make_usable(sandbox)
    monkeypatch.setattr(setup, "already_installed", lambda: False)
    assert setup.installed_and_ready() is False


def test_the_environment_cannot_be_deleted_by_the_python_inside_it(
    sandbox, monkeypatch, capsys
):
    """Windows will not allow it, and the error it gives says nothing."""
    inside = sandbox / "Scripts" / "python.exe"
    _make_usable(sandbox)
    monkeypatch.setattr(setup.sys, "executable", str(inside))

    assert setup.remove_environment() == 1
    assert sandbox.exists()
    assert "inside the environment" in capsys.readouterr().out
