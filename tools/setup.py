"""Set BetterSDR up and start it, in one command.

    py tools/setup.py

This is not a setuptools script despite the name - nothing here is invoked
by pip, and `pyproject.toml` is what describes the package. It is the
front door: it makes the virtual environment, installs what the radio
needs, checks the dongle, and opens the app. Running it again later is how
you start the app, and the second run skips everything already done.

    py tools/setup.py --check       stop after the driver check
    py tools/setup.py --update      reinstall the dependencies first
    py tools/setup.py --recreate    throw the environment away and rebuild
    py tools/setup.py --dev         install the test and lint tools too

Why this exists rather than a downloadable .exe: Smart App Control is on by
default on a clean Windows 11 machine, and it refuses to start any program
that is neither signed by a publisher it recognises nor already known to
Microsoft's reputation service. A freshly built, unsigned application is
neither, and the refusal is absolute - there is no "run anyway" on the
dialog, and nothing in the event that reaches the user explains it. A
Python interpreter, by contrast, is signed and recognised, and everything
this installs arrives as an ordinary package. See `docs/PLAN.md`.

Standard library only, deliberately: this is the script that runs *before*
anything is installed, so it cannot import anything that has to be.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"

# The floor `pyproject.toml` states. Checked here as well because the
# message pip gives for a too-old interpreter is not one a beginner can act
# on, and this is the first thing that would go wrong.
MINIMUM_PYTHON = (3, 12)

BAR = "-" * 68


def say(message: str = "") -> None:
    print(message, flush=True)


def step(number: int, total: int, message: str) -> None:
    say(f"\n[{number}/{total}] {message}")


def fail(problem: str, remedy: str) -> int:
    """A problem the user can act on, in the same voice as the doctor."""
    say(f"\n{BAR}\n{problem}\n\n{remedy}\n{BAR}")
    return 1


def venv_python(root: Path = VENV) -> Path:
    """The interpreter inside the environment, on either platform."""
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def run(command: list[str], quiet: bool = False) -> int:
    """Run a command, showing what it is. Output goes straight through."""
    if not quiet:
        say(f"    $ {' '.join(str(part) for part in command)}")
    return subprocess.run(command, cwd=ROOT).returncode


def check_interpreter() -> int | None:
    if sys.version_info < MINIMUM_PYTHON:
        wanted = ".".join(str(part) for part in MINIMUM_PYTHON)
        running = ".".join(str(part) for part in sys.version_info[:3])
        return fail(
            f"BetterSDR needs Python {wanted} or newer, and this is {running}.",
            "Install a current Python from https://www.python.org/downloads/\n"
            "and run this command again. Tick 'Add python.exe to PATH' in the\n"
            "installer.",
        )
    return None


def check_checkout() -> int | None:
    if (ROOT / "pyproject.toml").is_file() and (ROOT / "bettersdr").is_dir():
        return None
    return fail(
        f"This does not look like a BetterSDR checkout: {ROOT}",
        "Run this from inside the folder you cloned, for example\n"
        "    git clone https://github.com/pete9857/BetterSDR\n"
        "    cd BetterSDR\n"
        "    py tools/setup.py",
    )


def remove_environment() -> int | None:
    """Delete `.venv`, or explain why it could not be deleted.

    Every path that rebuilds goes through here. On Windows the usual reason
    a delete fails is that something is still holding the environment open -
    BetterSDR itself, or a shell sitting inside it - and a traceback at that
    point tells the user nothing they can act on. pip also leaves read-only
    files behind, which is a mode bit rather than a real problem, so those
    are cleared and retried.
    """

    if not VENV.exists():
        return None

    def clear_readonly(function, path, _exception) -> None:
        Path(path).chmod(0o700)
        function(path)

    try:
        shutil.rmtree(VENV, onexc=clear_readonly)
    except OSError as error:
        return fail(
            f"The environment in {VENV} could not be removed: {error}",
            "Close BetterSDR, and any terminal that is using the environment,\n"
            "then run this command again. If that does not help, delete the\n"
            f"{VENV.name} folder yourself and run it again - nothing in there\n"
            "is yours, and it is rebuilt from scratch.",
        )
    return None


def environment_version() -> tuple[int, ...]:
    """The Python version inside `.venv`, or `()` if it is not usable.

    Three things are asked at once, because all three have to be true before
    the environment can be trusted and none of them says so on its own: the
    interpreter is still there and runs, it meets the version floor, and pip
    survived. An interrupted first run leaves an environment that answers the
    first question and fails the third, and the message pip gives for that is
    not one a beginner can act on.
    """
    try:
        probe = subprocess.run(
            [
                str(venv_python()),
                "-c",
                "import pip, sys;"
                " print('.'.join(str(n) for n in sys.version_info[:3]))",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        # A file that is there and is not an interpreter - a truncated
        # download, or a leftover from an interrupted run.
        return ()
    if probe.returncode != 0:
        return ()
    try:
        return tuple(int(part) for part in probe.stdout.strip().split("."))
    except ValueError:
        return ()


def build_environment(recreate: bool) -> int | None:
    """Make `.venv`, or satisfy ourselves that the one there is usable.

    Safe to run any number of times, and that is the point rather than a
    convenience: the first thing anybody does after a failed setup is run it
    again, so every state a failure can leave behind - a half-made
    environment, one built by an interpreter that has since been upgraded
    away, one whose files are read-only - has to be recognised here and
    repaired rather than tripped over.
    """
    if recreate and VENV.exists():
        say(f"    removing {VENV.name} and starting again")
        code = remove_environment()
        if code is not None:
            return code

    if venv_python().is_file():
        version = environment_version()
        if version >= MINIMUM_PYTHON:
            shown = ".".join(str(part) for part in version)
            say(f"    using the environment already in {VENV.name}"
                f" (Python {shown})")
            return None
        if version:
            shown = ".".join(str(part) for part in version)
            wanted = ".".join(str(part) for part in MINIMUM_PYTHON)
            say(f"    {VENV.name} was built with Python {shown} and BetterSDR"
                f" needs {wanted}; rebuilding it")
        else:
            say(f"    {VENV.name} is there but is not usable; rebuilding it")
        code = remove_environment()
        if code is not None:
            return code
    elif VENV.exists():
        # A directory with no interpreter in it is an interrupted run. It
        # cannot be repaired in place, and building over it is how a broken
        # environment survives every later attempt to fix it.
        say(f"    {VENV.name} is unfinished; starting it again")
        code = remove_environment()
        if code is not None:
            return code

    say(f"    creating {VENV.name} - this takes a few seconds")
    try:
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(VENV)
    except Exception as error:  # noqa: BLE001 - the message is the product
        return fail(
            f"The virtual environment could not be created: {error}",
            "This usually means the Python installation is missing its 'venv'\n"
            "component. Reinstalling Python from python.org fixes it.",
        )
    return None


def already_installed() -> bool:
    """Is the package importable in the environment as it stands?

    This is what makes the second run instant. It is deliberately a real
    import rather than a check for a marker file: a half-finished install
    leaves the marker and not the package.
    """
    probe = subprocess.run(
        [str(venv_python()), "-c", "import bettersdr, numpy, scipy, PySide6"],
        capture_output=True,
    )
    return probe.returncode == 0


def install(dev: bool) -> int | None:
    extras = "[dev]" if dev else ""
    say("    downloading and installing - the first time takes a minute or two")
    if run([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        say("    (could not update pip; carrying on with the one that is there)")
    # Editable, and not as a convenience: the driver DLLs and the HD Radio
    # decoder live beside the package rather than inside it, and are found
    # by walking up from the package's own file. A copy in site-packages
    # would have nothing above it.
    if run([str(venv_python()), "-m", "pip", "install", "-e", f".{extras}"]) != 0:
        return fail(
            "The dependencies could not be installed.",
            "The usual cause is no internet connection, or a firewall blocking\n"
            "pip. The messages above say which. If you are behind a proxy, set\n"
            "HTTPS_PROXY and run this again.",
        )
    return None


def check_radio() -> int:
    """The driver and dongle check, printed where the user can read it."""
    return run([str(venv_python()), "-m", "bettersdr.core.device", "--info"], quiet=True)


def launch() -> int:
    say("    starting BetterSDR - close its window to come back here")
    return run([str(venv_python()), "-m", "bettersdr.app"], quiet=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py tools/setup.py",
        description="Install BetterSDR into .venv and start it.",
    )
    parser.add_argument(
        "--check", action="store_true", help="stop after the driver check"
    )
    parser.add_argument(
        "--no-launch", action="store_true", help="set everything up but do not start"
    )
    parser.add_argument(
        "--update", action="store_true", help="reinstall the dependencies"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="delete .venv and build it again"
    )
    parser.add_argument(
        "--dev", action="store_true", help="also install pytest and ruff"
    )
    args = parser.parse_args(argv)

    say(f"{BAR}\nBetterSDR setup\n{ROOT}\n{BAR}")

    total = 3 if args.check else 4
    for check in (check_interpreter, check_checkout):
        code = check()
        if code is not None:
            return code

    step(1, total, "Preparing the Python environment")
    code = build_environment(recreate=args.recreate)
    if code is not None:
        return code

    step(2, total, "Installing what the radio needs")
    if already_installed() and not (args.update or args.recreate):
        say("    everything is already installed - use --update to refresh it")
    else:
        code = install(dev=args.dev)
        if code is not None:
            return code

    step(3, total, "Checking the dongle")
    ready = check_radio() == 0
    if not ready:
        say(
            "\n    The radio is not ready yet. The steps above are the fix;\n"
            "    BetterSDR will also walk you through them on screen."
        )

    if args.check:
        return 0 if ready else 1

    if args.no_launch:
        here = Path(__file__).resolve().relative_to(ROOT).as_posix()
        say(f"\nReady. Start BetterSDR any time with:\n    py {here}")
        return 0

    step(4, total, "Starting BetterSDR")
    return launch()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\nStopped.")
        raise SystemExit(130) from None
