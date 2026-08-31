"""The console half of a packaged build.

A packaged user has no Python, so `python -m bettersdr.core.device --info` -
the one instruction every driver problem in this project ends with - is not
available to them at the moment they need it most. This module is that
instruction, reachable as `BetterSDR-Tools.exe check` from the same folder
as the application.

It exists for a second reason that only appears once the app is frozen: a
windowed build has no standard output, so a failure before the first window
opens has nowhere to print itself. `BetterSDR-Tools.exe app` runs exactly
the same program with a console attached, which turns "it does nothing when
I double-click it" into a traceback somebody can read.

No new behaviour lives here. Every command is an existing entry point.
"""

from __future__ import annotations

import sys

COMMANDS = {
    "check": ("bettersdr.core.device", "driver and dongle status, then a capture test"),
    "listen": ("bettersdr.listen", "tune and play, with no GUI in the way"),
    "app": ("bettersdr.app", "the application, with a console attached"),
}


def usage() -> str:
    lines = [
        "BetterSDR command-line tools",
        "",
        "Usage: BetterSDR-Tools <command> [options]",
        "",
    ]
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {description}")
    lines += [
        "",
        "Anything after the command is passed straight to it; try",
        "",
        "  BetterSDR-Tools check --info",
        "  BetterSDR-Tools listen --freq 94.9",
        "",
        "Start here when the application will not open: `check --info` exits 0",
        "when the radio is ready and prints the specific remedy when it is not.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage())
        return 0

    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    module_name, _ = COMMANDS[command]
    rest = args[1:]
    # `check` with nothing after it is what somebody in trouble will type.
    if command == "check" and not rest:
        rest = ["--info"]

    from importlib import import_module

    module = import_module(module_name)
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
