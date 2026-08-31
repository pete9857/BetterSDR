"""Entry point for the packaged windowed application.

A frozen build runs its entry script as `__main__` with no package around
it, so `bettersdr/app.py` cannot be that script: its imports are relative,
and a relative import with no parent package resolves to nothing. The
failure is quiet in the worst way - PyInstaller's analysis follows the
imports and reports no warning, and the executable it produces raises
`ModuleNotFoundError` on the first line the user sees.

So the entry points are these two shims, which import the package properly
and hand straight over. They live outside `bettersdr/` because they are not
part of the radio; they are the shape a frozen build needs.
"""

from bettersdr.app import main

if __name__ == "__main__":
    raise SystemExit(main())
