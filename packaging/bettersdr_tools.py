"""Entry point for the packaged console tools. See `bettersdr_app.py`."""

from bettersdr.diagnose import main

if __name__ == "__main__":
    raise SystemExit(main())
