"""ctypes bindings to the RTL-SDR Blog fork of librtlsdr.

We deliberately do not use pyrtlsdr. The most common failure mode with the
RTL-SDR Blog V4 is loading the *wrong* rtlsdr.dll: the stock Osmocom build
mis-detects the R828D tuner and produces garbage that looks like a hardware
fault. pyrtlsdr resolves the DLL through the system search path, so which one
it finds depends on whatever else is installed on the machine.

This module loads the DLL we ship, by absolute path, and reports clearly when
it is not the Blog fork.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_uint8,
    c_uint32,
    c_void_p,
)
from enum import IntEnum
from pathlib import Path

# --------------------------------------------------------------------------
# Enums and error codes
# --------------------------------------------------------------------------


class Tuner(IntEnum):
    """Values returned by rtlsdr_get_tuner_type()."""

    UNKNOWN = 0
    E4000 = 1
    FC0012 = 2
    FC0013 = 3
    FC2580 = 4
    R820T = 5
    R828D = 6

    @property
    def label(self) -> str:
        return _TUNER_LABELS.get(self, "Unknown")


_TUNER_LABELS = {
    Tuner.UNKNOWN: "Unknown",
    Tuner.E4000: "Elonics E4000",
    Tuner.FC0012: "Fitipower FC0012",
    Tuner.FC0013: "Fitipower FC0013",
    Tuner.FC2580: "FCI FC2580",
    Tuner.R820T: "Rafael Micro R820T/R820T2",
    Tuner.R828D: "Rafael Micro R828D",
}

# librtlsdr passes libusb error codes straight through.
_LIBUSB_ERRORS = {
    -1: "I/O error",
    -2: "invalid parameter",
    -3: "access denied (is another program using the dongle?)",
    -4: "no such device (was it unplugged?)",
    -5: "entity not found",
    -6: "resource busy",
    -7: "operation timed out",
    -8: "overflow",
    -9: "pipe error",
    -10: "system call interrupted",
    -11: "out of memory",
    -12: "operation not supported",
    -99: "other error",
}


class RtlSdrError(RuntimeError):
    """A librtlsdr call returned a failure code."""

    def __init__(self, code: int, operation: str) -> None:
        detail = _LIBUSB_ERRORS.get(code, f"error {code}")
        super().__init__(f"{operation} failed: {detail} (code {code})")
        self.code = code
        self.operation = operation


class DriverNotFoundError(RuntimeError):
    """The bundled driver DLLs are missing from the install."""


def _check(code: int, operation: str) -> int:
    if code < 0:
        raise RtlSdrError(code, operation)
    return code


# --------------------------------------------------------------------------
# DLL loading
# --------------------------------------------------------------------------

# Present only in the RTL-SDR Blog fork; used to tell it apart from the stock
# Osmocom build. Note that rtlsdr_set_dithering is NOT a usable marker: it is
# absent from the Blog Windows V1.4.0 release even though the fork's source
# defines it.
_FORK_MARKER = "rtlsdr_check_dongle_model"

# rtlsdr.dll links against these; we preload them by absolute path so the
# loader never has to search for them.
_DEPENDENCIES = ("msvcr100.dll", "pthreadVC2.dll")


def driver_dir() -> Path:
    """Directory holding the bundled driver DLLs, frozen or from source."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parents[2]
    return base / "drivers" / "win-x64"


class Library:
    """A loaded librtlsdr, with the provenance we care about recorded."""

    def __init__(self, lib: ctypes.CDLL, path: Path, is_blog_fork: bool) -> None:
        self.lib = lib
        self.path = path
        self.is_blog_fork = is_blog_fork


_library: Library | None = None


def load(force_path: Path | None = None) -> Library:
    """Load the bundled rtlsdr.dll. Cached after the first successful call."""
    global _library
    if _library is not None and force_path is None:
        return _library

    directory = force_path.parent if force_path else driver_dir()
    dll_path = force_path or (directory / "rtlsdr.dll")

    if not dll_path.exists():
        raise DriverNotFoundError(
            f"rtlsdr.dll not found at {dll_path}.\n"
            "The bundled RTL-SDR Blog driver is missing from this install."
        )

    # Preload dependencies by absolute path. Once they are in the process the
    # loader resolves rtlsdr.dll's imports against them regardless of PATH.
    for name in _DEPENDENCIES:
        dep = directory / name
        if dep.exists():
            ctypes.CDLL(str(dep))

    lib = ctypes.CDLL(str(dll_path))
    is_fork = hasattr(lib, _FORK_MARKER)
    _bind(lib, is_fork)

    _library = Library(lib, dll_path, is_fork)
    return _library


# name, restype, argtypes
# The callback `rtlsdr_read_async` delivers each USB transfer to: a
# pointer to the bytes, how many there are, and the opaque context that
# was handed to `read_async`. It is called on whichever thread called
# `read_async`, which is what lets the reader keep its one-owner rule.
ReadCallback = CFUNCTYPE(None, POINTER(c_uint8), c_uint32, c_void_p)


_PROTOTYPES: tuple[tuple[str, object, tuple], ...] = (
    ("rtlsdr_get_device_count", c_uint32, ()),
    ("rtlsdr_get_device_name", c_char_p, (c_uint32,)),
    ("rtlsdr_get_device_usb_strings", c_int, (c_uint32, c_char_p, c_char_p, c_char_p)),
    ("rtlsdr_open", c_int, (POINTER(c_void_p), c_uint32)),
    ("rtlsdr_close", c_int, (c_void_p,)),
    ("rtlsdr_get_usb_strings", c_int, (c_void_p, c_char_p, c_char_p, c_char_p)),
    ("rtlsdr_set_center_freq", c_int, (c_void_p, c_uint32)),
    ("rtlsdr_get_center_freq", c_uint32, (c_void_p,)),
    ("rtlsdr_set_freq_correction", c_int, (c_void_p, c_int)),
    ("rtlsdr_get_freq_correction", c_int, (c_void_p,)),
    ("rtlsdr_get_tuner_type", c_int, (c_void_p,)),
    ("rtlsdr_get_tuner_gains", c_int, (c_void_p, POINTER(c_int))),
    ("rtlsdr_set_tuner_gain", c_int, (c_void_p, c_int)),
    ("rtlsdr_get_tuner_gain", c_int, (c_void_p,)),
    ("rtlsdr_set_tuner_gain_mode", c_int, (c_void_p, c_int)),
    ("rtlsdr_set_tuner_bandwidth", c_int, (c_void_p, c_uint32)),
    ("rtlsdr_set_tuner_if_gain", c_int, (c_void_p, c_int, c_int)),
    ("rtlsdr_set_sample_rate", c_int, (c_void_p, c_uint32)),
    ("rtlsdr_get_sample_rate", c_uint32, (c_void_p,)),
    ("rtlsdr_set_agc_mode", c_int, (c_void_p, c_int)),
    ("rtlsdr_set_testmode", c_int, (c_void_p, c_int)),
    ("rtlsdr_set_direct_sampling", c_int, (c_void_p, c_int)),
    ("rtlsdr_get_direct_sampling", c_int, (c_void_p,)),
    ("rtlsdr_set_offset_tuning", c_int, (c_void_p, c_int)),
    ("rtlsdr_get_offset_tuning", c_int, (c_void_p,)),
    ("rtlsdr_set_xtal_freq", c_int, (c_void_p, c_uint32, c_uint32)),
    ("rtlsdr_get_xtal_freq", c_int, (c_void_p, POINTER(c_uint32), POINTER(c_uint32))),
    ("rtlsdr_reset_buffer", c_int, (c_void_p,)),
    ("rtlsdr_read_sync", c_int, (c_void_p, c_void_p, c_int, POINTER(c_int))),
    (
        "rtlsdr_read_async",
        c_int,
        (c_void_p, ReadCallback, c_void_p, c_uint32, c_uint32),
    ),
    ("rtlsdr_cancel_async", c_int, (c_void_p,)),
)

# Blog-fork-only entry points.
_FORK_PROTOTYPES: tuple[tuple[str, object, tuple], ...] = (
    ("rtlsdr_set_bias_tee", c_int, (c_void_p, c_int)),
    ("rtlsdr_set_bias_tee_gpio", c_int, (c_void_p, c_int, c_int)),
)


def _bind(lib: ctypes.CDLL, is_fork: bool) -> None:
    prototypes = _PROTOTYPES + (_FORK_PROTOTYPES if is_fork else ())
    for name, restype, argtypes in prototypes:
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue  # Older builds may lack a call; Device probes before use.
        fn.restype = restype
        fn.argtypes = argtypes


__all__ = [
    "DriverNotFoundError",
    "ReadCallback",
    "Library",
    "RtlSdrError",
    "Tuner",
    "byref",
    "driver_dir",
    "load",
    "_check",
]
