"""High-level control of an RTL-SDR dongle.

Wraps the raw bindings in `native` with a Pythonic surface, and adds the
V4-specific knowledge the rest of the app relies on: tuner identification,
the HF upconverter's frequency range, and bias-tee safety.
"""

from __future__ import annotations

import argparse
import sys
from ctypes import byref, c_int, c_uint32, c_void_p, create_string_buffer
from dataclasses import dataclass

import numpy as np

from . import native
from .native import RtlSdrError, Tuner

# USB bulk transfers must be a multiple of 512 bytes. librtlsdr's own tools
# read 16384 at a time; anything much smaller costs throughput.
BULK_GRANULARITY = 512
DEFAULT_BLOCK_BYTES = 16384

# The V4 covers this range continuously. Below HF_CEILING_HZ the signal goes
# through the dongle's built-in SA612 upconverter, which the Blog driver
# handles transparently -- no mode switch and no offset needed from us.
MIN_TUNE_HZ = 500_000
MAX_TUNE_HZ = 1_766_000_000
HF_CEILING_HZ = 28_800_000

# 2.4 MS/s is the highest rate the RTL2832U sustains without dropping samples.
DEFAULT_SAMPLE_RATE = 2_400_000

_USB_STRING_LEN = 256


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    manufacturer: str
    product: str
    serial: str
    tuner: Tuner

    @property
    def is_v4(self) -> bool:
        """The V4 is the only RTL-SDR Blog dongle using the R828D tuner."""
        return self.tuner is Tuner.R828D

    @property
    def model_guess(self) -> str:
        if self.tuner is Tuner.R828D:
            return "RTL-SDR Blog V4"
        if self.tuner is Tuner.R820T:
            return "RTL-SDR V3 or compatible (R820T/R820T2)"
        return f"Unrecognised dongle ({self.tuner.label})"


def device_count() -> int:
    lib = native.load().lib
    return int(lib.rtlsdr_get_device_count())


def list_devices() -> list[str]:
    """Names of every connected dongle, without opening them."""
    lib = native.load().lib
    names = []
    for index in range(device_count()):
        raw = lib.rtlsdr_get_device_name(index)
        names.append(raw.decode("utf-8", "replace") if raw else "")
    return names


class Device:
    """An open RTL-SDR dongle.

    Not thread-safe by design. The reader thread owns the device; other
    threads submit changes through a command queue rather than calling here.
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._lib = native.load()
        self._dev: c_void_p | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Device:
        if self._dev is not None:
            return self
        handle = c_void_p()
        code = self._lib.lib.rtlsdr_open(byref(handle), self.index)
        if code < 0:
            raise RtlSdrError(code, f"opening dongle {self.index}")
        self._dev = handle
        return self

    def close(self) -> None:
        if self._dev is None:
            return
        self._lib.lib.rtlsdr_close(self._dev)
        self._dev = None

    def __enter__(self) -> Device:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._dev is not None

    def _handle(self) -> c_void_p:
        if self._dev is None:
            raise RuntimeError("Device is not open; call open() first.")
        return self._dev

    def _call(self, name: str, *args: object) -> int:
        fn = getattr(self._lib.lib, name)
        code = fn(self._handle(), *args)
        if code < 0:
            raise RtlSdrError(code, name)
        return code

    # -- identity ----------------------------------------------------------

    @property
    def tuner(self) -> Tuner:
        raw = self._lib.lib.rtlsdr_get_tuner_type(self._handle())
        try:
            return Tuner(raw)
        except ValueError:
            return Tuner.UNKNOWN

    def usb_strings(self) -> tuple[str, str, str]:
        manufacturer = create_string_buffer(_USB_STRING_LEN)
        product = create_string_buffer(_USB_STRING_LEN)
        serial = create_string_buffer(_USB_STRING_LEN)
        self._call("rtlsdr_get_usb_strings", manufacturer, product, serial)
        return tuple(
            b.value.decode("utf-8", "replace") for b in (manufacturer, product, serial)
        )

    def info(self) -> DeviceInfo:
        manufacturer, product, serial = self.usb_strings()
        raw = self._lib.lib.rtlsdr_get_device_name(self.index)
        return DeviceInfo(
            index=self.index,
            name=raw.decode("utf-8", "replace") if raw else "",
            manufacturer=manufacturer,
            product=product,
            serial=serial,
            tuner=self.tuner,
        )

    # -- tuning ------------------------------------------------------------

    @property
    def center_freq(self) -> int:
        return int(self._lib.lib.rtlsdr_get_center_freq(self._handle()))

    @center_freq.setter
    def center_freq(self, hz: int) -> None:
        hz = int(hz)
        if not MIN_TUNE_HZ <= hz <= MAX_TUNE_HZ:
            raise ValueError(
                f"{hz / 1e6:.3f} MHz is outside the dongle's range "
                f"({MIN_TUNE_HZ / 1e6:.1f}-{MAX_TUNE_HZ / 1e6:.0f} MHz)."
            )
        self._call("rtlsdr_set_center_freq", c_uint32(hz))

    @property
    def uses_upconverter(self) -> bool:
        """True when tuned through the V4's built-in HF upconverter."""
        return self.center_freq < HF_CEILING_HZ

    @property
    def sample_rate(self) -> int:
        return int(self._lib.lib.rtlsdr_get_sample_rate(self._handle()))

    @sample_rate.setter
    def sample_rate(self, hz: int) -> None:
        self._call("rtlsdr_set_sample_rate", c_uint32(int(hz)))

    @property
    def freq_correction_ppm(self) -> int:
        return int(self._lib.lib.rtlsdr_get_freq_correction(self._handle()))

    @freq_correction_ppm.setter
    def freq_correction_ppm(self, ppm: int) -> None:
        # Setting the same value the driver already holds returns -2; harmless.
        if int(ppm) == self.freq_correction_ppm:
            return
        self._call("rtlsdr_set_freq_correction", c_int(int(ppm)))

    # -- gain --------------------------------------------------------------

    @property
    def gains_db(self) -> tuple[float, ...]:
        """Discrete gain steps this tuner supports, in dB."""
        handle = self._handle()
        count = self._lib.lib.rtlsdr_get_tuner_gains(handle, None)
        if count <= 0:
            return ()
        buf = (c_int * count)()
        self._lib.lib.rtlsdr_get_tuner_gains(handle, buf)
        return tuple(value / 10.0 for value in buf)

    @property
    def gain_db(self) -> float:
        return self._lib.lib.rtlsdr_get_tuner_gain(self._handle()) / 10.0

    @gain_db.setter
    def gain_db(self, db: float) -> None:
        self._call("rtlsdr_set_tuner_gain", c_int(int(round(db * 10))))

    def set_manual_gain(self, enabled: bool) -> None:
        """Manual gain mode. Disable to hand gain control to the tuner."""
        self._call("rtlsdr_set_tuner_gain_mode", c_int(1 if enabled else 0))

    def set_agc(self, enabled: bool) -> None:
        """The RTL2832U's digital AGC, separate from tuner gain."""
        self._call("rtlsdr_set_agc_mode", c_int(1 if enabled else 0))

    # -- V4 extras ---------------------------------------------------------

    @property
    def supports_bias_tee(self) -> bool:
        return self._lib.is_blog_fork and hasattr(self._lib.lib, "rtlsdr_set_bias_tee")

    def set_bias_tee(self, enabled: bool) -> None:
        """Switch the antenna port's 4.5 V supply.

        Only ever call this with the user's explicit consent: it feeds DC into
        whatever is connected, which can damage equipment not expecting it.
        """
        if not self.supports_bias_tee:
            raise RuntimeError(
                "This driver does not support the bias tee. The bundled "
                "RTL-SDR Blog driver is required."
            )
        self._call("rtlsdr_set_bias_tee", c_int(1 if enabled else 0))

    # -- streaming ---------------------------------------------------------

    def reset_buffer(self) -> None:
        """Drop stale samples. Required after tuning before reading."""
        self._call("rtlsdr_reset_buffer")

    def read(self, n_bytes: int = DEFAULT_BLOCK_BYTES) -> np.ndarray:
        """Read one block of interleaved 8-bit IQ.

        Returns a uint8 array of length n_bytes: I, Q, I, Q, ...
        """
        if n_bytes % BULK_GRANULARITY:
            raise ValueError(
                f"n_bytes must be a multiple of {BULK_GRANULARITY}, got {n_bytes}."
            )
        buf = np.empty(n_bytes, dtype=np.uint8)
        n_read = c_int(0)
        self._call(
            "rtlsdr_read_sync",
            buf.ctypes.data_as(c_void_p),
            c_int(n_bytes),
            byref(n_read),
        )
        if n_read.value != n_bytes:
            raise RtlSdrError(-8, f"read_sync (got {n_read.value} of {n_bytes} bytes)")
        return buf

    def read_async(
        self,
        callback,
        buffer_count: int = 16,
        buffer_bytes: int = 131_072,
    ) -> None:
        """Stream into `callback` with several USB transfers queued at once.

        **Blocks until `cancel_async` is called**, and calls `callback` on this
        thread for every transfer that completes. That is the difference from
        `read`: with transfers queued behind the one being delivered there is
        never a moment with nothing in flight, so the sample stream has no
        gaps in it at all.

        Measured off air on 94.9 MHz, both at 12.5 dB and 1,488,375 S/s:
        reading one block at a time gave the HD Radio decoder a modulation
        error ratio of **-13 dB and no audio**, and this gave **+9 to +10 dB,
        91.8 kbps and no loss of sync in fifteen seconds**. Nothing else
        differed. An OFDM receiver tracks a frame across blocks, so a
        discontinuity every read is fatal to it in a way it is not to any
        analog demodulator or to ADS-B.

        `callback` must not raise: an exception escaping a ctypes callback is
        printed and swallowed, so the caller has to catch its own.
        """
        wrapped = native.ReadCallback(callback)
        # Held on the instance for the duration of the call. A callback object
        # collected while librtlsdr still holds the pointer is a crash, and
        # ctypes gives no warning about it.
        self._async_callback = wrapped
        try:
            self._call(
                "rtlsdr_read_async",
                wrapped,
                None,
                c_uint32(int(buffer_count)),
                c_uint32(int(buffer_bytes)),
            )
        finally:
            self._async_callback = None

    def cancel_async(self) -> None:
        """Ask a running `read_async` to return. Safe to call more than once.

        Deliberately not routed through `_call`: this is called from inside
        the read callback, where raising would leave librtlsdr unwinding
        through a Python exception, and a second cancel after the first is a
        normal thing rather than an error.
        """
        self._lib.lib.rtlsdr_cancel_async(self._handle())

    def configure(
        self,
        *,
        center_freq: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        gain_db: float | None = None,
        ppm: int = 0,
    ) -> None:
        """Apply a full tuning setup in the order the hardware expects."""
        self.sample_rate = sample_rate
        if ppm:
            self.freq_correction_ppm = ppm
        self.center_freq = center_freq
        if gain_db is None:
            self.set_manual_gain(False)
        else:
            self.set_manual_gain(True)
            self.gain_db = gain_db
        self.reset_buffer()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_driver_report() -> native.Library | None:
    try:
        lib = native.load()
    except native.DriverNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return None
    print(f"  DLL          : {lib.path}")
    fork = "yes" if lib.is_blog_fork else "NO -- wrong driver!"
    print(f"  Blog fork    : {fork}")
    if not lib.is_blog_fork:
        print("  This looks like the stock Osmocom build. It mis-detects the")
        print("  V4's R828D tuner and will produce garbage. Reinstall BetterSDR.")
    return lib


def _capture_summary(dev: Device) -> None:
    """Read a short block and report basic statistics.

    This is the proof that samples actually flow: a dongle that opens but
    returns constant or all-zero data has a real problem.
    """
    dev.reset_buffer()
    dev.read()  # Discard the first block; it holds pre-tune samples.
    raw = dev.read(65536)
    iq = (raw.astype(np.float32) - 127.5) / 127.5
    samples = iq[0::2] + 1j * iq[1::2]

    dc = complex(np.mean(samples))
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    clipped = float(np.mean((raw == 0) | (raw == 255)) * 100)

    print(f"  Samples read : {samples.size}")
    print(f"  RMS level    : {rms:.4f}  ({20 * np.log10(max(rms, 1e-9)):.1f} dBFS)")
    print(f"  DC offset    : {dc.real:+.4f}{dc.imag:+.4f}j")
    print(f"  Clipped      : {clipped:.2f}% of samples at 0 or 255")
    if rms < 1e-4:
        print("  WARNING: signal is essentially silent. Antenna connected?")
    elif clipped > 1.0:
        print("  WARNING: input is overloading. Reduce gain.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bettersdr.core.device",
        description="Inspect the connected RTL-SDR dongle.",
    )
    parser.add_argument(
        "--info", action="store_true", help="Report driver and dongle status."
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=100.0,
        help="Frequency in MHz for the capture test (default: 100.0).",
    )
    args = parser.parse_args(argv)
    if not args.info:
        parser.print_help()
        return 0

    from .doctor import diagnose

    print("Driver")
    lib = _print_driver_report()
    if lib is None:
        return 1

    print("\nUSB")
    diagnosis = diagnose()
    print(f"  {diagnosis.headline}")
    for node in diagnosis.nodes:
        which = (
            "parent"
            if node.is_composite_parent
            else f"interface {node.interface_number}"
        )
        service = node.service or "(none)"
        print(f"    {which}: driver={service} problem={node.problem}")

    if not diagnosis.ok:
        print("\nWhat to do")
        for step in diagnosis.remedy:
            print(f"  - {step}")
        return 1

    print("\nDongle")
    count = device_count()
    print(f"  Dongles found: {count}")
    if count == 0:
        print("  The driver is installed but librtlsdr sees no device.")
        return 1

    try:
        with Device(0) as dev:
            info = dev.info()
            print(f"  Name         : {info.name}")
            print(f"  Manufacturer : {info.manufacturer}")
            print(f"  Product      : {info.product}")
            print(f"  Serial       : {info.serial}")
            print(f"  Tuner        : {info.tuner.label}")
            print(f"  Model        : {info.model_guess}")
            print(f"  Bias tee     : {'supported' if dev.supports_bias_tee else 'no'}")

            gains = dev.gains_db
            print(f"  Gain steps   : {len(gains)} ({gains[0]:.1f} - {gains[-1]:.1f} dB)")

            print(f"\nCapture test at {args.freq:.3f} MHz")
            dev.configure(center_freq=int(args.freq * 1e6), gain_db=None)
            print(f"  Sample rate  : {dev.sample_rate / 1e6:.3f} MS/s")
            print(f"  Upconverter  : {'yes (HF)' if dev.uses_upconverter else 'no'}")
            _capture_summary(dev)
    except RtlSdrError as exc:
        print(f"  ERROR: {exc}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
