"""Headless tune-and-listen harness.

This is the Phase 1 acceptance test in runnable form: point it at a frequency
and it should sound like a radio. It exists separately from the GUI because
audio faults and UI faults are easy to confuse, and this path has no Qt in it
at all.

It uses the real reader thread rather than reading and demodulating in one
loop. That is not a stylistic choice: demodulating between reads leaves no USB
transfer in flight, which measured at an 11% sample loss on a V4 and a steady
trickle of audio underruns. The GUI adds a third thread on top of this; the
reader and DSP halves here are the same ones it uses.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .audio.output import AudioSink, default_output_device, output_devices
from .core.device import DEFAULT_SAMPLE_RATE, Device, device_count
from .core.frontend import choose_gain
from .core.reader import Reader
from .dsp import convert, demod

# 64 KB is ~27 ms of radio at 2.4 MS/s: long enough that per-block overhead is
# irrelevant, short enough to keep the meter responsive.
READ_BYTES = 65_536


def _meter(level_dbfs: float, width: int = 20, floor: float = -60.0) -> str:
    fraction = (level_dbfs - floor) / (0.0 - floor)
    filled = int(np.clip(fraction, 0.0, 1.0) * width)
    return "#" * filled + "-" * (width - filled)


def _parse_frequency(text: str) -> int:
    """Accept 98.5 as MHz and 98500000 as Hz, the way people actually type."""
    value = float(text)
    return int(value if value > 10_000 else value * 1e6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bettersdr.listen",
        description="Tune the dongle and play the audio.",
    )
    parser.add_argument(
        "--freq", default="98.5", help="frequency in MHz, or in Hz if large"
    )
    parser.add_argument(
        "--mode", default="wfm", choices=sorted(demod.MODES), help="demodulation mode"
    )
    parser.add_argument("--bandwidth", type=float, default=None, help="channel width Hz")
    parser.add_argument("--gain", type=float, default=None, help="dB; omit for auto")
    parser.add_argument("--ppm", type=int, default=0, help="frequency correction")
    parser.add_argument("--volume", type=float, default=0.5)
    parser.add_argument("--squelch", type=float, default=None, help="dBFS threshold")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N s")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--audio-device", default=None, help="index or name substring")
    parser.add_argument(
        "--list-audio", action="store_true", help="show output devices and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_audio:
        print(f"Default output: {default_output_device()}")
        for index, name in output_devices():
            print(f"  {index:>3}  {name}")
        return 0

    if device_count() == 0:
        print("No dongle found. Run: python -m bettersdr.core.device --info")
        return 1

    freq_hz = _parse_frequency(args.freq)
    device = args.audio_device
    if device is not None and device.isdigit():
        device = int(device)

    with Device() as dev:
        info = dev.info()
        print(f"{info.model_guess}  ({info.tuner.name})")
        dev.configure(center_freq=freq_hz, sample_rate=args.rate, ppm=args.ppm)

        if args.gain is None:
            chosen = choose_gain(dev)
            print(
                f"Auto gain    : {chosen.gain_db:.1f} dB "
                f"(level {chosen.level_dbfs:.1f} dBFS, "
                f"{chosen.clipped_fraction * 100:.2f}% clipped)"
            )
        else:
            dev.set_manual_gain(True)
            dev.gain_db = args.gain
            print(f"Gain         : {dev.gain_db:.1f} dB")

        demodulator = demod.create(
            args.mode,
            float(args.rate),
            bandwidth_hz=args.bandwidth,
            volume=args.volume,
            squelch_dbfs=args.squelch,
        )
        print(
            f"Tuned to     : {freq_hz / 1e6:.4f} MHz  {demodulator.label} "
            f"({demodulator.bandwidth_hz / 1000:.1f} kHz wide)"
        )
        print(f"Audio out    : {default_output_device()}")
        print("Playing. Press Ctrl+C to stop.\n")

        reader = Reader(dev)
        reader.start()
        reader.wait_until_running()
        sink = AudioSink(rate=demod.AUDIO_RATE, device=device).start()
        started = time.perf_counter()
        last_meter = 0.0
        stalls = 0
        try:
            while True:
                elapsed = time.perf_counter() - started
                if args.seconds is not None and elapsed >= args.seconds:
                    break
                raw = reader.ring.read(READ_BYTES, timeout=1.0)
                if raw is None:
                    stalls += 1
                    if reader.last_error:
                        print(f"\nDevice error: {reader.last_error}")
                        break
                    continue
                sink.write(demodulator.process(convert.to_complex(raw)))

                if elapsed - last_meter >= 0.25:
                    last_meter = elapsed
                    gate = "" if demodulator.squelch is None else (
                        " open" if demodulator.squelch.is_open else " muted"
                    )
                    sys.stdout.write(
                        f"\r  {freq_hz / 1e6:9.4f} MHz  "
                        f"[{_meter(demodulator.channel_power_dbfs)}] "
                        f"{demodulator.channel_power_dbfs:6.1f} dBFS{gate}  "
                        f"buffer {sink.latency_s * 1000:3.0f} ms  "
                        f"drift {(sink.clock.ratio - 1) * 100:+.2f}%  "
                        f"underruns {sink.underruns}"
                    )
                    sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        finally:
            elapsed = time.perf_counter() - started
            reader.stop()
            sink.stop()

        captured = reader.blocks_read * reader.block_bytes / 2 / args.rate
        print("\n")
        print(f"Ran for      : {elapsed:.1f} s")
        print(f"Captured     : {captured:.1f} s of radio ({captured / elapsed:.1%})")
        print(f"Underruns    : {sink.underruns} audio, {stalls} device stall(s)")
        print(f"Ring overruns: {reader.ring.overruns} (DSP falling behind the radio)")
        print(f"Dropped      : {sink.dropped_blocks} block(s) to hold latency down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
