"""Tests for the IQ ring buffer.

The buffer sits between a thread that must never block and a thread that is
allowed to fall behind, so the tests care about two things: data comes out
byte-identical and in order, and falling behind loses the *oldest* samples in a
counted, visible way rather than corrupting the stream.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from bettersdr.core.ringbuffer import RingBuffer


def ramp(start: int, count: int) -> np.ndarray:
    """A recognisable byte pattern, so ordering errors are obvious."""
    return ((np.arange(start, start + count)) % 256).astype(np.uint8)


def test_round_trip_preserves_bytes():
    ring = RingBuffer(1024)
    ring.write(ramp(0, 256))
    np.testing.assert_array_equal(ring.read(256), ramp(0, 256))


def test_reads_in_order_across_many_writes():
    ring = RingBuffer(1024)
    for i in range(4):
        ring.write(ramp(i * 100, 100))
    np.testing.assert_array_equal(ring.read(400), ramp(0, 400))


def test_wraparound_is_contiguous():
    ring = RingBuffer(300)
    ring.write(ramp(0, 200))
    assert ring.read(200).size == 200
    # This write straddles the end of the backing array.
    ring.write(ramp(200, 200))
    np.testing.assert_array_equal(ring.read(200), ramp(200, 200))


def test_partial_read_leaves_the_rest():
    ring = RingBuffer(1024)
    ring.write(ramp(0, 100))
    np.testing.assert_array_equal(ring.read(40), ramp(0, 40))
    assert len(ring) == 60
    np.testing.assert_array_equal(ring.read(60), ramp(40, 60))


def test_read_times_out_rather_than_hanging():
    ring = RingBuffer(1024)
    ring.write(ramp(0, 10))
    assert ring.read(100, timeout=0.05) is None


def test_read_waits_for_a_later_write():
    ring = RingBuffer(1024)
    result: list[np.ndarray | None] = []

    def consume() -> None:
        result.append(ring.read(200, timeout=2.0))

    consumer = threading.Thread(target=consume)
    consumer.start()
    ring.write(ramp(0, 200))
    consumer.join(2.0)

    assert result and result[0] is not None
    np.testing.assert_array_equal(result[0], ramp(0, 200))


def test_overrun_discards_oldest_and_counts_it():
    ring = RingBuffer(256)
    ring.write(ramp(0, 200))
    ring.write(ramp(200, 200))  # 144 bytes too many

    assert ring.overruns == 1
    assert ring.dropped_bytes == 144
    assert len(ring) == 256
    # What survives is the newest 256 bytes, ending at the last byte written.
    np.testing.assert_array_equal(ring.read(256), ramp(144, 256))


def test_write_larger_than_capacity_keeps_the_newest_tail():
    ring = RingBuffer(128)
    ring.write(ramp(0, 500))
    assert len(ring) == 128
    np.testing.assert_array_equal(ring.read(128), ramp(372, 128))


def test_clear_empties_the_buffer():
    ring = RingBuffer(256)
    ring.write(ramp(0, 100))
    ring.clear()
    assert len(ring) == 0
    assert ring.read(1, timeout=0.01) is None


def test_no_data_lost_under_concurrent_producer_and_consumer():
    """The real usage pattern: a paced producer and a consumer keeping up.

    The producer is deliberately throttled. A dongle delivers samples on a
    fixed clock, so an unthrottled writer would be testing memcpy speed
    against the GIL rather than anything the app ever does.
    """
    ring = RingBuffer(1 << 16)
    block, blocks = 4096, 60
    received: list[np.ndarray] = []

    def produce() -> None:
        for i in range(blocks):
            ring.write(ramp(i * block, block))
            time.sleep(0.001)

    consumer_done = threading.Event()

    def consume() -> None:
        while len(received) < blocks:
            chunk = ring.read(block, timeout=1.0)
            if chunk is None:
                break
            received.append(chunk)
        consumer_done.set()

    consumer = threading.Thread(target=consume)
    consumer.start()
    produce()
    consumer_done.wait(10.0)
    consumer.join(1.0)

    # The consumer keeps up here, so nothing should have been discarded.
    assert ring.overruns == 0
    assert len(received) == blocks
    np.testing.assert_array_equal(np.concatenate(received), ramp(0, block * blocks))


def test_rejects_impossible_sizes():
    with pytest.raises(ValueError, match="capacity must be positive"):
        RingBuffer(0)
    with pytest.raises(ValueError, match="from a"):
        RingBuffer(64).read(128)
