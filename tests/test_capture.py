"""Capture buffering.

No audio hardware here — these exercise the queue and pre-roll bookkeeping,
which is where the observed defect was: a ~1s tier-4 round trip let the
callback fill the queue with audio belonging to no utterance.
"""

from __future__ import annotations

import numpy as np

from crew_chief_hearing_aid.audio.capture import AudioCapture, CaptureConfig
from crew_chief_hearing_aid.audio.devices import InputDevice


def _capture() -> AudioCapture:
    device = InputDevice(index=0, name="fake", channels=1, default_samplerate=16000)
    return AudioCapture(device, CaptureConfig(sample_rate=16000, block_ms=32))


def _feed(cap: AudioCapture, n: int) -> None:
    """Push n blocks through the callback without any audio backend."""
    block = np.zeros((cap.config.block_samples, 1), dtype=np.float32)
    for _ in range(n):
        cap._callback(block, cap.config.block_samples, None, None)


class TestConfig:
    def test_block_samples(self):
        assert CaptureConfig(sample_rate=16000, block_ms=32).block_samples == 512

    def test_preroll_blocks_covers_the_window(self):
        cfg = CaptureConfig(block_ms=32, preroll_ms=512)
        assert cfg.preroll_blocks == 16


class TestDrain:
    def test_discards_queued_blocks(self):
        cap = _capture()
        _feed(cap, 10)
        assert cap.drain() == 10
        assert cap.read(timeout=0.01) is None

    def test_is_safe_when_empty(self):
        assert _capture().drain() == 0

    def test_preroll_survives_a_drain(self):
        """The next press still needs its lead-in.

        Draining is about the work queue, not the ring buffer — clearing both
        would clip the first word of the following utterance.
        """
        cap = _capture()
        _feed(cap, 20)
        cap.drain()
        assert cap.snapshot_preroll().size > 0


class TestOverflow:
    def test_drops_are_counted_not_raised(self):
        """A full queue must never propagate into the audio callback — an
        exception there kills the stream."""
        cap = _capture()
        _feed(cap, 200)  # maxsize is 128
        assert cap.dropped_blocks > 0
        assert cap.read(timeout=0.01) is not None

    def test_preroll_is_unaffected_by_overflow(self):
        cap = _capture()
        _feed(cap, 200)
        expected = cap.config.preroll_blocks * cap.config.block_samples
        assert cap.snapshot_preroll().size == expected


class TestPreroll:
    def test_is_bounded(self):
        cap = _capture()
        _feed(cap, 500)
        expected = cap.config.preroll_blocks * cap.config.block_samples
        assert cap.snapshot_preroll().size == expected

    def test_empty_before_any_audio(self):
        assert _capture().snapshot_preroll().size == 0
