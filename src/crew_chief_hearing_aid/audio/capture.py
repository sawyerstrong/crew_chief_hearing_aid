"""Continuous capture with a pre-roll ring buffer.

The wake word is only detected *after* it has been spoken, and people run the
command straight into it ("hey chief what's the gap"). Without a pre-roll the
first word of the command is already gone by the time capture starts. The ring
buffer keeps the last `preroll_ms` permanently, so the utterance can be
reconstructed from before the trigger fired.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .devices import InputDevice

log = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    sample_rate: int = 16000
    block_ms: int = 32
    preroll_ms: int = 512

    @property
    def block_samples(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def preroll_blocks(self) -> int:
        return max(1, int(self.preroll_ms / self.block_ms))


class AudioCapture:
    """Background capture thread producing float32 mono blocks in [-1, 1]."""

    def __init__(self, device: InputDevice, config: CaptureConfig) -> None:
        self.device = device
        self.config = config
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self._preroll: deque[np.ndarray] = deque(maxlen=config.preroll_blocks)
        self._stream = None
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped_blocks(self) -> int:
        return self._dropped

    def drain(self) -> int:
        """Discard queued audio and return how much was thrown away.

        Called after an utterance is dispatched. Audio captured *while* we were
        transcribing and routing belongs to no utterance — under push-to-talk
        the button was already released — so replaying it would only feed stale
        blocks into the next capture.
        """
        discarded = 0
        try:
            while True:
                self._queue.get_nowait()
                discarded += 1
        except queue.Empty:
            pass
        return discarded

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Overflows mean we are not draining fast enough; the recogniser
            # will see a gap. Worth surfacing rather than swallowing.
            log.warning("audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            self._preroll.append(block)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Expected while an utterance is being processed: transcription and
            # a tier-4 round trip can take ~1s, during which nothing drains and
            # the callback keeps producing. The pre-roll ring buffer is
            # unaffected, so the next utterance still gets its lead-in.
            # Logged at debug because the pipeline drains after each utterance
            # (see Pipeline._finish_utterance) — a *sustained* run of these
            # means the consumer is genuinely too slow, not merely busy.
            self._dropped += 1
            log.debug("capture queue full; dropping block (%d total)", self._dropped)

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device.index,
            channels=1,
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info("capture started on %s", self.device)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("capture stopped")

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def snapshot_preroll(self) -> np.ndarray:
        with self._lock:
            blocks = list(self._preroll)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
