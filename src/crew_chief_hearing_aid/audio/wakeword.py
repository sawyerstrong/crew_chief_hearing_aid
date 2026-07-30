"""Wake word detection via openWakeWord.

Runs continuously on one CPU core so no wheel button is consumed. The cooldown
matters: without it a single "hey chief" spanning several frames retriggers the
pipeline repeatedly.

Portability note — a model trained on your own voice is more accurate but will
not generalise to anyone else. Ship a pretrained one as the default and keep
custom models in wakeword_custom/ (gitignored).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.55,
        cooldown_ms: int = 1500,
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.cooldown_s = cooldown_ms / 1000.0
        self.sample_rate = sample_rate
        self._model = None
        self._last_fire = 0.0

    def _load(self):
        if self._model is None:
            import onnxruntime as ort

            # openWakeWord's Model() exposes no provider argument, and
            # onnxruntime prefers CUDA whenever a GPU build is installed. The
            # only lever that actually works from here is hiding the GPU from
            # CUDA before the session is created -- nothing else in this
            # process uses CUDA, so the scope is safe.
            #
            # This must happen before `openwakeword.model` is imported: ORT
            # enumerates devices at session construction, and the import chain
            # can construct one eagerly.
            available = ort.get_available_providers()
            if available != ["CPUExecutionProvider"]:
                log.warning(
                    "onnxruntime exposes %s; hiding the GPU so the wake word stays "
                    "on CPU. Install the CPU-only 'onnxruntime' wheel, not "
                    "'onnxruntime-gpu'.",
                    available,
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            log.info("wake word model %r loaded", self.model_name)
        return self._model

    def warmup(self) -> None:
        self._load()

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def detect(self, frame: np.ndarray) -> bool:
        """Feed one frame. True on a fresh trigger (cooldown respected)."""
        model = self._load()
        # openWakeWord expects int16 PCM.
        pcm = np.clip(np.asarray(frame, dtype=np.float32) * 32767.0, -32768, 32767)
        scores = model.predict(pcm.astype(np.int16))
        best = max(scores.values()) if scores else 0.0

        if best < self.threshold:
            return False
        now = time.monotonic()
        if now - self._last_fire < self.cooldown_s:
            return False
        self._last_fire = now
        self.reset()
        log.info("wake word fired (score=%.3f)", best)
        return True


class AlwaysOpenDetector:
    """Pass-through for push-to-talk or dry-run testing."""

    def warmup(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def detect(self, frame: np.ndarray) -> bool:
        return False
