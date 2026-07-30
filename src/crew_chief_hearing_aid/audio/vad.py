"""Silero VAD wrapper — decides when the user stopped talking.

~1.8MB ONNX model, sub-millisecond per 32ms frame on CPU. Its job is endpointing
(when to stop recording and hand off to Whisper), not gating the wake word.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8kHz or 16kHz only")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = None
        self._reset_state()

    def _reset_state(self) -> None:
        # Silero v5 keeps a single (2, 1, 128) recurrent state.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def _load(self):
        if self._model is None:
            import onnxruntime as ort

            from ..models import ensure_model

            path = ensure_model("silero_vad")
            opts = ort.SessionOptions()
            # One thread: this runs every 32ms alongside a sim rendering VR.
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._model = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._model

    def reset(self) -> None:
        self._reset_state()

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= self.threshold

    def probability(self, frame: np.ndarray) -> float:
        model = self._load()
        audio = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        outputs = model.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        prob, self._state = float(outputs[0].item()), outputs[1]
        return prob


class EnergyVAD:
    """Zero-dependency fallback.

    Markedly worse than Silero in a cockpit — wheel and pedal noise carry real
    energy — but it keeps the pipeline running if the ONNX model is missing.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.02) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold

    def reset(self) -> None:
        return None

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(np.asarray(frame, dtype=np.float32)))))
        return min(1.0, rms / max(self.threshold, 1e-6))

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= 1.0
