"""Whisper transcription via faster-whisper (CTranslate2).

CPU, int8, single-threaded by default. The GPU is rendering VR at 90fps and is
not available; a 2-second utterance runs in roughly 150ms on CPU with tiny.en,
which is well inside the budget.

Model size is deliberately small because the downstream stage *classifies*
rather than reads. A transcript of "whats the gap to the guy in front" still
matches the right intent even with a word error, which is the whole reason a
closed grammar is not needed here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_ms: float
    audio_ms: float
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class WhisperTranscriber:
    def __init__(
        self,
        model: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        initial_prompt: str | None = None,
        cpu_threads: int = 2,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.cpu_threads = cpu_threads
        self._model = None

        if device != "cpu":
            # Not forbidden -- it's your GPU -- but the whole VR rationale rests
            # on this staying "cpu", so it must not change silently. tiny.en is
            # 39M params (~78MB fp16), while a CUDA context plus cuBLAS/cuDNN
            # kernels typically costs 300-600MB: the overhead dwarfs the model,
            # and the SM time comes out of the renderer's frame budget.
            log.warning(
                "asr.device is %r, not 'cpu'. This allocates VRAM and contends with "
                "the sim for SM time; expect frame drops in VR.",
                device,
            )

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            log.info(
                "whisper %s loaded in %.0fms", self.model_name, (time.perf_counter() - t0) * 1000
            )
        return self._model

    def warmup(self) -> None:
        """Load and run one silent inference.

        Without this the first command of a session pays model load plus
        first-call graph setup — the one time you least want the latency.
        """
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        model = self._load()
        audio = np.asarray(audio, dtype=np.float32)
        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            audio,
            beam_size=self.beam_size,
            language="en",
            initial_prompt=self.initial_prompt,
            # Our own VAD already endpointed this clip; running Whisper's too
            # can clip short commands.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = (time.perf_counter() - t0) * 1000

        avg_logprob = (
            float(np.mean([s.avg_logprob for s in segments])) if segments else None
        )
        no_speech = float(np.mean([s.no_speech_prob for s in segments])) if segments else None

        return Transcript(
            text=text,
            duration_ms=elapsed,
            audio_ms=len(audio) / sample_rate * 1000,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech,
        )
