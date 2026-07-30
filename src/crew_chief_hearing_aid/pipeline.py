"""Orchestration: wake word -> endpoint -> transcribe -> match -> fire.

State machine, one frame at a time:

    IDLE      every frame goes to the wake-word detector
    LISTENING accumulating; VAD decides when the user stopped
    (then)    transcribe, match, dispatch, back to IDLE

Deliberately single-threaded past capture. The whole chain is ~200-400ms and
concurrency would buy nothing but races against the audio callback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .asr import WhisperTranscriber
from .audio import AudioCapture, CaptureConfig, resolve_input_device
from .audio.vad import EnergyVAD, SileroVAD
from .audio.wakeword import AlwaysOpenDetector, WakeWordDetector
from .config import Config
from .intent import IntentMatcher, build_embedder
from .logging_setup import UtteranceLog
from .output import build_sink

log = logging.getLogger(__name__)


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"


@dataclass
class Stats:
    wake_events: int = 0
    transcribed: int = 0
    fired: int = 0
    rejected: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> str:
        if self.latencies_ms:
            arr = np.array(self.latencies_ms)
            p50, p95 = np.percentile(arr, 50), np.percentile(arr, 95)
            timing = f", latency p50={p50:.0f}ms p95={p95:.0f}ms"
        else:
            timing = ""
        return (
            f"wake={self.wake_events} transcribed={self.transcribed} "
            f"fired={self.fired} rejected={self.rejected}{timing}"
        )


class Pipeline:
    def __init__(self, config: Config, *, dry_run: bool = False) -> None:
        self.config = config
        self.stats = Stats()

        audio_cfg = config.section("audio")
        self.capture_config = CaptureConfig(
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            block_ms=int(audio_cfg.get("block_ms", 32)),
            preroll_ms=int(audio_cfg.get("preroll_ms", 512)),
        )
        self.silence_timeout_ms = int(audio_cfg.get("silence_timeout_ms", 900))
        self.max_utterance_ms = int(audio_cfg.get("max_utterance_ms", 6000))
        self.device = resolve_input_device(audio_cfg.get("input_device"))

        ww_cfg = config.section("wakeword")
        if ww_cfg.get("enabled", True):
            self.wakeword = WakeWordDetector(
                model=ww_cfg.get("model", "hey_jarvis"),
                threshold=float(ww_cfg.get("threshold", 0.55)),
                cooldown_ms=int(ww_cfg.get("cooldown_ms", 1500)),
                sample_rate=self.capture_config.sample_rate,
            )
        else:
            self.wakeword = AlwaysOpenDetector()

        vad_cfg = config.section("vad")
        if vad_cfg.get("enabled", True):
            self.vad = SileroVAD(
                sample_rate=self.capture_config.sample_rate,
                threshold=float(vad_cfg.get("threshold", 0.5)),
            )
        else:
            self.vad = EnergyVAD(sample_rate=self.capture_config.sample_rate)

        asr_cfg = config.section("asr")
        self.transcriber = WhisperTranscriber(
            model=asr_cfg.get("model", "tiny.en"),
            device=asr_cfg.get("device", "cpu"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            beam_size=int(asr_cfg.get("beam_size", 1)),
            initial_prompt=asr_cfg.get("initial_prompt"),
        )

        intent_cfg = config.section("intent")
        embedder = build_embedder(
            intent_cfg.get("embedder", "model2vec"), intent_cfg.get("embedder_model")
        )
        self.matcher = IntentMatcher(
            config.intents,
            embedder=embedder,
            token_threshold=float(intent_cfg.get("token_threshold", 0.72)),
            embed_threshold=float(intent_cfg.get("embed_threshold", 0.60)),
            margin=float(intent_cfg.get("margin", 0.05)),
        )

        out_cfg = config.section("output")
        sink_kind = "log" if dry_run else out_cfg.get("sink", "keypress")
        self.sink = build_sink(
            sink_kind,
            key_hold_ms=int(out_cfg.get("key_hold_ms", 150)),
            pipe_name=out_cfg.get("pipe_name", "crewchief-voice"),
        )

        log_cfg = config.section("logging")
        self.utterance_log = UtteranceLog(log_cfg.get("dir", "logs"))

        self.state = State.IDLE
        self._buffer: list[np.ndarray] = []
        self._utterance_ms = 0.0
        self._silence_ms = 0.0

    # -- lifecycle ------------------------------------------------------

    def preflight(self) -> list[str]:
        return list(self.sink.preflight(self.config.intents))

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.matcher.warmup()
        self.wakeword.warmup()
        self.transcriber.warmup()
        log.info("warmup complete in %.0fms", (time.perf_counter() - t0) * 1000)

    # -- frame handling -------------------------------------------------

    def _begin_listening(self, capture: AudioCapture) -> None:
        self.state = State.LISTENING
        self.stats.wake_events += 1
        self.vad.reset()
        # Pre-roll recovers the command words spoken before the wake word was
        # confirmed, which is most of them when people run the two together.
        self._buffer = [capture.snapshot_preroll()]
        self._utterance_ms = len(self._buffer[0]) / self.capture_config.sample_rate * 1000
        self._silence_ms = 0.0
        log.debug("listening (preroll %.0fms)", self._utterance_ms)

    def _finish_utterance(self) -> None:
        audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
        self._buffer = []
        self.state = State.IDLE

        if audio.size < self.capture_config.sample_rate * 0.2:
            log.debug("utterance too short (%d samples); discarded", audio.size)
            return

        t0 = time.perf_counter()
        transcript = self.transcriber.transcribe(audio, self.capture_config.sample_rate)
        self.stats.transcribed += 1

        if not transcript.text:
            self.stats.rejected += 1
            self.utterance_log.write(
                {"transcript": "", "intent": None, "reject_reason": "empty_transcript"}
            )
            log.info("no speech transcribed")
            return

        result = self.matcher.match(transcript.text)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats.latencies_ms.append(latency_ms)

        record = result.as_log_record()
        record.update(
            {
                "asr_ms": round(transcript.duration_ms, 1),
                "total_ms": round(latency_ms, 1),
                "audio_ms": round(transcript.audio_ms, 1),
                "no_speech_prob": transcript.no_speech_prob,
            }
        )

        if result.matched:
            delivered = self.sink.fire(result.intent)
            record["delivered"] = delivered
            self.stats.fired += 1
            log.info(
                "%r -> %s (%.2f via %s) in %.0fms",
                transcript.text,
                result.intent.id,
                result.score,
                result.method,
                latency_ms,
            )
        else:
            self.stats.rejected += 1
            log.info(
                "%r -> no match (best %.2f via %s, %s)",
                transcript.text,
                result.score,
                result.method,
                result.reject_reason,
            )

        self.utterance_log.write(record)

    def _handle_frame(self, frame: np.ndarray, capture: AudioCapture) -> None:
        if self.state is State.IDLE:
            if self.wakeword.detect(frame):
                self._begin_listening(capture)
            return

        self._buffer.append(frame)
        frame_ms = len(frame) / self.capture_config.sample_rate * 1000
        self._utterance_ms += frame_ms

        if self.vad.is_speech(frame):
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        if self._silence_ms >= self.silence_timeout_ms:
            self._finish_utterance()
        elif self._utterance_ms >= self.max_utterance_ms:
            log.debug("utterance hit max length; cutting off")
            self._finish_utterance()

    # -- main loop ------------------------------------------------------

    def run(self, stop_after_s: float | None = None) -> Stats:
        started = time.monotonic()
        with AudioCapture(self.device, self.capture_config) as capture:
            log.info("listening — say the wake word")
            try:
                while True:
                    if stop_after_s and (time.monotonic() - started) > stop_after_s:
                        break
                    frame = capture.read(timeout=0.5)
                    if frame is None:
                        continue
                    self._handle_frame(frame, capture)
            except KeyboardInterrupt:
                log.info("interrupted")
            finally:
                self.sink.close()
        log.info("session: %s", self.stats.summary())
        return self.stats
