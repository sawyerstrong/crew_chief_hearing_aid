"""Console logging plus a JSONL utterance log.

The JSONL file is the point. Every utterance records the transcript, the chosen
intent, the score, which tier matched, and the runner-up. Without it there is no
way to tell a bad transcript from a badly-tuned threshold — which is exactly the
blind spot CrewChief's own recogniser leaves you in, since it only ever reports
its own grammar match and never what you actually said.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # openWakeWord and faster-whisper are chatty at INFO.
    for noisy in ("openwakeword", "faster_whisper", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class UtteranceLog:
    def __init__(self, directory: str | Path = "logs") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d")
        self.path = self.dir / f"utterances-{stamp}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **record}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # never let logging kill a race
            log.warning("could not write utterance log: %s", exc)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
