"""Model artifact fetching.

Models are not committed. Git LFS bandwidth quotas make it a bad host for a
~40MB Whisper checkpoint plus embeddings, and the files change far less often
than the code. They are fetched on first run into %LOCALAPPDATA%\\crew_chief_hearing_aid\\models
and verified by SHA256 — an interrupted download that leaves a truncated ONNX
file otherwise surfaces as a baffling runtime error much later.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    filename: str
    sha256: str | None = None


# Whisper and model2vec weights are handled by their own libraries' caches;
# only models loaded directly through onnxruntime need fetching here.
MODELS: dict[str, ModelSpec] = {
    "silero_vad": ModelSpec(
        name="silero_vad",
        url="https://github.com/snakers4/silero-vad/raw/v5.1/src/silero_vad/data/silero_vad.onnx",
        filename="silero_vad.onnx",
        # Populate after first download via `crew_chief_hearing_aid models --print-hashes`,
        # then commit. Left None so a version bump does not hard-fail before
        # you have had a chance to verify the new artifact.
        sha256=None,
    ),
}


def models_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    path = Path(base) / "crew_chief_hearing_aid" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(name: str, *, force: bool = False) -> Path:
    try:
        spec = MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}") from None

    target = models_dir() / spec.filename
    if target.exists() and not force:
        if spec.sha256 and sha256_of(target) != spec.sha256:
            log.warning("%s failed checksum; re-downloading", target)
        else:
            return target

    # Download to a temp path and rename, so an interrupted transfer can never
    # leave a truncated file that looks valid to the next run.
    tmp = target.with_suffix(target.suffix + ".part")
    log.info("downloading %s -> %s", spec.url, target)
    with urllib.request.urlopen(spec.url, timeout=60) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)

    if spec.sha256:
        actual = sha256_of(tmp)
        if actual != spec.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{spec.filename} checksum mismatch: got {actual}")

    tmp.replace(target)
    return target
