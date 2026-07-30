from .base import LogSink, Sink

__all__ = ["LogSink", "Sink", "build_sink"]


def build_sink(kind: str, **kwargs) -> Sink:
    kind = (kind or "log").lower()
    if kind == "log":
        return LogSink()
    if kind == "keypress":
        from .keypress import KeypressSink

        return KeypressSink(hold_ms=kwargs.get("key_hold_ms", 150))
    if kind == "pipe":
        from .pipe import NamedPipeSink

        return NamedPipeSink(pipe_name=kwargs.get("pipe_name", "crewchief-voice"))
    raise ValueError(f"unknown sink {kind!r}")
