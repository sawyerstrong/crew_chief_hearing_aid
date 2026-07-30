"""CLI entry point.

    crew_chief_hearing_aid devices        list input devices (run this first)
    crew_chief_hearing_aid doctor         check CrewChief install, bindings, config health
    crew_chief_hearing_aid match "..."    test intent matching without audio
    crew_chief_hearing_aid run --dry-run  full pipeline, logs instead of sending keys
    crew_chief_hearing_aid run            for real
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from .config import Config, default_config_path, load_config, user_config_path
from .logging_setup import setup_logging


def _load(args) -> Config:
    return load_config(user_path=args.config)


def cmd_devices(args) -> int:
    from .audio import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return 1
    print("Input devices (use a distinctive substring as audio.input_device):\n")
    for device in devices:
        print(f"  {device}")
    return 0


def cmd_init_config(args) -> int:
    target = user_config_path()
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_config_path(), target)
    print(f"Wrote {target}")
    print("Edit audio.input_device and the [[intents]] keys, then run:")
    print("  crew_chief_hearing_aid doctor")
    return 0


def cmd_doctor(args) -> int:
    from . import crewchief

    config = _load(args)
    problems = 0

    print("== Config ==")
    for path in config.source_paths:
        print(f"  loaded {path}")
    print(f"  {len(config.intents)} intents defined")
    if len(config.source_paths) == 1:
        print(
            "  ! no user config; run `crew_chief_hearing_aid init-config` "
            "(defaults assume this rig's mic)"
        )

    print("\n== CrewChief ==")
    user_config = crewchief.find_user_config()
    if user_config is None:
        print("  ! CrewChief V4 settings not found")
        problems += 1
    else:
        print(f"  settings: {user_config}")
        settings = crewchief.read_settings(user_config)
        report = crewchief.binding_report(settings)
        print(f"  bound actions: {len(report.bound)}, unbound: {len(report.unbound)}")
        if not report.any_bound:
            print("  ! nothing is bound yet — bind your intent keys in Add/Remove Actions")
            problems += 1
        for note in crewchief.recognition_health(settings):
            print(f"  note: {note}")

    print("\n== Audio ==")
    try:
        from .audio import resolve_input_device

        device = resolve_input_device(config.get("audio", "input_device"))
        print(f"  resolved: {device}")
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        print(f"  ! {exc}")
        problems += 1

    print("\n== Trigger ==")
    ptt_cfg = config.section("ptt")
    if not ptt_cfg.get("enabled", True):
        print("  ! push-to-talk disabled and wake word is out of scope — no trigger")
        problems += 1
    elif not ptt_cfg.get("device_guid"):
        print("  ! no push-to-talk button bound — run `setup-ptt`")
        problems += 1
    else:
        try:
            from .audio.ptt import WheelPTT

            ptt = WheelPTT(str(ptt_cfg["device_guid"]), int(ptt_cfg.get("button_index", -1)))
            ptt.open()
            print(f"  push-to-talk: {ptt._device_name} button {ptt.button_index}")
            ptt.close()
        except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
            print(f"  ! {exc}")
            problems += 1

    print("\n== Tier 4 (LLM) ==")
    if not config.get("llm", "enabled", True):
        print("  disabled in config; tiers 1-3 only")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        # Presence only. Never the value, never a prefix — see AC6.7.
        print(f"  ANTHROPIC_API_KEY set; model {config.get('llm', 'model')}")
    else:
        print("  ANTHROPIC_API_KEY not set — cascade will stop at tier 3")

    print("\n== Compute placement ==")
    # The zero-VRAM property is load-bearing (the GPU is rendering VR), so it
    # gets checked rather than assumed.
    asr_device = config.get("asr", "device", "cpu")
    if asr_device == "cpu":
        print(f"  whisper: cpu / {config.get('asr', 'compute_type', 'int8')}")
    else:
        print(f"  ! whisper is on {asr_device!r} — this allocates VRAM and contends with VR")
        problems += 1
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if providers == ["CPUExecutionProvider"]:
            print("  onnxruntime: CPU-only build (VAD + wake word)")
        else:
            print(f"  ! onnxruntime exposes {providers}; expect VRAM use unless the GPU is hidden")
            problems += 1
    except ImportError:
        print("  ! onnxruntime not installed — VAD and wake word unavailable")
        problems += 1

    print("\n== Output sink ==")
    try:
        from .output import build_sink

        sink = build_sink(
            config.get("output", "sink", "keypress"),
            key_hold_ms=int(config.get("output", "key_hold_ms", 150)),
            pipe_name=config.get("output", "pipe_name", "crewchief-voice"),
        )
        issues = sink.preflight(config.intents)
        if issues:
            problems += len(issues)
            for issue in issues:
                print(f"  ! {issue}")
        else:
            print(f"  {config.get('output', 'sink', 'keypress')} sink ready")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {exc}")
        problems += 1

    print(f"\n{problems} problem(s).")
    return 1 if problems else 0


def cmd_bindings(args) -> int:
    """The sheet you work from while binding keys in CrewChief."""
    config = _load(args)
    print("In CrewChief: Add/Remove Actions -> add each action, then Assign the key.\n")
    width = max(len(i.action) for i in config.intents)
    for intent in config.intents:
        print(f"  {intent.action:<{width}}  ->  {intent.key}")
    print(f"\n{len(config.intents)} actions.")
    print("F13-F24 have no physical keys, so nothing else can ever emit them.")
    return 0


def cmd_setup_ptt(args) -> int:
    from .audio.ptt import JoystickUnavailable, capture_button, list_joysticks

    try:
        devices = list_joysticks()
    except JoystickUnavailable as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    if not devices:
        print("No wheel or joystick detected. Is it plugged in and powered?", file=sys.stderr)
        return 1

    print("Detected devices:")
    for _guid, name, buttons in devices:
        print(f"  {name}  ({buttons} buttons)")

    print(f"\nPress the wheel button you want for push-to-talk ({args.timeout:.0f}s)...")
    try:
        button = capture_button(timeout_s=args.timeout)
    except JoystickUnavailable as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    if button is None:
        print("Timed out; nothing captured.", file=sys.stderr)
        return 1

    print(f"\nCaptured: {button}")
    target = user_config_path()
    print("\nAdd this to your config:")
    print(f"  {target}\n")
    print("[ptt]")
    print("enabled = true")
    print(f'device_guid = "{button.device_guid}"')
    print(f"button_index = {button.button_index}")
    print(
        "\n# Bound by GUID, not index: joystick indices reshuffle when USB devices\n"
        "# re-enumerate, which would silently move push-to-talk to another device."
    )
    return 0


def cmd_import_phrases(args) -> int:
    """Preview what would be imported per action, before it ships."""
    from .config import load_phrase_source

    config = _load(args)
    source = load_phrase_source()
    print(f"phrase corpus: {len(source)} entries\n")
    imported = missing = handwritten = 0
    for intent in config.intents:
        if not intent.sre_key:
            handwritten += 1
            continue
        phrases = source.get(intent.sre_key)
        if phrases is None:
            missing += 1
            print(f"  ! {intent.id}: sre_key {intent.sre_key!r} NOT FOUND")
        else:
            imported += 1
            print(f"  {intent.id}  <-  {intent.sre_key}")
            for p in phrases:
                print(f"      {p!r}")
    print(f"\n{imported} imported, {handwritten} hand-written, {missing} missing.")
    return 1 if missing else 0


def cmd_match(args) -> int:
    from .intent import IntentMatcher, build_embedder

    config = _load(args)
    embedder = build_embedder(
        args.embedder or config.get("intent", "embedder", "model2vec"),
        config.get("intent", "embedder_model"),
    )
    matcher = IntentMatcher(
        config.intents,
        embedder=embedder,
        token_threshold=float(config.get("intent", "token_threshold", 0.72)),
        embed_threshold=float(config.get("intent", "embed_threshold", 0.60)),
        margin=float(config.get("intent", "margin", 0.05)),
    )
    for phrase in args.phrase:
        result = matcher.match(phrase)
        if result.matched:
            print(
                f"{phrase!r}\n  -> {result.intent.id} ({result.intent.key}) "
                f"score={result.score:.3f} via {result.method}"
            )
        else:
            print(
                f"{phrase!r}\n  -> REJECTED ({result.reject_reason}) "
                f"best={result.score:.3f} via {result.method} runner_up={result.runner_up_id}"
            )
    return 0


def cmd_run(args) -> int:
    from .pipeline import Pipeline

    config = _load(args)
    pipeline = Pipeline(config, dry_run=args.dry_run)

    issues = pipeline.preflight()
    if issues:
        for issue in issues:
            print(f"! {issue}", file=sys.stderr)
        if not args.force:
            print("\nRefusing to start. Fix the above or pass --force.", file=sys.stderr)
            return 1

    pipeline.warmup()
    pipeline.run(stop_after_s=args.seconds)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crew_chief_hearing_aid", description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="path to a config.toml override")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio input devices").set_defaults(func=cmd_devices)

    init = sub.add_parser("init-config", help="write a user config from the defaults")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init_config)

    doctor = sub.add_parser("doctor", help="check install, bindings and config")
    doctor.set_defaults(func=cmd_doctor)

    bindings = sub.add_parser("bindings", help="print the CrewChief action->key sheet")
    bindings.set_defaults(func=cmd_bindings)

    ptt = sub.add_parser("setup-ptt", help="capture a wheel button for push-to-talk")
    ptt.add_argument("--timeout", type=float, default=30.0)
    ptt.set_defaults(func=cmd_setup_ptt)

    imp = sub.add_parser("import-phrases", help="preview phrases imported from CrewChief")
    imp.set_defaults(func=cmd_import_phrases)

    match = sub.add_parser("match", help="test intent matching on one or more phrases")
    match.add_argument("phrase", nargs="+")
    match.add_argument("--embedder", default=None, help="override the configured embedder")
    match.set_defaults(func=cmd_match)

    run = sub.add_parser("run", help="run the pipeline")
    run.add_argument("--dry-run", action="store_true", help="log intents instead of sending keys")
    run.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    run.add_argument("--force", action="store_true", help="start despite preflight problems")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    if args.config:
        from pathlib import Path

        args.config = Path(args.config)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
