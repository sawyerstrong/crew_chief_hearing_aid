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
from .intent.phrases import Intent
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
        print(f"  {len(report.bound) + len(report.unbound)} bindable actions known")
        # Deliberately NOT reported as bound/unbound. Measured on a live
        # 4.19.4.0 install: a binding made in the UI and flushed on exit still
        # leaves every *_button_index at -1, so this file does not hold binding
        # state. Claiming otherwise sent me chasing a phantom once already.
        print("  note: binding state is not stored in user.config — check")
        print("        CrewChief's Add/Remove Actions dialog to confirm")
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


def _ask(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer.startswith("y")


def _step(n: int, total: int, title: str) -> None:
    print(f"\n{'=' * 62}\nStep {n}/{total} — {title}\n{'=' * 62}")


def cmd_setup(args) -> int:
    """Guided first-run setup.

    Ordered so the riskiest thing is settled early: you verify one keypress
    reaches CrewChief *before* being asked to hand-bind 27 actions. Finding out
    afterwards that scancodes never arrive would waste the whole exercise.
    """
    from .audio import list_input_devices
    from .userconfig import UserConfigError, ensure_user_config, set_values

    total = 5

    # --- 1. user config -------------------------------------------------
    _step(1, total, "User config")
    try:
        target = ensure_user_config()
    except UserConfigError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1
    print(f"  {target}")

    # --- 2. microphone --------------------------------------------------
    _step(2, total, "Microphone")
    print("CrewChief's own recogniser reads whatever Windows calls default, which")
    print("is how it ends up listening to a webcam. Pick explicitly.\n")
    try:
        devices = list_input_devices()
    except Exception as exc:  # noqa: BLE001 - setup reports, never crashes
        print(f"  ! could not enumerate audio devices: {exc}")
        devices = []

    if devices:
        for i, device in enumerate(devices):
            print(f"  [{i}] {device.name}")
        try:
            raw = input("\nWhich number? (enter to skip) ").strip()
        except EOFError:
            raw = ""
        if raw.isdigit() and int(raw) < len(devices):
            chosen = devices[int(raw)]
            # Store a name substring, never the index — indices reshuffle.
            set_values({"audio": {"input_device": chosen.name}})
            print(f"  set audio.input_device = {chosen.name!r}")
        else:
            print("  skipped")

    # --- 3. push-to-talk ------------------------------------------------
    _step(3, total, "Push-to-talk button")
    updates = _capture_ptt(args.timeout)
    if updates is None:
        print("\n! No button captured. Re-run `setup-ptt` when the wheel is connected.")
    else:
        set_values(updates)
        print("  written")

    # --- 4. bind the actions --------------------------------------------
    _step(4, total, "Bind actions in CrewChief")
    config = _load(args)
    print("F13-F24 have no physical keys — that is why nothing else on the system")
    print("can ever emit them, and why CrewChief's Assign dialog cannot capture")
    print("them from your keyboard. We inject each key instead, through the same")
    print("SendInput path used at runtime — so a successful bind also proves the")
    print("runtime sink works.\n")
    print(f"There are {len(config.intents)} actions. Binding each is three clicks in")
    print("CrewChief, so doing all of them in one sitting is a real chunk of time.")
    print("The first 12 (plain F13-F24) are the race-critical ones; the rest are")
    print("toggles and rally features you can add later.\n")
    print("  [a] all 27")
    print("  [c] core 12 only (recommended for a first pass)")
    print("  [s] skip — bind later with `cchear bind-all`")

    try:
        choice = input("\nWhich? [a/c/s] ").strip().lower() or "c"
    except EOFError:
        choice = "s"

    if choice.startswith("s"):
        print("\n  Skipped. The action -> key sheet:\n")
        cmd_bindings(args)
        print("\nRun `cchear bind-all` when CrewChief is open.")
    else:
        if choice.startswith("c"):
            args.only = [i.id for i in config.intents if "+" not in i.key]
            print(f"\n  Binding the {len(args.only)} core actions.\n")
        else:
            args.only = None
            print(f"\n  Binding all {len(config.intents)} actions.\n")
        args.delay = 3
        print("Open CrewChief now, but do NOT press Start — it greys out Assign")
        print("while a session is running. The main button should read 'Start'.")
        if _ask("CrewChief open and stopped?", default=True):
            if cmd_bind_all(args) != 0:
                # bind-all stops on a failed first capture; nothing downstream
                # can work until that is fixed, so do not pretend otherwise.
                return 1
        else:
            print("  Skipped — run `cchear bind-all` once CrewChief is open.")

    # --- 5. check -------------------------------------------------------
    _step(5, total, "Check")
    cmd_doctor(args)

    print("\nWhen doctor is clean:")
    print("  crew_chief_hearing_aid run --dry-run   # logs intents, sends no keys")
    print("  crew_chief_hearing_aid run")
    return 0


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


def _capture_ptt(timeout: float) -> dict[str, dict[str, object]] | None:
    """Shared by `setup-ptt` and the `setup` wizard. None means give up."""
    from .audio.ptt import JoystickUnavailable, capture_button, list_joysticks

    try:
        devices = list_joysticks()
    except JoystickUnavailable as exc:
        print(f"! {exc}", file=sys.stderr)
        return None

    if not devices:
        print("No wheel or joystick detected. Is it plugged in and powered?", file=sys.stderr)
        return None

    print("Detected devices:")
    for _guid, name, buttons in devices:
        print(f"  {name}  ({buttons} buttons)")

    print(f"\nPress the wheel button you want for push-to-talk ({timeout:.0f}s)...")
    try:
        button = capture_button(timeout_s=timeout)
    except JoystickUnavailable as exc:
        print(f"! {exc}", file=sys.stderr)
        return None

    if button is None:
        print("Timed out; nothing captured.", file=sys.stderr)
        return None

    print(f"\nCaptured: {button}")
    return {
        "ptt": {
            "enabled": True,
            "device_guid": button.device_guid,
            "button_index": button.button_index,
        }
    }


def cmd_setup_ptt(args) -> int:
    from .userconfig import UserConfigError, describe_changes, set_values

    updates = _capture_ptt(args.timeout)
    if updates is None:
        return 1

    if args.print_only:
        print(f"\nAdd this to {user_config_path()}:\n")
        print(describe_changes(updates))
        return 0

    try:
        target = set_values(updates)
    except UserConfigError as exc:
        print(f"! {exc}", file=sys.stderr)
        print(f"\nAdd this manually to {user_config_path()}:\n", file=sys.stderr)
        print(describe_changes(updates))
        return 1

    print(f"\nWritten to {target}")
    print("Bound by device GUID, not index — a replugged wheel is detected rather")
    print("than silently mapping push-to-talk onto a different device.")
    return 0


def cmd_send_key(args) -> int:
    """Inject a keypress while CrewChief's Assign dialog is listening.

    F13-F24 have no physical keys, which is exactly why they cannot collide
    with a sim or Windows binding -- and exactly why CrewChief's
    press-the-key-to-bind dialog cannot capture them from hardware. So we send
    it ourselves, through the same SendInput path used at runtime.

    That equivalence is the useful part: if the bind lands, the runtime sink is
    proven to reach CrewChief. Binding and verifying become one step.
    """
    import time

    from .output.keypress import SCAN_CODES, KeypressSink, normalize_key

    config = _load(args)
    intent = config.intent_by_id(args.key)
    if intent is not None:
        key_spec, label = intent.key, intent.action
    else:
        key_spec, label = args.key, "(raw key)"
        try:
            normalize_key(key_spec)  # raises on an unparseable spec
        except ValueError as exc:
            print(f"! {exc}", file=sys.stderr)
            return 1

    probe = Intent(
        id="__send__", action=label, key=key_spec, phrases=("x",), description="x"
    )
    sink = KeypressSink(hold_ms=int(config.get("output", "key_hold_ms", 150)))
    problems = sink.preflight([probe])
    for problem in problems:
        print(f"! {problem}", file=sys.stderr)
    if problems:
        print(f"\nKnown keys: {', '.join(sorted(SCAN_CODES))}", file=sys.stderr)
        return 1

    print(f"Key    : {normalize_key(key_spec)}")
    print(f"Action : {label}")
    print("\nCrewChief must be OPEN but STOPPED — Assign is greyed out while a")
    print("session is running (the main button should read 'Start').")
    print("Then: select the action, click Assign, and leave the dialog waiting.")
    for remaining in range(args.delay, 0, -1):
        print(f"  sending in {remaining}... ", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    delivered = sink.fire(probe)
    sink.close()
    if not delivered:
        print("! send failed", file=sys.stderr)
        return 1

    print(f"Sent {normalize_key(key_spec)}.")
    print("\nCheck the Assigned actions list. The row should have changed from")
    print(f"  '{label} not assigned'")
    print("to")
    print(f"  '{label} assigned to Keyboard, button: NN'")
    print("\nIf it did, the binding is done AND the runtime sink is proven — same")
    print("SendInput path. If it still says 'not assigned', CrewChief is not")
    print("seeing injected scancodes; stop and fix that before going further.")
    return 0


def cmd_bind_all(args) -> int:
    """Walk every action, injecting its key while CrewChief's dialog waits.

    Why not write CrewChief's user.config directly? It is tempting -- 27
    dialogs is tedious -- but:

      * CrewChief rewrites user.config on exit, so anything written while it
        runs is discarded.
      * The keyboard `button_index` encoding is undocumented, and a wrong guess
        writes a binding that looks right and silently never fires.
      * Two of the 27 actions have no settings entry at all until CrewChief
        creates one.

    Letting CrewChief capture an injected key makes CrewChief author its own
    encoding, which is correct by construction. It also proves the runtime sink
    on the first action, since it is the same SendInput path.
    """
    import time

    from .output.keypress import KeypressSink, normalize_key

    config = _load(args)
    intents = config.intents
    if args.only:
        wanted = set(args.only)
        intents = [i for i in intents if i.id in wanted]
        missing = wanted - {i.id for i in intents}
        if missing:
            print(f"! unknown intent ids: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    sink = KeypressSink(hold_ms=int(config.get("output", "key_hold_ms", 150)))
    problems = sink.preflight(intents)
    for problem in problems:
        print(f"! {problem}", file=sys.stderr)
    if problems:
        return 1

    print(f"Binding {len(intents)} actions.\n")
    print("CrewChief must be OPEN but STOPPED — it greys out Assign while a")
    print("session is running. The main button should read 'Start', not 'Stop'.")
    print("Bind everything first, then Start it.\n")
    print("For each action, in CrewChief:")
    print("  Add/Remove Actions -> add the action -> select it -> click Assign")
    print("Then press Enter here and the key is injected into the waiting dialog.\n")
    print("Enter = send, s = skip, q = stop.\n")

    done, skipped = 0, 0
    for n, intent in enumerate(intents, 1):
        key = normalize_key(intent.key)
        print(f"[{n}/{len(intents)}] {intent.action}")
        print(f"          -> {key}")
        try:
            answer = input("          Assign dialog waiting? [Enter/s/q] ").strip().lower()
        except EOFError:
            answer = "q"
        if answer.startswith("q"):
            print("\nStopped.")
            break
        if answer.startswith("s"):
            skipped += 1
            print()
            continue

        for remaining in range(args.delay, 0, -1):
            print(f"          sending in {remaining}... ", end="\r", flush=True)
            time.sleep(1)
        print(" " * 44, end="\r")
        sink.fire(intent)
        print(f"          sent {key}")
        done += 1

        if n == 1:
            # The first one is the real test. If CrewChief did not capture it,
            # nothing downstream can work and 26 more will not help.
            print("\n          Check the Assigned actions list in CrewChief.")
            print("          The row should have changed from:")
            print(f"            {intent.action} not assigned")
            print("          to:")
            print(f"            {intent.action} assigned to Keyboard, button: NN")
            if not _ask("\n          Does it say 'assigned to Keyboard'?", default=False):
                print("\nStop. CrewChief is not seeing injected scancodes, so the")
                print("runtime sink cannot work either. Fix that before continuing.")
                sink.close()
                return 1
            print("          Confirmed — binding and runtime sink both work.\n")
        else:
            print()

    sink.close()
    print(f"\n{done} sent, {skipped} skipped.")
    print("Run `cchear doctor` to confirm CrewChief recorded them.")
    print("CrewChief writes its config on exit — close it before checking.")
    return 0


def cmd_test_key(args) -> int:
    """Fire one keypress so you can confirm CrewChief actually receives it.

    This is the highest-risk unknown in the whole project: if synthetic
    scancodes do not reach CrewChief, every one of the 27 bindings is worthless.
    Settle it with one action before hand-binding the rest.
    """
    import time

    from .output import build_sink

    config = _load(args)
    intent = config.intent_by_id(args.intent)
    if intent is None:
        print(f"! no intent {args.intent!r}. Known ids:", file=sys.stderr)
        for i in config.intents:
            print(f"    {i.id}", file=sys.stderr)
        return 1

    sink = build_sink(
        "log" if args.dry_run else config.get("output", "sink", "keypress"),
        key_hold_ms=int(config.get("output", "key_hold_ms", 150)),
        pipe_name=config.get("output", "pipe_name", "crewchief-voice"),
    )
    problems = sink.preflight([intent])
    for problem in problems:
        print(f"! {problem}", file=sys.stderr)
    if problems and not args.force:
        return 1

    print(f"Action : {intent.action}")
    print(f"Key    : {intent.key}")
    print("\nMake sure CrewChief is running and this action is bound to that key.")
    for remaining in range(args.delay, 0, -1):
        print(f"  firing in {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")

    delivered = sink.fire(intent)
    sink.close()
    print(f"Sent {intent.key}." if delivered else "! send failed")
    print("\nDid CrewChief respond? If not, the scancode is not reaching it —")
    print("stop here and fix the sink before binding the other actions.")
    return 0 if delivered else 1


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

    setup = sub.add_parser("setup", help="guided first-run setup (start here)")
    setup.add_argument("--timeout", type=float, default=30.0)
    setup.set_defaults(func=cmd_setup)

    ptt = sub.add_parser("setup-ptt", help="capture a wheel button for push-to-talk")
    ptt.add_argument("--timeout", type=float, default=30.0)
    ptt.add_argument(
        "--print-only",
        action="store_true",
        help="print the config block instead of writing it",
    )
    ptt.set_defaults(func=cmd_setup_ptt)

    sendkey = sub.add_parser(
        "send-key", help="inject a key into CrewChief's waiting Assign dialog"
    )
    sendkey.add_argument("key", help="intent id, or a raw key like F13 / ctrl+F14")
    sendkey.add_argument("--delay", type=int, default=4)
    sendkey.set_defaults(func=cmd_send_key)

    bindall = sub.add_parser(
        "bind-all", help="walk every action, injecting its key as you click Assign"
    )
    bindall.add_argument("--delay", type=int, default=3)
    bindall.add_argument("--only", nargs="*", help="restrict to these intent ids")
    bindall.set_defaults(func=cmd_bind_all)

    testkey = sub.add_parser(
        "test-key", help="fire one keypress to check CrewChief receives it"
    )
    testkey.add_argument("intent", help="intent id (see `bindings`)")
    testkey.add_argument("--delay", type=int, default=5, help="countdown before firing")
    testkey.add_argument("--dry-run", action="store_true")
    testkey.add_argument("--force", action="store_true")
    testkey.set_defaults(func=cmd_test_key)

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
