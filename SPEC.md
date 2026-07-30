# SPEC: crew_chief_hearing_aid v1

Status: **draft, awaiting approval**
Repo: https://github.com/sawyerstrong/crew_chief_hearing_aid (public)
Target: CrewChief V4 4.19.4.0, iRacing, Windows 11, VR, RTX 5080

---

## 1. Problem

CrewChief's built-in speech recognition is a closed-grammar SAPI engine. Evidence from a live session log on the target machine:

```
15:16:00.945 : System recogniser recognised : "lap time", Confidence = 0.989
15:16:01.418 : Sound: acknowledge/no_data
15:16:09.392 : System recogniser recognised : "lap time", Confidence = 0.653
15:16:09.392 : Confidence 0.653 is below the minimum threshold of 0.750
```

Three distinct defects, only the third of which is a tuning problem:

| # | Defect | Mechanism |
|---|---|---|
| P1 | Identical utterances score 0.65–0.99 | Acoustic confidence is noisy; a fixed threshold sits inside the variance band |
| P2 | Confident wrong matches | A closed grammar must return its nearest entry — it has no "none of the above" |
| P3 | Short aliases capture long sentences | `WHAT_WAS_MY_LAST_LAP_TIME` registers the bare alias `lap time`, a substring of many longer commands |

P2 is the one that motivates the project. P1 and P3 have zero-code mitigations inside CrewChief (lower `minimum_voice_recognition_confidence_system_sre` to 0.60; set `disable_alternative_voice_commands = True`) and should be applied regardless as the measurement baseline.

## 2. Goals

- **G1** Voice control of CrewChief more accurate than its native SRE, measured, not asserted.
- **G2** Consume **no wheel buttons for commands**. One button for push-to-talk is an accepted spend (D9) — the constraint was never "zero buttons", it was "don't burn 27 of them".
- **G3** Require no modification to CrewChief; survive its auto-updates.
- **G4** Portable: clone from GitHub, one-shot install, run on a different machine.
- **G5** Reach **every** CrewChief action exposed in Add/Remove Actions.
- **G6** Produce a diagnostic record that makes misses debuggable — the thing CrewChief's SRE structurally cannot give you, since it only ever reports its own grammar match and never what you actually said.

## 3. Non-goals

- **N1** Reading iRacing telemetry directly. This project routes to CrewChief; it does not compute answers.
- **N2** Cloud *audio* processing. Whisper stays local — only ~15 tokens of transcript ever leave the machine.
- **N3** Replacing CrewChief's spotter, or any always-on messaging.
- **N4** Redistribution as a product. Public repo, single operator.

## 4. Resolved decisions

| ID | Decision | Rationale |
|---|---|---|
| D1 | Output is **synthetic keypress only**. No CrewChief fork in v1. | Keeps the auto-updater and avoids owning a C# fork across CrewChief's release cadence (4.19.1 → 4.19.4 already observed on this machine). |
| D2 | Trigger is **push-to-talk only**. Wake word deferred out of v1. | Removes the entire false-positive surface (Discord bleed, sim audio) and — see §5.3 — makes the pipeline *faster than the thing it replaces*, because PTT release endpoints directly and the 900ms VAD timeout disappears. The wake-word code stays in the tree, disabled. |
| **D9** | **PTT is a wheel button, captured during setup.** Resolves O3. | A keyboard key is unreachable by touch in a headset. Setup prompts "press the button you want", captures device GUID + button index, and persists — the same flow CrewChief uses. Requires DirectInput/joystick polling, not a keyboard hook. |
| **D10** | **Phrases are imported from CrewChief's SRE config; descriptions are hand-authored metadata.** Resolves O6. | CrewChief's `speech_recognition_config.txt` already carries good phrasings, and the subset-dropping parser is built and tested. But it has no tool *descriptions*, which the Haiku stage needs — and no mapping from SRE key to bindable action. That mapping plus descriptions is the hand-authored layer. |
| D3 | Intent routing includes an **LLM tool-calling stage**. | Multi-intent handling and paraphrase headroom; the project is explicitly partly for its own sake. |
| D4 | Done means **works in a live race**, measured from the utterance log. | Bench testing does not exercise cockpit noise, VR load, or Discord bleed. |
| D5 | Keys are **F13–F24, with Ctrl/Shift/Alt modifiers**. | No physical keyboard emits F13–F24, so collision is impossible. 27 actions need more than 12 keys; modifiers give 48 slots. |
| D6 | Audio devices resolved by **name substring, never index**. | Four active mics plus NVIDIA Broadcast virtual endpoints; indices reshuffle across reboots and replugs. |
| **D7** | **The LLM is Claude Haiku 4.5 via API, not a local model.** | Zero VRAM and zero SM contention with the VR renderer — the binding constraint on a 5080 already running iRacing. Also better quality than any 0.6–1B local model. Supersedes the earlier Ollama plan. |
| **D8** | **LLM runs as a cascade (tier 4), not always-on.** | Resolves O1. Tiers 1–3 answer in ~5ms locally and handle the common case; the API is only consulted on reject. Keeps the happy path fast and the network off the critical path for most commands. |

## 5. Scope consequences that need acknowledging

### 5.1 The full action set is 28; gap ahead/behind is still not in it

Enumerated from CrewChief's Add/Remove Actions dialog (28 entries) and cross-checked against the 25 `*_button_index` settings present in `user.config`. The two lap-time actions appear in the dialog but have no setting yet, because CrewChief only writes one once an action is added.

**"Talk to Crew Chief"** (currently bound to Keyboard button 91) is CrewChief's own push-to-talk and is irrelevant here — we bypass its SRE entirely. That leaves **27 actions to map**, which D5's 48 key slots cover.

The original ask — lap time ahead and behind — **is** in the set:

- ✅ `What's the car ahead's last lap time`
- ✅ `What's the car behind's last lap time`
- ❌ **gap** ahead / **gap** behind — voice-only, still not button-bindable

So D1 delivers G5 as scoped (every *bindable* action) but not literally every CrewChief voice command. Gap queries remain behind the named-pipe fork, which D1 puts out of scope. `NamedPipeSink` is written and sits behind the same `Sink` interface, so lifting this later is a config change plus the C# work — not a rewrite.

Several of the 27 are poor fits for voice regardless (Reset VR view wants a real button; the pace-notes and stage-recce actions are rally-only). They get mapped anyway per G5; prune in config.

### 5.2 D1 and D3 interact: argument extraction is still dead

Tool-calling earns its cost through three things. Under keypress output only two survive:

| LLM capability | Survives D1? | Why |
|---|---|---|
| Multi-intent ("fuel and lap time ahead?") | ✅ | Fire two keypresses in sequence; parallel tool use is on by default |
| Paraphrase robustness | ✅ | Better than cosine on unusual phrasings |
| **Argument extraction** ("what's **P4's** last lap") | ❌ | A keypress carries no payload. There is no channel for the argument. |

Argument extraction returns only with the fork. This is a real halving of what tool-calling is normally worth, and it is the strongest argument for revisiting D1 later.

### 5.3 Latency budget

Recomputed after D2 dropped the wake word. **PTT release is the endpoint, so the 900ms VAD timeout leaves the critical path entirely** — VAD is no longer needed for endpointing at all.

| Stage | Budget | Notes |
|---|---|---|
| PTT press → capture start | ~0ms | Ring buffer already running |
| PTT release → endpoint | **~0ms** | Button release *is* the endpoint. Was 900ms of VAD silence-timeout. |
| Whisper tiny.en (CPU int8) | ~150ms | For a 2s utterance |
| Tiers 1–3 (exact / coverage / embedding) | ~5ms | Local, always runs |
| Tier 4 (Haiku 4.5) | ~600ms–1s *estimated* | Only on cascade miss. **Unverified — must be measured on this connection.** |
| CrewChief response | ~1s | Outside our control |
| **Total, happy path** | **~1.2s** | Tiers 1–3 hit |
| **Total, cascade to Haiku** | **~1.8–2.2s** | |

CrewChief's native SRE path is roughly 1.3s. **The happy path is now faster than the thing it replaces**, while also buying a real reject — that inverts the earlier trade, where the project cost ~800ms for correctness. The cascade path costs ~0.5–0.9s over native and is paid only when the local tiers fail.

This is the single largest design improvement in the spec so far, and it came from *removing* a component rather than adding one.

### 5.4 Cost and API mechanics (D7)

`claude-haiku-4-5`, $1.00/M input, $5.00/M output.

Per command: ~1,300–1,500 input tokens (27 tool definitions with mostly empty schemas + short system prompt), ~15 tokens transcript, ~40 output tokens. **≈ $0.0018 per command.** At 30 commands per racing hour that is about a nickel. Cost is not a design constraint here.

Three API specifics that are load-bearing:

- **Prompt caching does not apply.** Haiku 4.5's minimum cacheable prefix is **4,096 tokens**; the tool-definition prefix lands near 1,300. Below the minimum it silently does not cache — no error, `cache_creation_input_tokens: 0` forever. Do not add `cache_control` and do not pad the prefix to reach the threshold.
- **The SDK default timeout is 10 minutes.** Untouched, a network stall parks the pipeline mid-race. Set ~2.5s explicitly with `max_retries=0` — falling through to the tier-3 result beats retrying.
- **`tool_choice={"type": "any"}` plus an explicit `no_match` tool.** Forcing a tool call makes an invalid tool name unrepresentable, which satisfies AC6.1 at the API layer rather than via a local grammar. The `no_match` tool is mandatory: without it, forcing a call reintroduces exactly the P2 failure this project exists to escape.

Haiku 4.5 does not accept the `effort` parameter, and omitting `thinking` means no thinking — both correct for this workload.

## 6. Architecture

```
mic ──► ring buffer (512ms pre-roll)
         │
         ├─► openWakeWord ─────────┐
         ├─► PTT key state ────────┤ (either fires)
         │                         ▼
         └───────────────► accumulate + Silero VAD endpointing
                                   │
                                   ▼
                     Whisper tiny.en (CPU, int8, 2 threads)
                                   │
                                   ▼
                    ┌── tier 1  exact canonical match ──┐  µs   ┐
                    ├── tier 2  symmetric token F1 ─────┤  µs   │ local
                    ├── tier 3  embedding cosine ───────┤  ~5ms ┘
                    └── tier 4  Haiku 4.5 tool call ────┘  ~600ms-1s, network
                                   │
                    best ≥ threshold AND margin over runner-up?
                             yes ──┴── no ──► reject, log, stay silent
                              │
                              ▼
                     Sink.fire(intent) ×N   [keypress + modifiers]
```

Tier 2 is symmetric (F1 over content-token sets), **not** containment. Containment would score `lap time` at 1.0 against "what's the lap time of the car ahead" — reproducing P3 exactly. F1 scores it 0.5 and declines. Enforced by test.

## 7. Components and acceptance criteria

### C1 Audio capture — **built**
- AC1.1 Device resolved by name substring; unresolvable name fails at startup with the available list. ✅ tested
- AC1.2 Empty device name logs a warning and falls back to system default, never silently. ✅ tested
- AC1.3 Pre-roll ring buffer retains ≥512ms so a command run into the wake word is not clipped. ⬜ needs hardware

### C2 Wake word — **built, OUT OF SCOPE for v1** (D2)

Code stays in the tree behind `wakeword.enabled = false`. Not on the v1 critical path, not validated, not a release blocker. Revisit only if PTT proves annoying in practice.

### C3 Push-to-talk on a wheel button — **NOT BUILT** (D2, D9)
- AC3.1 A setup command prompts "press the button you want for push-to-talk", captures the first button-down event, and persists device GUID + button index to the user config. Never a bare index — GUIDs survive re-enumeration where indices don't (same reasoning as D6).
- AC3.2 Holding the button opens capture; **releasing it endpoints immediately, bypassing VAD entirely.**
- AC3.3 Works while iRacing has focus. DirectInput polling, not a keyboard hook — the wheel is a joystick device.
- AC3.4 PTT and wake word coexist; either can fire, neither double-fires while the other is active.
- AC3.5 Wheel disconnected or button unresolvable at startup → loud failure, fall back to wake-word-only, never a silent no-op.
- AC3.6 Setup refuses to bind a button CrewChief already claims (its own PTT is on Keyboard button 91; a wheel button collision would double-trigger).

### C4 Transcription — **built, unverified**
- AC4.1 2s utterance transcribes in ≤300ms on CPU. ⬜ needs hardware
- AC4.2 Model loads at warmup, not on first command. ✅ implemented
- AC4.3 Empty/silent audio yields empty transcript and a logged reject, not a crash. ✅ tested

### C5 Intent matcher, tiers 1–3 — **built**

Tier 2 scores **IDF-weighted query coverage** — how much of what the user *said* is explained by the phrase — not symmetric F1. The rewrite came from a failing case: `car ahead laptime` scored 0.518 and ranked `car_behind_last_lap` as runner-up. Symmetric F1 structurally caps a 3-token query against a 7-token phrase near 0.6, so tier 2 could never fire on terse input.

Coverage is asymmetric in the **opposite** direction from containment, which is what lets it separate two cases that look alike:

- P3 (reject): long utterance, short registered alias tries to claim it → the discriminative tokens go unexplained → low score.
- Terse input (accept): short utterance, long registered phrase → every query token explained → 1.0.

- AC5.1 Exact match ignores case, punctuation, apostrophes, filler words. ✅ tested
- AC5.2 A phrase sharing no token with any intent rejects cleanly. ✅ tested (was a crash; regression test added)
- AC5.3 Top-two within `margin` rejects as ambiguous even at high score. ✅ tested
- AC5.4 A short alias never claims a long utterance that is mostly about something else. ✅ tested
- AC5.5 Every reject records the best candidate and score for tuning. ✅ tested
- AC5.6 Possessives fold onto their stem (`ahead's` → `ahead`) — without this the most discriminative token in the ahead/behind pair never matches. ✅ tested
- AC5.7 Run-together compounds split against the corpus vocabulary (`laptime` → `lap time`), with no hardcoded compound list. ✅ tested
- AC5.8 Terse phrasing resolves at **tier 2**, not by falling through to the embedder. ✅ tested — asserts `method == "token"` so a regression to the slow path fails CI.
- AC5.9 A query of only low-information words is rejected on absolute evidence, not just coverage ratio. ✅ tested against the real corpus (IDF is corpus-relative, so a toy fixture cannot exercise this).

### C6 Haiku 4.5 tier — **NOT BUILT** (D3, D7, D8)
- AC6.1 `tool_choice={"type": "any"}` — an invalid tool name is unrepresentable, not merely unlikely.
- AC6.2 A `no_match` tool exists and is respected; zero intents is a valid answer (preserves the P2 fix).
- AC6.3 Multi-intent fires sinks in utterance order with a configurable inter-key gap. Parallel tool use stays enabled.
- AC6.4 Timeout ≤2.5s with `max_retries=0`; on timeout, network error, or any API failure, fall back to the tier-3 result and log the reason. **Never raises into the pipeline.**
- AC6.5 Tool set is exactly the configured intents plus `no_match` — no open catalogue.
- AC6.6 Testable offline with a stubbed client; **CI must never make a network call or require an API key.**
- AC6.7 The API key is read from `ANTHROPIC_API_KEY` only. Never written to config, never logged, never committed. `doctor` reports presence/absence, never the value.

### C7 Keypress sink — **built, unverified against CrewChief**
- AC7.1 Sends scancode **and** virtual key; CrewChief reads via DirectInput which consumes scancodes. ✅ implemented
- AC7.2 Hold ≥120ms — CrewChief polls at `hold_button_poll_frequency` (100ms), so shorter holds fall between polls. ✅ implemented
- AC7.3 Modifier combos parse and normalise so `shift+ctrl+F13` and `ctrl+shift+F13` collide in the duplicate-key check. ✅ implemented, ⬜ untested
- AC7.4 Two intents on one key is fatal at config load. ✅ tested
- AC7.5 A keypress actually triggers the bound CrewChief action. ⬜ **needs hardware — the single highest-risk unknown**
- AC7.6 A modifier-combo keypress triggers its bound action. ⬜ needs hardware

### Operational constraints on binding (measured, not assumed)

Three facts about CrewChief that shape how binding has to work. All confirmed
against the live 4.19.4.0 install rather than inferred:

1. **F13–F24 cannot be bound by pressing them.** No keyboard emits them — the
   property that makes them collision-proof. So the binding key is *injected*
   into CrewChief's waiting Assign dialog via the same `SendInput` path used at
   runtime. This is a feature, not a workaround: a successful bind proves the
   runtime sink.
2. **CrewChief greys out Assign while a session is running.** It must be open
   but stopped — the main button reading `Start`, not `Stop`. Bind first, then
   start.
2b. **Assign binds *from a selected device*.** "Keyboard" must be selected in
   the Available controllers list before clicking Assign, or it listens on the
   wrong device and captures nothing. This cost a full debugging cycle: an
   injected key that never appeared looked identical to DirectInput filtering
   injected input, which would have killed the keypress sink entirely and sent
   us to a ViGEm virtual-HID driver or the fork. It was a UI-flow step.
3. **Bindings are not persisted to `user.config`.** A binding made in the UI,
   with CrewChief then closed so it flushed (mtime confirmed to move), still
   left every `*_button_index` at `-1` and every `*_device_type` empty. Where
   they do live is unknown: `current_settings_profile` names a
   `defaultSettings.json` that is not in the sound-pack directory, and the
   install dir is not discoverable via Start Menu, ClickOnce cache, or the
   uninstall registry.

(3) is why binding is not automated by writing CrewChief's config directly, and
why `doctor` cannot report what is bound — only CrewChief's own dialog can.

### C8 Full action coverage — **NOT BUILT** (G5)
- AC8.1 All 27 usable actions appear in the shipped config with phrase sets.
- AC8.2 A `bindings` CLI command prints the action→key sheet for manual entry into CrewChief.
- AC8.3 `doctor` reports which configured intents have no CrewChief binding yet.

### C11 Action metadata registry — **NOT BUILT** (D10)

The join between three things that live in different places: CrewChief's bindable action labels, its SRE phrase config, and the tool descriptions Haiku needs. Only the metadata is hand-authored; phrases come from import.

```toml
[[actions]]
id          = "car_ahead_last_lap"
crewchief   = "What's the car ahead's last lap time"   # Add/Remove Actions label
sre_key     = "WHATS_THE_CAR_AHEADS_LAST_LAP_TIME"     # key in speech_recognition_config.txt
description = "Report the last lap time set by the car directly ahead on track."
key         = "F13"
phrases     = []   # empty = import from sre_key; non-empty = override
```

- AC11.1 Every action carries a `description` written for a tool-calling model — states *when to call it*, not just what it does. Prescriptive descriptions measurably improve triggering.
- AC11.2 `phrases` empty → import from `sre_key` via the subset-dropping parser (drops the greedy short aliases that cause P3). Non-empty → use verbatim.
- AC11.3 An `sre_key` absent from the installed CrewChief config fails at load with the key name — not a silent empty phrase set.
- AC11.4 Actions with no natural `sre_key` (the two lap-time ones have no SRE entry) require hand-written `phrases`; loading with both empty is fatal.
- AC11.5 An `import-phrases` command previews what would be imported per action, so the mapping is verifiable before it ships.
- AC11.6 The `sre_key` → action mapping is asserted in CI against a committed fixture, so a CrewChief update that renames a key fails a test rather than a race.

### C9 Installer — **built, unverified**
- AC9.1 `install.ps1` is idempotent and re-runnable. ⬜ needs a clean machine
- AC9.2 Creates venv, installs deps, pre-downloads all models, writes user config, runs tests, runs doctor.
- AC9.3 Fails loudly with actionable text when Python <3.11 is missing.

### C10 Diagnostics — **built**
- AC10.1 Every utterance logs transcript, intent, score, tier, runner-up, latency. ✅ implemented
- AC10.2 Log-write failure never interrupts the pipeline. ✅ implemented
- AC10.3 `doctor` reports CrewChief bindings and config health. ✅ verified against the real 4.19.4.0 install
- AC10.4 Eval fixture runs offline and deterministically in CI. ✅ 83 tests passing

## 8. Failure modes

| ID | Failure | Mitigation | Artifact |
|---|---|---|---|
| ~~F1~~ | ~~Wake word fires on Discord/sim audio~~ | **Eliminated by D2** — no always-on listening in v1 | — |
| F2 | Whisper mis-transcribes | Classification not transcription; tiers tolerate word errors | Eval fixture |
| F3 | Matcher fires wrong intent | Ambiguity margin + threshold | AC5.3, AC5.4 |
| F4 | LLM emits an invalid tool name | `tool_choice: any` makes it unrepresentable | AC6.1 |
| F5 | **Network down or slow mid-race** | 2.5s timeout, no retries, silent fallback to tier 3 | AC6.4 |
| F6 | **API key leaks into the public repo** | Env var only; gitignored config; never logged | AC6.7 |
| F7 | Keypress silently ignored by CrewChief | Scancode+VK, ≥120ms hold, `doctor` preflight | AC7.5 |
| F8 | Modifier combo collides with a sim or Windows binding | F13–F24 base keys are unreachable by physical keyboards | AC7.3 |
| F9 | Wrong mic after replug | Name resolution, fail loudly | AC1.1 |
| F10 | Model download fails on the rig | HashingEmbedder fallback, checksummed atomic download | Implemented |
| F11 | **Wheel disconnects or re-enumerates; PTT silently dead** | Bind by device GUID not index; loud startup failure; degrade to wake-word-only | AC3.1, AC3.5 |
| F12 | **PTT button collides with a sim or CrewChief binding** | Setup refuses a claimed button | AC3.6 |
| F13 | **CrewChief update renames an SRE key; phrases import empty** | Load fails naming the key; CI asserts the mapping against a fixture | AC11.3, AC11.6 |
| F14 | **Silent VRAM use** — `asr.device` changed, or `onnxruntime-gpu` installed | `doctor` compute-placement check; warning at transcriber construction; GPU hidden before wake-word session creation | Implemented |

## 9. Validation plan (D4)

1. **Control.** Apply the zero-code CrewChief fixes (threshold 0.60, disable alternatives, set the recording device explicitly). Run one session. Record hit rate from CrewChief's console. This is the baseline the project must beat.
2. **Bench.** `run --dry-run` in the garage. Verify wake word, PTT, transcription, routing. Then one real keypress confirming AC7.5, and one modifier combo confirming AC7.6.
3. **Measure tier 4 latency.** Time 20 Haiku calls on this connection before trusting §5.3's estimate.
4. **Race.** One full session. Success: **≥90% of commands route correctly, p95 end-to-end under 2.5s**, zero wrong-command fires.
5. **Iterate.** Feed every miss from `logs/utterances-*.jsonl` into `tests/fixtures/utterances.jsonl` and fix the phrase map. The eval is the regression net.

Hit rate is reported as **cold, first-attempt** — not after repeating yourself. Sample size and conditions stated with the number.

## 10. Open decisions

**None blocking.** All six original open decisions are resolved:

| | Resolution |
|---|---|
| O1 — LLM routing position | → D8 (cascade, tier 4 only on local reject) |
| O2 — LLM runtime | → D7 (Claude Haiku 4.5 API, not Ollama) |
| O3 — PTT key | → D9 (wheel button, captured at setup, bound by device GUID) |
| O4 — Wake word model | **Moot** — D2 drops the wake word from v1 |
| O5 — VAD silence timeout | **Moot** — D2 makes PTT release the endpoint; VAD leaves the critical path |
| O6 — Phrase authorship | → D10 (import from CrewChief SRE config, hand-author descriptions) |

## 11. Delta from what is on disk

83 tests passing, lint clean, pushed to the public repo.

**127 tests passing, lint clean.** Every component is now built.

| Component | State |
|---|---|
| C1 audio capture | Built, tested |
| C2 wake word | Built, **disabled** (out of scope, D2) |
| C3 wheel PTT | Built — `setup-ptt` captures a button, pipeline endpoints on release |
| C4 transcription | Built, tested |
| C5 matcher tiers 1–3 | Built, tested — IDF query coverage |
| C6 Haiku tier | Built, 20 offline tests with a stub client |
| C7 keypress sink | Built, modifier combos supported |
| C8 27-action coverage | Built — 12 imported, 15 hand-written, `bindings` sheet |
| C9 installer | Built |
| C10 diagnostics | Built |
| C11 action metadata | Built — `sre_key` import, `import-phrases` preview |
| Compute-placement guards (F14) | Built |
| `NamedPipeSink` | Built, unsupported under D1 — kept for the phase-2 fork |
| Silero VAD | Built, off the critical path under D2 |

### What is verified, and what is not

**Verified offline** (127 tests, no network, no hardware, no API key): phrase parsing and import, all three local matcher tiers, the evidence gate, config layering and every fatal-misconfiguration path, tool construction, and every tier-4 failure mode via a stub client.

**Verified by hand against the real 4.19.4.0 install:** `doctor`, `bindings`, `import-phrases` (12/12 SRE keys resolve, 0 missing), `match`.

**Not verified — needs the rig:**

| | Why it matters |
|---|---|
| AC7.5 — a keypress actually triggers its CrewChief action | **Highest-risk unknown.** Everything downstream assumes it. |
| AC7.6 — modifier combos trigger | 15 of 27 actions depend on Ctrl/Shift |
| AC3.1–3.5 — wheel button capture and release timing | No wheel in this environment |
| AC4.1 — Whisper ≤300ms on CPU | Estimated, not measured |
| §5.3 tier-4 latency | Estimated at 600ms–1s; measure 20 calls before trusting it |

The first bench session should run in this order: `setup-ptt` → bind keys from `bindings` → `doctor` → `run --dry-run` → **one real keypress to settle AC7.5** → one modifier combo for AC7.6.

Nothing already written needs deleting. The new components attach at existing seams: PTT alongside the wake-word check in the pipeline's IDLE state, and Haiku as tier 4 behind the same `MatchResult` contract the other three tiers already return.
