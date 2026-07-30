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
- **G2** Consume zero wheel buttons.
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
| D2 | Trigger is **wake word and push-to-talk, both configurable**. | Wake word satisfies G2. PTT gives a low-false-positive baseline to measure the wake word against. |
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

| Stage | Budget | Notes |
|---|---|---|
| Wake word / PTT → capture start | ~0ms | Pre-roll ring buffer covers it |
| **VAD endpointing** | **900ms** | Trailing silence timeout. **The single largest lever — larger than the entire model choice.** Tunable. |
| Whisper tiny.en (CPU int8) | ~150ms | For a 2s utterance |
| Tiers 1–3 (exact / token / embedding) | ~5ms | Local, always runs |
| Tier 4 (Haiku 4.5) | ~600ms–1s *estimated* | Only on cascade miss. **Unverified — must be measured on this connection.** |
| CrewChief response | ~1s | Outside our control |
| **Total, happy path** | **~2.1s** | Tiers 1–3 hit |
| **Total, cascade to Haiku** | **~2.7–3.1s** | |

For comparison, CrewChief's native SRE path is roughly 1.3s. The happy path is within ~800ms of that and buys a real reject; the cascade path is the price of paraphrase freedom, paid only when the local tiers fail.

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

### C2 Wake word — **built, unverified**
- AC2.1 Fires on the configured word; cooldown prevents one utterance retriggering. ⬜ needs hardware
- AC2.2 **≤1 false positive per hour** with Discord voice and sim audio active. ⬜ needs hardware
- AC2.3 Custom-trained models live in `wakeword_custom/`, gitignored, never required (G4).

### C3 Push-to-talk — **NOT BUILT** (D2)
- AC3.1 A configurable key held down opens capture; release endpoints immediately, bypassing VAD.
- AC3.2 Works while an iRacing window has focus — needs a low-level keyboard hook, not polling.
- AC3.3 PTT and wake word coexist; either can fire, neither double-fires.

### C4 Transcription — **built, unverified**
- AC4.1 2s utterance transcribes in ≤300ms on CPU. ⬜ needs hardware
- AC4.2 Model loads at warmup, not on first command. ✅ implemented
- AC4.3 Empty/silent audio yields empty transcript and a logged reject, not a crash. ✅ tested

### C5 Intent matcher, tiers 1–3 — **built**
- AC5.1 Exact match ignores case, punctuation, apostrophes, filler words. ✅ tested
- AC5.2 A phrase sharing no token with any intent rejects cleanly. ✅ tested (was a crash; regression test added)
- AC5.3 Top-two within `margin` rejects as ambiguous even at high score. ✅ tested
- AC5.4 Containment never scores 1.0 for a short phrase inside a long one. ✅ tested
- AC5.5 Every reject records the best candidate and score for tuning. ✅ tested

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

### C8 Full action coverage — **NOT BUILT** (G5)
- AC8.1 All 27 usable actions appear in `config.default.toml` with phrase sets.
- AC8.2 A `bindings` CLI command prints the action→key sheet for manual entry into CrewChief.
- AC8.3 `doctor` reports which configured intents have no CrewChief binding yet.

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
| F1 | Wake word fires on Discord/sim audio | VAD gate, cooldown, tier rejection | AC2.2 measurement |
| F2 | Whisper mis-transcribes | Classification not transcription; tiers tolerate word errors | Eval fixture |
| F3 | Matcher fires wrong intent | Ambiguity margin + threshold | AC5.3, AC5.4 |
| F4 | LLM emits an invalid tool name | `tool_choice: any` makes it unrepresentable | AC6.1 |
| F5 | **Network down or slow mid-race** | 2.5s timeout, no retries, silent fallback to tier 3 | AC6.4 |
| F6 | **API key leaks into the public repo** | Env var only; gitignored config; never logged | AC6.7 |
| F7 | Keypress silently ignored by CrewChief | Scancode+VK, ≥120ms hold, `doctor` preflight | AC7.5 |
| F8 | Modifier combo collides with a sim or Windows binding | F13–F24 base keys are unreachable by physical keyboards | AC7.3 |
| F9 | Wrong mic after replug | Name resolution, fail loudly | AC1.1 |
| F10 | Model download fails on the rig | HashingEmbedder fallback, checksummed atomic download | Implemented |

## 9. Validation plan (D4)

1. **Control.** Apply the zero-code CrewChief fixes (threshold 0.60, disable alternatives, set the recording device explicitly). Run one session. Record hit rate from CrewChief's console. This is the baseline the project must beat.
2. **Bench.** `run --dry-run` in the garage. Verify wake word, PTT, transcription, routing. Then one real keypress confirming AC7.5, and one modifier combo confirming AC7.6.
3. **Measure tier 4 latency.** Time 20 Haiku calls on this connection before trusting §5.3's estimate.
4. **Race.** One full session. Success: **≥90% of commands route correctly, p95 end-to-end under 2.5s**, zero wrong-command fires.
5. **Iterate.** Feed every miss from `logs/utterances-*.jsonl` into `tests/fixtures/utterances.jsonl` and fix the phrase map. The eval is the regression net.

Hit rate is reported as **cold, first-attempt** — not after repeating yourself. Sample size and conditions stated with the number.

## 10. Open decisions

- **O3 — PTT key.** Which key, and is it reachable by touch in a headset? (Note: CrewChief's own PTT is already on Keyboard button 91 — avoid collision.)
- **O4 — Wake word.** Ship a pretrained model (portable, G4) or train on your own voice (more accurate, not portable)?
- **O5 — VAD silence timeout.** 900ms is the default and dominates the latency budget. 600ms is worth trying but risks clipping slow speech. Needs a measured call, not a guess.
- **O6 — Phrase authorship for 27 actions.** Hand-write phrase sets, or import and prune CrewChief's own `speech_recognition_config.txt` through the subset-dropping parser already built?

*Resolved since the last revision: O1 → D8 (cascade). O2 → D7 (Haiku 4.5 API, not Ollama).*

## 11. Delta from what is on disk

83 tests passing, lint clean, pushed to the public repo.

| Component | State | Action |
|---|---|---|
| C1, C4, C5, C7, C10 | Built, tested | Keep |
| C9 installer | Built | Verify on a clean machine |
| C7 modifier support | Built | Add unit tests (AC7.3) |
| C3 push-to-talk | Not built | **Build** (D2) |
| C6 Haiku tier | Not built | **Build** (D3/D7/D8) |
| C8 full action coverage | 12 of 27 intents | **Expand**, add `bindings` command |
| `NamedPipeSink` | Built | Unsupported under D1; keep for the phase-2 fork |
| `config.default.toml` | 12 intents | Add PTT, LLM, and the remaining 15 actions |

Nothing already written needs deleting. The new components attach at existing seams: PTT alongside the wake-word check in the pipeline's IDLE state, and Haiku as tier 4 behind the same `MatchResult` contract the other three tiers already return.
