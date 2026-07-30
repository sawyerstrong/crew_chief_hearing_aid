# SPEC: crew_chief_hearing_aid v1

Status: **draft, awaiting approval**
Target: CrewChief V4 4.19.4.0, iRacing, Windows 11, VR

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

P2 is the one that motivates the project. P1 and P3 have zero-code mitigations inside CrewChief (lower `minimum_voice_recognition_confidence_system_sre` to 0.60; set `disable_alternative_voice_commands = True`) and should be applied regardless as a control.

## 2. Goals

- **G1** Voice control of CrewChief that is more accurate than its native SRE, measured, not asserted.
- **G2** Consume zero wheel buttons.
- **G3** Require no modification to CrewChief; survive its auto-updates.
- **G4** Portable: clone from GitHub, run on a different machine after config.
- **G5** Produce a diagnostic record that makes misses debuggable — the thing CrewChief's SRE structurally cannot give you, since it only ever reports its own grammar match and never what you actually said.

## 3. Non-goals

- **N1** Reading iRacing telemetry directly. This project routes to CrewChief; it does not compute answers.
- **N2** Cloud inference. The mic and the keypress target are both local; cloud adds a network round trip on top of a local agent you still need, on the connection carrying iRacing netcode.
- **N3** Replacing CrewChief's spotter, or any always-on messaging.
- **N4** Distribution to other users. Single-operator tool; GPL obligations only bite on distribution.

## 4. Resolved decisions

| ID | Decision | Rationale |
|---|---|---|
| D1 | Output is **synthetic keypress only**. No CrewChief fork in v1. | Keeps the auto-updater and avoids owning a C# fork across CrewChief's release cadence (4.19.1 → 4.19.4 already observed on this machine). |
| D2 | Trigger is **wake word and push-to-talk, both configurable**. | Wake word satisfies G2. PTT gives a low-false-positive baseline to measure the wake word against. |
| D3 | Intent routing includes an **LLM tool-calling stage**, not embeddings alone. | Multi-intent handling, and the project is explicitly partly for its own sake. |
| D4 | Done means **works in a live race**, measured from the utterance log. | Bench testing does not exercise cockpit noise, VR load, or Discord bleed. |
| D5 | Keys are **F13–F24**. | No physical keyboard emits them, so no collision with sim or Windows bindings is possible. |
| D6 | Audio devices resolved by **name substring, never index**. | Four active mics plus NVIDIA Broadcast virtual endpoints on this rig; indices reshuffle across reboots and replugs. |

## 5. Scope consequences that need acknowledging

### 5.1 Gap ahead/behind is not achievable in v1

The original ask named "lap time ahead and behind." Checking CrewChief's Add/Remove Actions against that:

- ✅ `What's the car ahead's last lap time` — bindable
- ✅ `What's the car behind's last lap time` — bindable
- ❌ **gap** ahead / **gap** behind — voice-only, not in the bindable action set

D1 therefore delivers lap times ahead and behind but **not gaps**. Gaps require the named-pipe fork, which D1 puts out of scope. `NamedPipeSink` is already written behind the `Sink` interface and stays in the tree as unsupported code, so lifting this later is a config change plus the C# work — not a rewrite.

**If gap queries are actually the point, D1 is the wrong call and should be revisited before implementation starts.**

### 5.2 D1 and D3 interact: argument extraction is dead

Tool-calling earns its cost through two things. Under keypress output only one survives:

| LLM capability | Survives D1? | Why |
|---|---|---|
| Multi-intent ("fuel and lap time ahead?") | ✅ | Fire two keypresses in sequence |
| Paraphrase robustness | ✅ | Better than cosine on unusual phrasings |
| **Argument extraction** ("what's **P4's** last lap") | ❌ | A keypress carries no payload. There is no channel for the argument. |

So the LLM in v1 buys multi-intent sequencing and paraphrase headroom. That is real, but it is roughly half of what tool-calling is normally worth. Argument extraction returns only with the fork.

### 5.3 D3 and D4 interact: latency

D4 is a live race. The LLM stage costs 400ms–1.5s depending on model and whether it gets VRAM it is competing with a VR sim for. Budget:

| Stage | Budget | Notes |
|---|---|---|
| Wake word → capture start | ~0ms | Pre-roll ring buffer covers it |
| VAD endpointing | 900ms | Trailing silence timeout; dominates and is tunable |
| Whisper tiny.en (CPU int8) | ~150ms | For a 2s utterance |
| Embedding match | ~5ms | |
| LLM stage | 400–1500ms | The variable |
| CrewChief response | ~1000ms | Outside our control |
| **Total** | **2.5–3.5s** | vs ~1.3s for CrewChief's native SRE path |

**Open decision O1 below.** At 3.5s you ask entering a corner and hear the answer at the exit.

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
                    ┌── tier 1  exact canonical match ──┐  µs
                    ├── tier 2  symmetric token F1 ─────┤  µs
                    ├── tier 3  embedding cosine ───────┤  ~5ms
                    └── tier 4  LLM tool-call ──────────┘  400-1500ms
                                   │
                    best ≥ threshold AND margin over runner-up?
                             yes ──┴── no ──► reject, log, stay silent
                              │
                              ▼
                     Sink.fire(intent) ×N   [keypress]
```

Tier 2 is symmetric (F1 over content-token sets), **not** containment. Containment would score `lap time` at 1.0 against "what's the lap time of the car ahead" — reproducing P3 exactly. F1 scores it 0.5 and declines. This is enforced by test.

## 7. Components and acceptance criteria

### C1 Audio capture — **built**
- AC1.1 Device resolved by name substring; unresolvable name fails at startup with the available list. ✅ tested
- AC1.2 Empty device name logs a warning and falls back to system default, never silently. ✅ tested
- AC1.3 Pre-roll ring buffer retains ≥512ms so a command run into the wake word is not clipped. ⬜ needs hardware

### C2 Wake word — **built, unverified**
- AC2.1 Fires on the configured word; cooldown prevents one utterance retriggering. ⬜ needs hardware
- AC2.2 **≤1 false positive per hour** with Discord voice and sim audio active. ⬜ needs hardware
- AC2.3 Custom-trained models live in `wakeword_custom/`, gitignored, never required (G4).

### C3 Push-to-talk — **NOT BUILT** (new under D2)
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

### C6 LLM tool-call stage — **NOT BUILT** (new under D3)
- AC6.1 Constrained decoding (JSON schema / GBNF). An invalid tool name must be **unrepresentable**, not merely unlikely.
- AC6.2 Emits zero or more intents; zero is a valid, respected answer (preserves P2's fix).
- AC6.3 Multi-intent fires sinks in utterance order with a configurable inter-key gap.
- AC6.4 Tool set capped at the intents in config (12), not an open catalogue — small models degrade past ~20 tools.
- AC6.5 A stage timeout falls back to the tier-3 result rather than hanging.
- AC6.6 Testable offline with a stubbed client; CI must not require a model.

### C7 Keypress sink — **built, unverified against CrewChief**
- AC7.1 Sends scancode **and** virtual key; CrewChief reads via DirectInput which consumes scancodes. ✅ implemented
- AC7.2 Hold ≥120ms — CrewChief polls at `hold_button_poll_frequency` (100ms), so shorter holds fall between polls. ✅ implemented
- AC7.3 Two intents on one key is fatal at config load. ✅ tested
- AC7.4 A keypress actually triggers the bound CrewChief action. ⬜ **needs hardware — the single highest-risk unknown**

### C8 Diagnostics — **built**
- AC8.1 Every utterance logs transcript, intent, score, tier, runner-up, latency. ✅ implemented
- AC8.2 Log-write failure never interrupts the pipeline. ✅ implemented
- AC8.3 `doctor` reports CrewChief bindings and config health. ✅ verified against the real 4.19.4.0 install
- AC8.4 Eval fixture runs offline and deterministically in CI. ✅ 83 tests passing

## 8. Failure modes

| ID | Failure | Mitigation | Artifact |
|---|---|---|---|
| F1 | Wake word fires on Discord/sim audio | VAD gate, cooldown, tier rejection | AC2.2 measurement |
| F2 | Whisper mis-transcribes | Classification not transcription; tiers tolerate word errors | Eval fixture |
| F3 | Matcher fires wrong intent | Ambiguity margin + threshold | AC5.3, AC5.4 |
| F4 | **LLM hallucinates a tool name** | Constrained decoding makes it unrepresentable | AC6.1 |
| F5 | **LLM latency blows the budget** | Stage timeout → tier-3 fallback | AC6.5, O1 |
| F6 | Keypress silently ignored by CrewChief | Scancode+VK, ≥120ms hold, `doctor` preflight | AC7.4 |
| F7 | Wrong mic after replug | Name resolution, fail loudly | AC1.1 |
| F8 | Model download fails on the rig | HashingEmbedder fallback, checksummed atomic download | Implemented |

## 9. Validation plan (D4)

1. **Control.** Apply the zero-code CrewChief fixes (threshold 0.60, disable alternatives, set the recording device explicitly). Run one session. Record hit rate from CrewChief's console. This is the baseline the project must beat.
2. **Bench.** `crew_chief_hearing_aid run --dry-run` in the garage. Verify wake word, PTT, transcription, routing. Then one real keypress confirming AC7.4.
3. **Race.** One full session. Success: **≥90% of commands route correctly, p95 end-to-end under 2.5s**, zero wrong-command fires.
4. **Iterate.** Feed every miss from `logs/utterances-*.jsonl` into `tests/fixtures/utterances.jsonl` and fix the phrase map. The eval is the regression net.

Hit rate is reported as **cold, first-attempt** — not after repeating yourself. Sample size and conditions stated with the number.

## 10. Open decisions

- **O1 — LLM routing position.** Cascade (LLM only when tiers 1–3 reject, keeping the happy path at ~5ms) or always-on (every utterance pays 400–1500ms)? Cascade is strongly recommended given D4; always-on makes the common case worse to serve the uncommon one. **Needs a ruling before C6 is built.**
- **O2 — LLM runtime.** Ollama is already installed on this machine. Qwen3-0.6B or Llama-3.2-1B, and CPU or GPU? GPU contends with VR.
- **O3 — PTT key.** Which key, and is it reachable by touch in a headset?
- **O4 — Wake word.** Ship a pretrained model (portable, G4) or train on your own voice (more accurate, not portable)?

## 11. Delta from what is already on disk

| Component | State | Action |
|---|---|---|
| C1, C4, C5, C7, C8 | Built, 83 tests passing | Keep |
| C3 push-to-talk | Not built | **Build** (D2) |
| C6 LLM stage | Not built | **Build** (D3), pending O1 |
| `NamedPipeSink` | Built | Demote to unsupported; D1 puts it out of scope |
| `config.default.toml` | 12 intents | Add PTT and LLM sections |

Nothing already written needs deleting. The two new components attach at existing seams: PTT alongside the wake-word check in the pipeline's IDLE state, and the LLM as tier 4 behind the same `MatchResult` contract.
