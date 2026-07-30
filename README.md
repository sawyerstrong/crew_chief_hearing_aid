# crew_chief_hearing_aid

Local voice control for [CrewChief V4](https://github.com/mrbelowski/CrewChiefV4). Wake word → Whisper → semantic intent match → synthetic keypress. No wheel buttons consumed, no changes to CrewChief, no cloud.

## Why this exists

CrewChief's built-in recogniser is a *closed-grammar* SAPI engine: it matches audio against a fixed phrase list and must always return its nearest entry. Three consequences show up in its own logs:

```
15:16:00.945 : System recogniser recognised : "lap time", Confidence = 0.989
15:16:09.392 : System recogniser recognised : "lap time", Confidence = 0.653
15:16:09.392 : Confidence 0.653 is below the minimum threshold of 0.750
```

1. **Identical utterances score anywhere from 0.65 to 0.99.** A fixed threshold sits inside that band, so the same sentence is accepted or rejected at random.
2. **It cannot say "none of the above."** Say something outside the grammar and it returns the closest registered phrase anyway.
3. **Short aliases are greedy.** `WHAT_WAS_MY_LAST_LAP_TIME` registers the bare alias `lap time`, which is a substring of many longer sentences. Say "what's the lap time of the car ahead" and it latches onto `lap time`, scores it 0.989 because you genuinely said those words clearly, and fires the wrong command. High confidence, wrong answer.

Whisper is an *open* transcriber with no grammar to collapse into. Matching then happens on the full sentence via symmetric similarity, which is commensurable across intents — so a low best score is meaningful and a real reject exists.

## Quickstart

```bash
git clone https://github.com/sawyerstrong/crew_chief_hearing_aid
cd crew_chief_hearing_aid
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer creates a virtualenv, installs everything, pre-downloads the models, runs the tests, and puts a `cchear` launcher on your PATH. Then:

```bash
cchear setup
```

Five steps: user config → pick your microphone → capture a wheel button for push-to-talk → bind actions in CrewChief → health check.

### Why binding works the way it does

F13–F24 have no physical keys. That is exactly why nothing else on your system can ever emit them — and exactly why CrewChief's press-the-key-to-bind dialog cannot capture them from your keyboard.

So setup **injects** each key while CrewChief's Assign dialog is waiting, through the same `SendInput` path used at runtime. A successful bind therefore also proves the runtime sink reaches CrewChief. It checks after the first action and stops if the key was not captured — if injected scancodes do not arrive, nothing downstream can work and binding 26 more will not help.

Two things about CrewChief's UI that are easy to miss, and both make binding silently fail:

> **It must be open but *stopped*.** CrewChief greys out **Assign** whenever a session is running — the main button has to read `Start`, not `Stop`. Bind everything first, then start it.

> **Select `Keyboard` in the Available controllers list first.** Assign binds *from a device* and listens on whichever one is selected. With a wheel selected, a keyboard key is captured by nothing at all — which looks exactly like injection being blocked.

Setup offers the **core 12** (plain F13–F24, the race-critical ones) as a first pass; the remaining toggles and rally features can be added later with:

```bash
cchear bind-all
```

> **Note:** CrewChief does not persist bindings to `user.config` — verified on a live 4.19.4.0 install, where a binding made in the UI and flushed on exit still left every `*_button_index` at `-1`. `doctor` therefore cannot report what is bound; confirm in CrewChief's own dialog.

### Running

```bash
cchear doctor          # install, trigger, tier-4, compute placement, sink
cchear run --dry-run   # logs intents, sends no keys
cchear run             # for real
```

## Test matching without a microphone

```bash
crew_chief_hearing_aid match "whats the lap time of the car ahead" "how quick is the bloke in front"
```

```
'whats the lap time of the car ahead'
  -> car_ahead_last_lap (F13) score=0.669 via embedding
'how quick is the bloke in front'
  -> REJECTED (below_threshold) best=0.453 via embedding runner_up=damage_report
```

That second one is a miss — which is the point of the next section.

## The tuning loop

Every utterance is appended to `logs/utterances-YYYYMMDD.jsonl` with the transcript, chosen intent, score, which tier matched, and the runner-up. This is the artifact CrewChief never gave you: it only ever reports its own grammar match, so you cannot tell a misheard sentence from a badly-tuned threshold.

After a session:

1. Read the log for rejects and wrong routes.
2. Add the phrasing to the relevant intent in your config.
3. Add the case to `tests/fixtures/utterances.jsonl`.
4. `pytest` — it is now a regression test.

## How it works

```
mic ──► ring buffer (512ms preroll)
         │
         ├─► openWakeWord ──── fires ──┐
         │                             ▼
         └─────────────────────► accumulate + Silero VAD endpointing
                                       │
                                       ▼
                                 Whisper tiny.en (CPU, int8, ~150ms)
                                       │
                                       ▼
                          ┌──── exact canonical match ─────┐  tier 1, µs
                          ├──── symmetric token F1 ────────┤  tier 2, µs
                          └──── embedding cosine ──────────┘  tier 3, ~1ms
                                       │
                          best ≥ threshold AND margin over runner-up?
                                   yes ─┴─ no ──► reject, log, stay silent
                                    │
                                    ▼
                              Sink.fire(intent)
```

The pre-roll matters because people run the command straight into the wake word; without it the first word is gone before capture starts.

Tier 2 is **symmetric** (F1 over content-token sets), not containment. Containment would score the phrase `lap time` at 1.0 against "what's the lap time of the car ahead" — reproducing exactly the bug this project exists to avoid. F1 scores that pair 0.5 and declines.

## Design notes

**Why no LLM router.** If the tool's implementation is "send the canonical phrase," the model's entire job is string→string mapping over a fixed set — a classifier in a costume, at 10–100× the latency. Tool-calling earns its cost on arguments and multi-intent composition, neither of which the button-bindable action set supports. Revisit at phase 2.

**Why not AWS.** The microphone and the keypress target are both on the rig, so cloud adds a network round trip *on top of* a local agent you still need, on the same connection carrying your iRacing netcode. Local CPU inference is faster and does not fail when the WiFi hiccups. Cloud's legitimate role here is CI and artifact hosting.

**Why CPU.** The GPU is rendering VR at 90fps. Whisper tiny.en int8 on two pinned threads transcribes a 2s utterance in roughly 150ms, well inside budget.

## Phase 2: the fork

Keypress output is capped by what CrewChief exposes in Add/Remove Actions — which notably excludes gap ahead/behind. Lifting that means forking CrewChief and adding a named pipe:

Refactor `sre_SpeechRecognized` so everything after `e.Result.Text` moves into

```csharp
internal void handleRecognisedText(string text, float confidence)
```

then have both the SAPI event and a `NamedPipeServerStream` reader call it. **Do not call `getEventForSpeech` directly** — it is only part of dispatch; macros, iRacing pit commands and the mute toggles live in the other branches.

Then flip `output.sink = "pipe"`. `NamedPipeSink` is already written. The cost is losing the ClickOnce auto-updater, which is the real tax — not the ~100 lines of C#.

## Status

Verified by `pytest` (83 tests, offline, no models):

- phrase normalisation and CrewChief config parsing
- all three matcher tiers, threshold rejection, ambiguity margin
- config layering and duplicate-key detection
- the 43-case eval set against the shipped phrase map

Verified by hand against a real 4.19.4.0 install: `crew_chief_hearing_aid doctor`, `crew_chief_hearing_aid match`.

**Not yet run against hardware:** audio capture, wake word, Silero VAD, Whisper transcription, and the keypress actually reaching CrewChief. Those need the rig.
