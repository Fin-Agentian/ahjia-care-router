# AhJia Care Router: Stage I prototype

**Small models keep watch. Big models wake only when it matters.**

AhJia is an offline care box for elders who live alone. It uses the cameras a home already has and an ASUS UGen300 (Hailo-10H, 8 GB, about 2.5 W). Video never leaves the home; family members get event text only.

This repository is the **Stage I laptop prototype** for the ASUS UGen AI League 2026 (Battlefield Lightning · Onsite Track · Everyday AI). It runs on a normal laptop without the UGen300. The care logic is real code. Hailo-10H timing and power are **simulated** from official numbers, and every simulated value is marked. In Stage II we run the same logic on a real UGen300 and measure it.

## What is real and what is simulated

| Part | Stage I prototype | Stage II |
|---|---|---|
| Pose detection | **Real**: YOLOv8 pose (Ultralytics) on the laptop CPU | yolov8s/m_pose on Hailo-10H |
| Six care agents | **Real** (rules) | Rules, GBDT or a small MLP |
| Care Router and safety rules | **Real** | Real, plus a learned GenAI window |
| Speech (Whisper-Base) | Simulated: a timed transcript file | Whisper-Base on Hailo-10H |
| Face ID (SCRFD + ArcFace) | Simulated: identity labels | SCRFD + ArcFace on Hailo-10H |
| Who is speaking (elder or TV) | Simulated: a source label | Speaker ID or camera context |
| NPU residency, timing, power | Simulated from `specs/hailo10h.yaml` | Measured with a USB-C power meter and logs |

Every value in `specs/hailo10h.yaml` is tagged `official`, `estimate` or `assumption`, with its source.

## Quick start

Python 3.10 or newer.

```bash
pip install -e ".[dev]"

# Replay every scenario under the three strategies and write the KPI report
python -m ahjia replay --all --out outputs
#   outputs/report.md, outputs/kpi.csv, outputs/kpi.json, outputs/kpi.png

# Build the Workbench: one self-contained HTML page (no install, no network)
python -m ahjia workbench --out outputs/workbench
#   open outputs/workbench/workbench.html in any browser
```

Run live pose detection on a video (optional, needs the vision extras):

```bash
pip install -e ".[vision]"
python -m ahjia demo --video my_room.mp4 --camera configs/camera_example.yaml \
    --speech configs/example.speech.yaml --out outputs/demo
#   outputs/demo/annotated.mp4 (skeleton + 7-line HUD), decisions.jsonl
```

## The Workbench

A prebuilt copy is in [`docs/workbench.html`](docs/workbench.html): download it and open it in any browser. Rebuild it after code changes with `python -m ahjia workbench --out docs`.

It shows the prototype at work:

- **Live**: a home floor plan with cameras, people and their posture, the 7-line HUD, the alert banner, the GenAI state and every A0–A5 decision on a timeline. "What if…" buttons replay other branches, for example "she says I'm OK and stands up" or "the TV shouts help".
- **Compare**: the five KPIs for S0, S1 and S2, marked simulated.
- **How it works**: the three layers, the six agents and the safety rules.

## How it works

```
Layer 1  see      vision and voice models stay loaded on the Hailo-10H
Layer 2  care     six agents: fall · inactivity · night exit · visitor & caregiver · voice SOS · companion
Layer 3  decide   the Care Router gives out chip time and decides when the chat model (LLM) may wake
```

**One action per event**: A0 log only · A1 voice check-in · A2 wake the chat model · A3 look closer (full frame rate, no model switch) · A4 text the family · A5 call for help.

**Fixed safety rules** (no AI can change them):

1. "Help" or "I can't get up" → call at once (A5). Any voice counts, even the TV.
2. Other alerts → if nobody answers in 30 s, call (A5).
3. Only people clear an alarm: the elder says "I'm OK" and stands for 5 s, or family taps confirm in the app.

**Frame rate per camera**: empty room 1–2 FPS · person present 10 FPS · bathroom door, bedside or hallway at night 15 FPS · look closer = full rate.

**GenAI window**: the chat model is loaded only in low-risk times (daytime, nobody in a high-risk zone). At night (22:00–07:00) only vision runs.

**Three strategies** on the same replay: **S0** fixed (everything always on, the cost ceiling) · **S1** rules (fixed frame rate and GenAI hours) · **S2** Care Router (frame rate from people and risk, safety-first scheduling, GenAI windows). S2 vs S1 shows what the router adds.

**Five KPIs**: average power · cameras per UGen300 · event → response time · safety events delayed by the chat model · high-risk misses (plus false alarms per day).

## Repository layout

```
ahjia/events.py       shared records: Observation, ActionRequest, Decision, HudState
ahjia/agents/         the six care agents
ahjia/guardrails.py   the fixed safety rules
ahjia/router.py       frame rate per camera, GenAI window, actions
ahjia/residency.py    Hailo-10H simulator (models loaded, time, power)
ahjia/strategies.py   S0, S1, S2
ahjia/engine.py       runs layers 2 and 3 on a stream of observations
ahjia/replay.py, kpi.py, report.py   replay scenarios and compute KPIs
ahjia/perception/     YOLOv8 pose, posture, zones, tracking
ahjia/overlay.py, demo.py            annotated video
ahjia/workbench/      the Workbench page
scenarios/            three demo scenarios, a 24-hour synthetic day, 13 scripted test cases
specs/hailo10h.yaml   simulator inputs with sources
DESIGN.md             design, contracts and rules
```

## Tests

```bash
pytest            # unit tests (integration tests that need model weights are skipped)
pytest --cov=ahjia
```

## Honesty notes

- Timing and power come from the simulator, not from hardware. The Workbench and every report say so.
- Scenarios are scripted or synthetic. They are fair to S0 and S1; where S2 does not win, the report says so.
- A TV voice saying "help" is treated as real. Telling a TV from the elder is Stage II work.
- Keyword rules are simple on purpose. "I don't need help" still calls (a false call is safer than a missed one). Symptom words understand a short negation list ("not dizzy", 不暈); it can miss phrases like 我會不會暈倒. After a call, repeated "help" within 60 s is logged only, because the call is already in progress.

## References

Hailo Model Zoo GenAI v5.3.0 / v5.4.0 · HailoRT GenAI documentation · hailo-apps (pose, v2a_demo) · Ultralytics YOLOv8 · Shazeer et al. 2017 (Mixture-of-Experts) · WHO Falls fact sheet. The full list is in the proposal deck, Appendix H.

## License

To be decided before the public release. Ultralytics YOLOv8 is AGPL-3.0, so the perception part follows AGPL-3.0.
