# AhJia Care Router: Stage I prototype design

## Goal

Show on a normal laptop that the core of AhJia works end to end:

camera video → pose → six care agents → Care Router and safety rules → actions

Then compare three strategies (S0, S1, S2) on the same scenarios.

Stage I runs without the UGen300. Hailo-10H timing and power are **simulated** from official figures, and every simulated number is marked. Stage II replaces the simulator with measurements on a real UGen300.

The source of truth is the proposal v4 (`../提案計畫_居家長照陪伴盒.md`, called **P** below). Line numbers like P:243 point into it. If this file and P disagree, P wins; tell the leader.

## Real vs simulated

| Part | Stage I prototype | Stage II |
|---|---|---|
| Pose | **Real**: YOLOv8 pose (Ultralytics) on the laptop CPU | yolov8s/m_pose HEF on Hailo-10H |
| Speech (Whisper-Base) | Simulated: a timed transcript file | Whisper-Base on Hailo-10H |
| Face ID (SCRFD + ArcFace) | Simulated: identity labels in the config or scenario | SCRFD + ArcFace on Hailo-10H |
| Who is speaking (elder or TV) | Simulated: a source label in the transcript. Any voice can raise an alert; only the elder's speech can stop the 30 s rule or clear an alert (with standing ≥ 5 s) | Speaker ID or camera context |
| Six care agents | **Real** (rules) | Rules, GBDT or a small MLP |
| Care Router and guardrails | **Real** | Real, plus a learned GenAI window |
| NPU residency, power, latency | Simulated from `specs/hailo10h.yaml` | Measured with a USB-C power meter and logs |
| Companion LLM | Simulated: fixed reply text, timing from the spec file | Qwen2.5-1.5B on Hailo-10H or the host CPU |

## Architecture (same as P:112-138)

```
Layer 1  perception    video -> pose -> posture + zone ; transcript -> speech      (ahjia/perception)
            |  Observation
Layer 2  care agents   fall | inactivity | night_exit | visitor | voice_sos | companion   (ahjia/agents)
            |  ActionRequest
Layer 3  Care Router   frame rate per camera, GenAI window, actions A0-A5             (ahjia/router.py)
         guardrails    alert / escalate / clear rules, deterministic                  (ahjia/guardrails.py)
            |  Decision  -> decision log (JSONL), HUD overlay, KPIs
NPU simulator          which models are loaded, time and power per state              (ahjia/residency.py)
```

## Files and owners

Workers only edit files they own. Shared files belong to the leader; ask before changing them.

| Owner | Files |
|---|---|
| Leader | `DESIGN.md`, `ahjia/events.py`, `ahjia/cli.py`, `ahjia/__init__.py`, `ahjia/__main__.py`, `pyproject.toml`, `requirements*.txt`, `.gitignore`, `tests/test_contracts.py` |
| Core | `ahjia/agents/`, `ahjia/guardrails.py`, `ahjia/router.py`, `ahjia/residency.py`, `ahjia/strategies.py`, `ahjia/engine.py`, `ahjia/replay.py`, `ahjia/kpi.py`, `ahjia/report.py`, `scenarios/`, `specs/`, matching tests |
| Perception | `ahjia/perception/`, `ahjia/overlay.py`, `ahjia/overlay_text.py`, `ahjia/demo.py`, `configs/`, matching tests |
| Workbench | `ahjia/workbench/`, `tests/test_workbench.py` |

## Contracts

### Records (`ahjia/events.py`)

`Observation` (layer 1) → `ActionRequest` (layer 2) → `Decision` and `HudState` (layer 3). All are frozen dataclasses. Read the docstrings in `events.py`.

### Engine (implemented by Core in `ahjia/engine.py`)

```python
def make_engine(strategy: StrategyName, config: RunConfig) -> Engine: ...

class Engine(Protocol):           # also declared in events.py
    strategy: StrategyName
    def observe(self, obs: Observation) -> tuple[Decision, ...]: ...  # moves time to obs.t first
    def advance(self, t: float) -> tuple[Decision, ...]: ...         # timeouts: 30 s rule, 60 s rule, windows
    def hud(self) -> HudState: ...
    def camera_fps(self, camera: str) -> float: ...                  # current frame rate the router wants
```

`RunConfig` (defined in `events.py`) holds the start clock, camera names, whether the elder is alone, and the spec file path.

### Scenario file (`scenarios/*.yaml`, used by `replay`)

```yaml
name: night_fall
title: "Scenario 3: night fall (GenAI off, vision only)"
source: "P:511-522"
start_clock: "02:10:00"
cameras: [living, hallway, bathroom_door, front_door]
elder_alone: true
duration_s: 60
labels:                       # ground truth, used by KPI 3 and KPI 5
  - {t: 12, kind: high_risk, what: fall}
  - {t: 19, kind: sos_immediate, what: "I can't get up"}
events:                       # true state of the world; it stays the same until the next change
  - {t: 0,  camera: hallway, persons: [{track_id: 1, identity: elder, posture: stand, zone: hallway}]}
  - {t: 12, camera: hallway, persons: [{track_id: 1, identity: elder, posture: lie, zone: floor}]}
  - {t: 19, speech: {text: "I can't get up", source: elder, directed: false}}
```

The replay samples this true state **at the frame rate each strategy picks for each camera**. A lower frame rate therefore means later detection and lower power, so the frame-rate choice shows up in the KPIs.

### Camera config (`configs/*.yaml`, used by `demo`)

```yaml
camera: living
start_clock: "14:05:00"
elder_alone: true
identity: elder               # face ID is simulated: the person in view is this identity
zones:                        # normalized polygons, x and y in 0..1
  sofa: [[0.05, 0.55], [0.35, 0.55], [0.35, 0.95], [0.05, 0.95]]
  bed: [[0.60, 0.50], [0.95, 0.50], [0.95, 0.90], [0.60, 0.90]]
default_zone: floor           # ground outside every polygon
```

The zone of a person is the zone that contains the foot point (middle of the ankles, or the bottom center of the box).

### Speech file (`*.speech.yaml`, used by `demo`)

```yaml
- {t: 21.5, text: "I can't get up", source: elder, directed: false}
- {t: 41.0, text: "I'm OK", source: elder, directed: true}
```

### Decision log (JSONL)

One `Decision` per line, written with `events.decision_to_json`.

### CLI (`python -m ahjia ...`)

- `replay [--scenario NAME | --all] [--strategies S0,S1,S2] [--out outputs]` writes `outputs/replay/<scenario>/<strategy>.jsonl`, `outputs/report.md`, `outputs/kpi.csv` and `outputs/kpi.png`.
- `demo --video PATH --camera CFG [--speech FILE] [--out outputs/demo] [--fps 10] [--model yolov8n-pose.pt]` writes an annotated mp4 and `decisions.jsonl`.

### Workbench (`python -m ahjia workbench [--out outputs/workbench]`)

Replays every scenario under S0, S1 and S2 and writes **one self-contained `workbench.html`** (inline CSS, JS and data; no network, no CDN). It opens in any browser, so judges can use it without installing anything.

- **Live**: home floor plan with zones and cameras (frame rate per camera), people with their posture, speech bubbles, the 7-line HUD, the alert banner, the GenAI state and a decision timeline with A0–A5 chips and response times. "What if…" branches come from the scripted scenarios, for example "she says I'm OK and stands up" or "family confirms in the app".
- **Compare**: KPI table and charts for S0, S1 and S2, marked simulated, with the spec file's official / estimate / assumption counts.
- **How it works**: three layers, six agents, A0–A5 and the safety rules, in the deck's words.
- **Recording**: URL parameters `?scenario=&strategy=&t=&tab=&autoplay=` and `window.ahjiaSeek(t)` give deterministic frames for the video.

## Rules (copy these exactly from P)

### Six care agents (P:140-149)

| Agent | Trigger | Actions |
|---|---|---|
| fall | posture goes from upright to the ground, then lying outside the bed or sofa zone ≥ 2 s | A1 + A4 + A3 |
| inactivity | in the bathroom ≥ 30 min; no movement ≥ 2 h in the daytime; not up 60 min after the usual wake time | A1, then A4 if no answer in 60 s |
| night_exit | at the front door between 00:00 and 06:00 | A1 + A4 |
| visitor | identity is family, caregiver or stranger; log caregiver arrival and leaving times | stranger at night → A4 |
| voice_sos | keywords: 救命, 好暈, 胸口痛, 起不來, "help", "dizzy", "chest pain", "can't get up" | A1 + A4; 救命, 起不來, "help", "can't get up" → A5 at once |
| companion | family comes in, the elder talks to AhJia, reminder times, daily summary | A2, only inside a GenAI window or on the host CPU; **never makes a safety decision** |

Long thresholds (30 min, 2 h) can be shortened in demo scenarios with a `time_scale` field. Keep the real values as defaults.

### Actions (P:180-191)

A0 no-op (log only) · A1 template voice · A2 wake the LLM (companion) · A3 look closer (full frame rate and full-resolution pose, no model switch) · A4 pre-alert family (event text only) · A5 escalate (call family or the emergency contact).

### Guardrails (P:243-251), deterministic

- Trigger: suspected fall (lying outside the bed or sofa ≥ 2 s), an SOS keyword, or night exit. Inactivity is not an instant trigger: it goes A1, then A4 after 60 s with no answer.
- After a trigger: A1 + A4 at once (targets: voice ≤ 2 s, family text ≤ 5 s), and A3.
- Escalate to A5 at once only for 救命, 起不來, "help" or "can't get up". Otherwise escalate when there is no answer for 30 s. Other SOS words (好暈, 胸口痛, "dizzy", "chest pain") give A1 + A4, then follow the 30 s rule.
- Only two ways to clear an alert:
  - (a) the elder says 我沒事 or "I'm OK" **and** stays upright ≥ 5 s; family gets "cleared by the elder";
  - (b) family or a caregiver confirms in the app (`Observation.app_confirm`).
- No AI can clear or lower an alert. An alert never waits for a generative model.
- During an alert:
  - 救命, 起不來, "help" or "can't get up" again gives A1 + A4 + A5 at once.
  - A new trigger from another agent, or any trigger while the 30 s timer is stopped, re-issues A1 (+A3) + A4 and re-arms the 30 s timer.
  - If the timer is still running, the earlier deadline stays, so a second trigger never delays A5.
  - A repeat from the same agent while its timer runs is only logged.
- Speech that contains a symptom word is never an answer, and never counts as "I'm OK". A symptom word right after a simple negation does not count ("not dizzy", "no chest pain", 不暈, 沒有胸口痛). This negation list is a small heuristic; Stage II replaces it with better language understanding. Immediate words ignore negation on purpose, so "I don't need help" still calls: a false call is safer than a missed one.
- Known limits of the negation heuristic: it can miss a symptom in phrases like "oh no dizzy" (no comma), 我會不會暈倒 or 沒有不暈. During an alert such speech counts as an answer; the family text (A4) has already gone out.
- During the 60 s after a call, a repeated immediate word is only logged: the call is already in progress, so there is no new voice line or family text.
- After a call (A5), a repeat call is held back for 60 s. While escalated, a new trigger always re-issues A1 (+A3) + A4. If the last call was 60 s or more ago, it also arms a new 30 s timer.
- The elder's self-clear uses only the freshest view of her. A view goes stale after max(2 s, 3 frames at that camera's frame rate). While an alert is on, the engine asks that camera for frames so the view stays fresh.
- An app confirm also cancels a pending inactivity check-in. A stranger-at-night notice (A4 only) does not change any frame rate.

### Frame rate per camera (P:216-222)

Empty room 1–2 FPS (motion gate + person detection only); person present 10 FPS; person present in a high-risk zone or time (bathroom door, bedside, hallway at night) 15 FPS; A3: the camera's full rate with full-resolution pose. When the NPU is short, lower empty and low-risk cameras first and the companion last. High-risk cameras always come first.

### GenAI window (P:224-241)

- Load the LLM on the NPU only in a low-risk window: daytime, nobody in a high-risk zone, or someone is talking to AhJia.
- Night (22:00–07:00): vision only by default; night talk uses template voice or the LLM on the host CPU.
- Preload rule: open the window if `p_L × S_L × w_L > ρ × C_risk` (P:233-238). `S_L` ≈ 5.2 s (official load time).

### Strategies (P:423-431)

| Strategy | Role | Definition |
|---|---|---|
| S0 fixed | cost ceiling | every camera at full rate; every perception model on every frame; LLM always loaded and answers every speech segment |
| S1 rules | real baseline | fixed frame rate; rule-based agents; fixed GenAI hours |
| S2 Care Router | our method | frame rate from occupancy and risk; zone cascade; safety-first QoS; GenAI window from the preload rule |

### KPIs (P:433-443)

1. Average power over a 24 h replay = Σ (time share of each state × power of that state).
2. Cameras per UGen300 while KPI 3 still passes (estimated from NPU time per frame).
3. Event → response time, p50 and p95 (voice ≤ 2 s, family text ≤ 5 s, A5 at once for 救命 / 起不來, otherwise after 30 s; talk p50 ≤ 3 s; greeting ≤ 4 s).
4. GenAI interference: safety events delayed by LLM loading or running (target 0); LLM wake rate; reply rate when someone talks to AhJia (≥ 95 %).
5. High-risk misses: labelled high-risk events with no pre-alert (target 0 on the scripted set).

## Honesty rules

- No KPI number is typed by hand. Every number comes from the simulation or from `specs/hailo10h.yaml`.
- Every value in the spec file has `kind: official | estimate | assumption` and a `source`.
- Scenarios must be fair to S0 and S1. Do not tune a scenario so that S2 wins; if S2 does not win on a KPI, report it.
- Output files and the HUD say "simulated" wherever a timing or power value comes from the simulator.

## Acceptance

- `pytest` passes; coverage of `ahjia/` ≥ 80 % (tests that need model weights or video are marked `integration` and skipped by default).
- `python -m ahjia replay --all` writes the KPI table for S0 / S1 / S2 on the three demo scenarios (P:484-522) and on a 24 h synthetic day.
- `python -m ahjia demo ...` writes an annotated mp4 with the skeleton and the 7-line HUD (P:470-482), plus `decisions.jsonl`.
