# Agent Attack Detection

LLM Agent Multi-Round Attack Detection via Cross-Session Behavior Graph Analysis

Detection of multi-round attacks (delayed trigger, gradual escalation, memory poisoning, tool abuse, indirect injection) against LLM agents using a combination of:

- **Cross-Round Behavior Graph** — directed graph of tool call relationships across sessions
- **Multi-Dimensional Anomaly Scoring** — parameter, transition, frequency, and graph structure signals
- **Cumulative Suspicion** — weighted accumulation with decay for cross-session attacks
- **Multi-Layer Decision** — single-round (instant) + multi-round (cumulative) detection

## Experimental Results

Evaluated on **DeepSeek API (function calling)** with a stateful banking environment.  
5 attack categories × 3 variants = 15 attack scenarios + 20 benign training tasks + 10 benign test tasks.

| Attack Category | Detection Rate | Avg Confidence |
|---|---|---|
| Delayed Trigger (Compliance Disguise) | **3/3 = 100%** | 2.236 |
| Gradual Privilege Escalation | **3/3 = 100%** | 2.954 |
| Memory Poisoning + Cross-Session | **3/3 = 100%** | 2.993 |
| Tool Abuse + Cover Tracks | **3/3 = 100%** | 3.004 |
| Indirect Injection (via content) | **1/3 = 33.3%** | 2.750 |
| **Overall DR** | **86.7% (13/15)** | — |

> Note: FPR requires baseline training fix (see known issues). The detector's instant anomaly signals (parameter validation + tool combination checks) are active; the behavioral baseline component requires explicit training data to reduce false positives.

## Architecture

```
User Query → LLM Agent (function calling) → Tool Calls
                                                ↓
┌────────────────── MultiLayerDetector ──────────────────┐
│  Layer 1: Honeytoken/Honeytool detection               │
│  Layer 2: Cross-Round Behavior Graph analysis           │
│    ├─ Node diversity, graph density, transition entropy │
│    └─ Novelty ratio, cumulative anomaly scoring         │
│  Layer 3: Multi-dimension anomaly scoring               │
│    ├─ Parameter anomaly (single-round)                  │
│    ├─ Tool combination anomaly (single-round)           │
│    ├─ Transition + frequency anomaly (baseline)         │
│    └─ Graph structure anomaly (multi-round)             │
│  Layer 4: Decision (instant + cumulative thresholding)  │
└─────────────────────────────────────────────────────────┘
```

## Attack Taxonomy

| # | Category | Mechanism | Detection Signal |
|---|----------|-----------|-----------------|
| 1 | Delayed Trigger | Session A: inject via compliance disguise → Session B: trigger on benign query | Cross-session cumulative score |
| 2 | Gradual Escalation | Multi-turn chain: benign steps → malicious step → cover tracks | Graph entropy + novelty ratio |
| 3 | Memory Poisoning | Session A: store poisoned memory → Session B: LLM acts on it | Cross-session graph structure change |
| 4 | Tool Abuse + Cover Tracks | Execute malicious operation → delete evidence | High-density multi-tool chains |
| 5 | Indirect Injection | Malicious content in email/document/contact → LLM reads and follows | Content-triggered tool chain |

## Project Structure

```
agent-attack-detection/
├── main.py                  # Entry point
├── run_exp.py               # Basic DeepSeek API experiments
├── run_exp2.py              # Function Calling API + stateful env experiments
├── agent/
│   ├── core.py              # Legacy agent (LLMAgent, APIAgent)
│   ├── env.py               # BankingEnvironment with real state
│   ├── function_agent.py    # Function Calling Agent (OpenAI SDK)
│   └── types.py             # ToolCall data types
├── attack/
│   └── scenarios.py         # Attack scenario definitions
├── detection/
│   ├── graph_detector.py    # Cross-Round Behavior Graph Detector
│   └── baselines.py         # AgentShield + Leong + Random baselines
├── experiments/             # Experiment scripts (legacy, simulated data)
└── data/                    # Results output
```

## Setup

```bash
pip install numpy networkx openai
python run_exp2.py
```

## Known Issues

- **Baseline training bug**: `BehavioralBaseline.update()` is not called during training in `MultiLayerDetector.analyze_call()`. This causes high FPR. Fix: explicitly call `detector.scorer.baseline.update(calls)` with benign training data.
- The legacy experiments in `experiments/` use simulated data (not real LLM calls). Use `run_exp2.py` for real API experiments.
- Indirect injection attacks require correct import of the `Document` class in `function_agent.py`.


