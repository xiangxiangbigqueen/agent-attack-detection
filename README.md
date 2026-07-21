# Agent Attack Detection — Multi-Round Attack Detection via Cross-Session Behavior Graph Analysis

**DSC 2026 (The 11th International Conference on Data Science in Cyberspace)**

Detecting multi-round and delayed-trigger attacks on LLM-based AI agents through cross-session tool-call behavior graph analysis.

## Project Structure

```
agent_attack_detection/
├── agent/
│   ├── core.py            # Agent system (API-based + local model)
│   └── __init__.py
├── attack/
│   ├── scenarios.py       # Attack scenarios (delayed trigger, multi-round, etc.)
│   └── __init__.py
├── detection/
│   ├── graph_detector.py  # ⭐ Core innovation: behavior graph-based detection
│   │   ├── BehavioralBaseline     — Learn normal tool-call patterns
│   │   ├── BehaviorGraph           — Cross-round/session call graph
│   │   ├── GraphAnomalyScorer      — Cumulative anomaly scoring
│   │   └── MultiLayerDetector      — Unified detection interface
│   └── __init__.py
├── experiments/
│   ├── runner.py          # Experiment orchestrator
│   └── __init__.py
├── data/                  # Experiment results (JSON)
├── figures/               # Generated figures
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulated experiments
python main.py --quick

# Run with API (requires DeepSeek/OpenAI key)
python main.py
```

## Method Overview

Our detector builds a **cross-session behavior graph** from agent tool-call sequences and uses **cumulative anomaly scoring** to detect attacks that span multiple rounds or sessions.

Key innovation: Instead of checking individual tool calls (which look benign), we track the *pattern* of tool calls over time and flag sequences that deviate from the learned behavioral baseline.

## Citation

```
@inproceedings{agentdetect2026,
  title={Multi-Round Attack Detection for LLM Agents via Cross-Session Behavior Graph Analysis},
  author={...},
  booktitle={DSC 2026},
  year={2026}
}
```
