# Agent Attack Detection

LLM Agent Multi-Round Attack Detection via Cross-Session Behavior Graph Analysis

Detection of multi-round attacks (delayed trigger, gradual escalation, memory poisoning, tool abuse, indirect injection) against LLM agents using a combination of:

- **Cross-Round Behavior Graph** — directed graph of tool call relationships across sessions
- **Enhanced Graph Features** — motif counts, path analysis, PageRank entropy, category transitions
- **Content-Aware Embedding** — SBERT-based semantic analysis of tool call parameters
- **Adaptive Threshold** — Statistical Process Control (SPC) for dynamic anomaly thresholding
- **GNN Support** — Optional GAT-style graph attention network for learned anomaly detection

## Architecture

```
User Query → LLM Agent (function calling) → Tool Calls
                                                ↓
┌────────────── EnhancedMultiLayerDetector ──────────────┐
│  Layer 1: Honeytoken/Honeytool detection               │
│  Layer 2: Enhanced Behavior Graph analysis              │
│    ├─ Base: diversity, density, entropy, novelty        │
│    ├─ Enhanced: triangle count, PageRank, reciprocity   │
│    ├─ Category: tool category transition anomalies      │
│    └─ Temporal: burst detection, interval analysis      │
│  Layer 3: Multi-dimension anomaly scoring               │
│    ├─ Parameter anomaly (single-round)                  │
│    ├─ Tool combination anomaly (single-round)           │
│    ├─ Content anomaly (embedding-based)        ← NEW    │
│    ├─ Transition + frequency anomaly (baseline)         │
│    └─ Graph structure anomaly (multi-round)             │
│  Layer 4: Decision                                       │
│    ├─ Adaptive threshold (SPC, 3-sigma)        ← NEW    │
│    └─ Fixed threshold fallback                          │
└─────────────────────────────────────────────────────────┘
```

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

> **Note:** The new `run_exp3.py` fixes the baseline training bug and adds content-aware detection.
> See [Known Issues](#known-issues) for details on previous bugs.

## Attack Taxonomy

| # | Category | Mechanism | Detection Signal |
|---|----------|-----------|-----------------|
| 1 | Delayed Trigger | Session A: inject via compliance disguise → Session B: trigger on benign query | Cross-session cumulative score |
| 2 | Gradual Escalation | Multi-turn chain: benign steps → malicious step → cover tracks | Graph entropy + novelty ratio + category transitions |
| 3 | Memory Poisoning | Session A: store poisoned memory → Session B: LLM acts on it | Cross-session graph structure change |
| 4 | Tool Abuse + Cover Tracks | Execute malicious operation → delete evidence | High-density multi-tool chains + temporal burst |
| 5 | Indirect Injection | Malicious content in email/document → LLM reads and follows | Content-triggered anomaly (embedding-based) |

## Project Structure

```
agent-attack-detection/
├── main.py                  # Entry point
├── run_exp.py               # Basic DeepSeek API experiments (v1)
├── run_exp2.py              # Function Calling API + stateful env (v2)
├── run_exp3.py              # Enhanced detector + content-aware (v3)  ← NEW
├── agent/
│   ├── core.py              # Legacy agent (LLMAgent, APIAgent)
│   ├── env.py               # BankingEnvironment with real state
│   ├── function_agent.py    # Function Calling Agent (OpenAI SDK)
│   └── types.py             # ToolCall data types
├── attack/
│   └── scenarios.py         # Attack scenario definitions
├── detection/
│   ├── __init__.py
│   ├── graph_detector.py    # Base Cross-Round Behavior Graph Detector
│   ├── neural_detector.py   # Enhanced: GNN + content + adaptive     ← NEW
│   └── baselines.py         # AgentShield + Leong + Random baselines
├── experiments/             # Legacy experiment scripts
├── data/                    # Results output
└── figures/                 # Generated figures
```

## Setup

```bash
# Base dependencies
pip install numpy networkx openai sentence-transformers scikit-learn

# Optional: GNN support
pip install torch

# Run experiment
python run_exp3.py
```

## Experiment Versions

| Script | Detector | Features | Status |
|--------|----------|----------|--------|
| `run_exp.py` | `APIAgent` + `MultiLayerDetector` | Basic hand-crafted features | Legacy |
| `run_exp2.py` | `FunctionCallingAgent` + `MultiLayerDetector` | Function calling API, stateful env | Active (has known bugs) |
| `run_exp3.py` | `FunctionCallingAgent` + `EnhancedMultiLayerDetector` | Content embedding, enhanced features, adaptive threshold | ✅ Recommended |

## Known Issues (已修复)

- ~~**Baseline training bug**: `BehavioralBaseline.update()` was never called during training in `MultiLayerDetector.analyze_call()`.~~ ✅ **已修复** — `train_on()` 方法已添加到 `MultiLayerDetector` 和 `EnhancedMultiLayerDetector`，`run_exp3.py` 在训练阶段显式调用。
- ~~**Document import missing**: `function_agent.py` 中 `inject_content()` 使用了未导入的 `Document` 类。~~ ✅ **已修复** — 已添加 `from agent.env import Document`。
- The legacy experiments in `experiments/` use simulated data (not real LLM calls). Use `run_exp3.py` for real API experiments.

## To-Do (Next Steps)

- [ ] Run `run_exp3.py` with DeepSeek API to validate DR improvement
- [ ] Add adaptive attack evaluation (white-box attacker knowing detector internals)
- [ ] Compare against MCPShield, FragBench, CASPIAN baselines
- [ ] Add ROC curves and AUC metrics
- [ ] Add cross-model transfer experiments (DeepSeek → GPT → Qwen)
- [ ] Add GAMMAF standardized evaluation
- [ ] Add k-fold cross-validation with statistical significance tests
