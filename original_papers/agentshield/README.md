# AgentShield

A multi-layer deception-based detection framework for identifying when tool-using AI agents have been compromised by indirect prompt injection.

## Overview

AgentShield detects agent compromise through three independent mechanisms:

1. **Honeytools** (Layer 1): Decoy tools registered alongside real tools. Only a compromised agent would call them.
2. **Honeytokens** (Layer 2): Fake credentials planted in the environment. An egress monitor flags exfiltration attempts.
3. **Parameter Validator** (Layer 3): Allowlisting rules that flag tool calls with suspicious parameters (e.g., unknown IBANs, external emails).

AgentShield is **detection-only** --- it monitors and logs but does not block tool execution. This makes it complementary to prevention-based defenses at near-zero computational cost.

## Key Results

- **Detection rate:** 90.7--100% on clearly successful attacks (commercial models), 0% false positives across all experiments
- **Cross-lingual evaluation:** 128 attack prompts in English, Kurdish (Sorani), Arabic, and code-switched variants
- **4 models tested:** GPT-4o-mini, GPT-5-mini, Llama 3.3 70B, DeepSeek-V3 (2 commercial, 2 open-source)
- **Cross-lingual gap:** 6.4pp (GPT-4o-mini) narrowing to 1.9pp (DeepSeek-V3) between English and Kurdish

## Project Structure

```
agentshield/
  defenses/              # Core framework
    pipeline.py          # Pipeline builder (supports OpenAI, Together AI, Ollama)
    honeytools.py        # Layer 1: Honeytool detector
    honeytokens.py       # Layer 2: Honeytoken monitor
    parameter_validator.py  # Layer 3: Parameter allowlisting
    auto_honeytools.py   # LLM-generated suite-specific honeytools
  attacks/
    attack_prompts.py    # 128 cross-lingual attack prompts (EN/KU/AR/CS)
    Revised_Attack_Prompts_Full_UTF8_updated.xlsx  # Source of truth
  results/               # Experiment results (timestamped JSON)
  run_all_suites.py      # Main experiment runner (GPT models)
  run_together.py        # Open-source model runner (via Together AI)
  run_jisa_baseline.py   # Baseline defense comparison (no defense / spotlighting)
  run_ablation.py        # Ablation study (per-layer contribution)
  run_repeated_trials.py # Repeated trials for statistical significance
  compute_statistics.py  # Chi-squared, Cohen's h, Wilson CIs
  analyze_false_negatives.py  # False negative classification
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install agentdojo openai python-dotenv

# For open-source models via Together AI
pip install together

# Add API keys to .env
echo "OPENAI_API_KEY=your_key" >> .env
echo "TOGETHER_API_KEY=your_key" >> .env  # optional, for Llama/DeepSeek
```

## Running Experiments

```bash
# Run 128 attacks across all 4 suites with AgentShield (GPT-4o-mini)
python -m agentshield.run_all_suites

# Run with open-source models via Together AI
python -m agentshield.run_together --model llama3.3-70b
python -m agentshield.run_together --model deepseek-v3

# Run baseline defense comparison (no defense vs spotlighting)
python -m agentshield.run_jisa_baseline --model gpt-4o-mini --all-defenses

# Run ablation study
python -m agentshield.run_ablation --model gpt-4o-mini-2024-07-18

# Run repeated trials for statistical significance
python -m agentshield.run_repeated_trials
```

## Attack Prompts

The 128 cross-lingual attack prompts span 4 languages and multiple categories:

| Category | Count | Description |
|----------|-------|-------------|
| Goal hijacking | 16 | Redirect agent to attacker-controlled actions |
| Data exfiltration | 16 | Extract sensitive data to external endpoints |
| Tool misuse | 16 | Abuse legitimate tools for malicious purposes |
| Adaptive | 16 | Context-aware attacks that adapt to tool outputs |
| Encoding-based | 16 | Zero-width, transliteration, homoglyph attacks |
| Domain-agnostic (Set B) | 48 | Cross-suite goal-based attacks |

Languages: English (EN), Kurdish Sorani (KU), Arabic (AR), Code-switched EN/KU (CS)

## Benchmark

AgentShield is evaluated on [AgentDojo](https://github.com/ethz-spylab/agentdojo) (v1.2.2), an open-source benchmark for testing prompt injection attacks and defenses on tool-using LLM agents. AgentDojo provides 4 agent suites: banking, messaging (Slack), travel, and workspace.

## Citation

If you use AgentShield in your research, please cite:

```bibtex
@article{rassul2026agentshield,
  author  = {Rassul, Yassin H. and Rashid, Tarik A.},
  title   = {{AgentShield}: Deception-based Compromise Detection for Tool-using {LLM} Agents},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/Yassin-H-Rassul/AgentShield}
}
```

(Update the entry with the assigned `arXiv:XXXX.XXXXX` identifier once the preprint is announced.)

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Yassin H. Rassul, Tarik A. Rashid.
