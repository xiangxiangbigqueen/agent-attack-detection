# Cross-Session Tool-Trajectory Detection for LLM Agents

This repository is the reproducibility package for an empirical study of multi-round attacks against tool-using LLM agents. It contains the detector, API-executed experimental protocol, raw local experiment records, analysis scripts, an official-baseline adapter, and a submission-style manuscript.

**Current scientific status:** this is not a production-ready detector and this repository does not claim one. The independent R4/R5 confirmation shows low false-positive rates but limited recall for successful attacks. The central contribution is a frozen evaluation protocol and an auditable account of the detection--false-alarm trade-off.

Chinese summary: 本项目不是“高分检测器已可部署”的宣传版本。R4/R5 独立确认实验表明：误报低，但对成功攻击的召回有限；项目的价值在于冻结协议、可复现轨迹数据和诚实的机制诊断。

## Repository map

| Path | Contents |
| --- | --- |
| `agent/` | Tool-using banking agent, environment, and function-calling adapter. |
| `detection/` | Cross-session/trajectory scoring implementation. |
| `attack/` | Attack scenario definitions used by legacy experiments. |
| `experiments/` | Protocol construction, API collectors, evaluators, statistics, baseline adapter, and paper builders. |
| `local_results/canonical/` | Frozen R1--R3 development artifacts; independent R4/R5 records, metrics, and paper tables/figures. |
| `external_baselines/AgentShield_official_main.zip` | Archived public official AgentShield source used for the adapter; see the provenance section below. |
| `tests/` | Unit tests for protocol, collection schema, evaluator, outcomes, and injection handling. |
| `output/pdf/` | Corrected submission manuscript, source builder, and data-backed figure assets. |
| `docs/EXPERIMENT_HANDOFF.md` | Detailed handoff: all decisions, runs, data locations, metrics, risks, and next actions. |

## Headline independent results

The trajectory-only candidate was selected using development runs R1--R3, frozen, and then evaluated on two new API-executed runs (R4/R5). Each confirmation run contains 240 attack groups and 200 test-normal groups. The threshold is the smallest representable float above the maximum of 100 benign validation group scores; test data never set the threshold.

| Metric | R4 | R5 | Macro mean +/- sample SD |
| --- | ---: | ---: | ---: |
| All-attempt detection | 28.3% [22.5, 34.2] | 29.6% [23.8, 35.4] | 29.0% +/- 0.9% |
| Successful-attack recall | 41.9% [34.4, 49.4] | 43.8% [36.3, 51.9] | 42.8% +/- 1.3% |
| False-positive rate | 0.0% [0.0, 0.0] | 0.5% [0.0, 1.5] | 0.25% +/- 0.35% |
| Exact length-matched detection | 22.2% [14.5, 29.9] | 25.7% [17.7, 33.6] | -- |

Intervals are 95% group bootstrap intervals. Full values, denominators, and exact matching strata are in `local_results/canonical/confirmation_paper_artifacts/confirmation_summary.json`.

## Important interpretation

- The measured cross-session component provides almost no gain in the post-confirmation diagnostic: 28.3% to 27.9% in R4 and 29.6% to 29.2% in R5 when removed.
- Transition/frequency signals are necessary under the frozen calibration; removing them gives 0% all-attempt detection in both runs.
- Removing cumulative scoring gives 95%/100%, but this was discovered after R4/R5. It is **exploratory**, not a confirmatory replacement. A preregistered R6/R7 replication is required before making a new method claim.
- Official AgentShield measurements are from separate honeytool/honeytoken-protected executions. They are descriptive, not paired comparisons and not evidence of a causal performance difference.
- Only `deepseek-chat` and the banking-style tool sandbox are in scope. No cross-model or real-world deployment claim is supported.

## Paper

- Final complete-length submission PDF: `output/pdf/behaviorgraph_submission_complete.pdf`
- Complete-length source builder: `experiments/build_complete_submission_pdf.py`
- Short corrected PDF retained for provenance: `output/pdf/behaviorgraph_submission_final.pdf`
- Short-form source builder: `experiments/build_final_submission_pdf.py`
- Corrected figure generator: `experiments/generate_corrected_figures.py`
- Corrected figure manifest: `output/pdf/assets_corrected/manifest.json`
- Independent consistency audit: `output/audit/submission_consistency.json`
- Method/threshold audit: `docs/METHOD_CONSISTENCY_AUDIT.md`
- Earlier draft retained for provenance: `output/pdf/cross_session_trajectory_detection_revised.pdf`

The complete-length manuscript deliberately distinguishes the full BehaviorGraph framework from the independently confirmed trajectory-only candidate (TO-CF). It uses six corrected figures with one-to-one figure/caption/data correspondence, preserves the full methods/results/limitations narrative, and includes an audit appendix. Adapt the author metadata, bibliography style, and page limits to the target venue before submission.

## Reproduce the analyses

Python 3.11 is recommended. Create a fresh environment; the ignored `external_baselines/agentshield_py311/` directory is a machine-local virtual environment and is not part of the repository.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v

# Regenerate data-backed publication assets and the manuscript.
python experiments\generate_submission_assets.py
python experiments\build_submission_pdf.py
```

API collection requires a valid `DEEPSEEK_API_KEY` in the environment. Never place it in a tracked file.

```powershell
$env:DEEPSEEK_API_KEY = "<your-key>"
python experiments\collect_confirmation_api.py --run R4
python experiments\collect_confirmation_long_benign.py --run R4
python experiments\evaluate_confirmation.py --run R4
python experiments\analyze_confirmation.py
```

Collection scripts can incur API cost. Do not rerun R4/R5 into their existing directories: preserve the frozen records and use a fresh run label for any new collection.

## Protocol and data

The protocol is documented as machine-readable artifacts:

- `local_results/canonical/development_freeze_20260730.json`: freezes R1--R3 as development/diagnostic evidence only, including source and data-tree hashes.
- `local_results/canonical/confirmation_preregistration_20260730.json`: locked R4/R5 candidate, thresholds, sample sizes, primary metrics, and reporting rules.
- `local_results/canonical/confirmation_paper_artifacts/confirmation_summary.json`: paper-ready R4/R5 and official-baseline summaries.
- `local_results/canonical/confirmation_paper_artifacts/primary_successful_attack_misses.csv`: error taxonomy for the 183 missed successful attacks.

The `local_results/` directory contains approximately 10,000 JSON/JSONL/CSV records (about 14 MiB) and is intentionally versioned. Raw records are synthetic banking-sandbox trajectories; they contain no real customer data or API credentials. Some attack fixtures deliberately contain **fake trap strings** resembling credentials. They are test data, not usable secrets.

## Official baseline provenance

The archival source `external_baselines/AgentShield_official_main.zip` has SHA-256:

```text
EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E
```

The adapter scripts are `experiments/collect_confirmation_official_agentshield.py`, `experiments/collect_confirmation_official_long.py`, and `experiments/evaluate_confirmation_official_agentshield.py`. The public source itself may be obtained from the upstream project; its local extracted copy and the local Python environment are excluded from Git to avoid vendoring third-party source and 1+ GiB of machine-specific binaries.

## Security and versioning rules

- `.env`, virtual environments, temporary renderings, and local installation logs are ignored.
- Do not commit keys, provider responses containing real secrets, or private chat transcripts. This repository contains a decision record, not private conversations.
- Do not overwrite frozen R1--R5 data. New hypotheses require a new labeled run and a new preregistration file.
- Do not report R4/R5 exploratory ablations as independent confirmation.
- Do not call the separate AgentShield comparison paired, statistically significant, or a direct winner.

## What to read first

1. `docs/EXPERIMENT_HANDOFF.md`
2. `local_results/canonical/confirmation_preregistration_20260730.json`
3. `local_results/canonical/confirmation_paper_artifacts/confirmation_summary.json`
4. `output/pdf/cross_session_trajectory_detection_revised.pdf`
5. `experiments/evaluate_confirmation.py` and `detection/graph_detector.py`

## License and attribution

This repository includes an archived copy of publicly obtained official AgentShield source only for reproduction provenance. Respect upstream licensing and citation requirements before redistributing or modifying that baseline.
