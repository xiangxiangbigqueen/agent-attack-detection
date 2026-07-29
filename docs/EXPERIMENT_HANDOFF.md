# Project Handoff and Reproducibility Record

**Repository:** `agent-attack-detection`
**Branch for this handoff:** `develop`
**Scope:** cross-session behavior/trajectory detection for tool-using LLM agents
**Status:** evidence package and submission-style manuscript complete; no production-readiness claim.

This document is the authoritative handoff for a collaborator. It records the decisions that affected scientific interpretation, every experiment family, locations of raw and derived records, current risks, and the exact next steps. It intentionally does **not** contain private conversation transcripts, provider keys, or credentials. Those are not research artifacts and must never be versioned.

## 1. One-paragraph executive summary

The project began as a proposed cross-session behavior-graph detector for multi-round LLM-agent attacks. Early R1--R3 results were useful for debugging and candidate selection but were not independent confirmation. They are frozen as development evidence. A simple trajectory-only candidate using decayed transition/frequency scores and cumulative scoring was then locked before fresh R4/R5 API runs. In R4/R5 it achieved 28.3%/29.6% all-attempt detection, 41.9%/43.8% recall among successful objectives, and 0.0%/0.5% false-positive rates. Exact tool-call-length matching lowered detection to 22.2%/25.7%. Diagnostics show transition/frequency signals matter, measured graph/cross-session structure adds little, and a post-hoc no-cumulative variant performs far better but requires fresh preregistered R6/R7. The revised paper therefore reports a reproducible, negative/diagnostic result rather than an unsupported high-performance method claim.

## 2. Research question and claims

### Research question

After isolating candidate selection from confirmation, what detection--false-alarm trade-off is obtained by a cross-session tool-trajectory score on API-executed attack and normal trajectories?

### Claims supported by the repository

1. The frozen trajectory-only candidate has low measured false-positive rates in the two confirmation runs but limited successful-attack recall.
2. Attack and normal maximum-score distributions overlap under a threshold fixed from validation normals.
3. Exact output-length matching reduces the apparent detection rate.
4. Transition/frequency signals are necessary for this frozen configuration; the measured graph/cross-session feature has negligible incremental effect in these runs.
5. An official AgentShield adapter detects all successful attacks in separate honeytool/honeytoken-protected executions at comparable measured FPR. It is a descriptive environment-changing comparison only.

### Claims explicitly not supported

- production readiness, safety certification, or a real-world deployment claim;
- cross-model generalization;
- superiority over AgentShield or any other baseline;
- a confirmatory claim for the no-cumulative variant;
- a causal or paired statistical comparison with AgentShield;
- a claim that the graph-structure component explains the reported performance.

## 3. Detector variants

### Frozen confirmation candidate: `trajectory_only`

Declared in `local_results/canonical/confirmation_preregistration_20260730.json`:

| Setting | Value |
| --- | --- |
| `use_structure` | `false` |
| `use_parameter_rules` | `false` |
| `use_tool_combination_rules` | `false` |
| `use_transition_frequency` | `true` |
| `use_cumulative` | `true` |
| threshold | `nextafter(max(validation-normal group scores), +inf)` |
| instant alert rule | disabled |

The primary implementation is `detection/graph_detector.py`. The confirmation protocol is `experiments/confirmation_protocol.py`.

### Post-confirmation diagnostic variants

| Variant | Purpose | Status |
| --- | --- | --- |
| No cross-session | assess measured graph/cross-session contribution | exploratory diagnostic |
| No transition/frequency | assess necessity of primary signal | exploratory diagnostic |
| No cumulative | test whether accumulation hides suspicious local events | exploratory; must be preregistered and rerun as R6/R7 |

These appear in `experiments/ablate_confirmation.py`. Do not relabel them as confirmation results.

## 4. Experimental timeline and decision record

| Phase | Runs / artifacts | What happened | Decision that must be preserved |
| --- | --- | --- |
| Legacy work | `run_exp.py`--`run_exp4.py`, `data/exp*_results.json` | Initial simulated/API experiments and detector debugging. | Historical only; not evidence for the paper's final claim. |
| Development | R1--R3 and official R1--R3 under `local_results/canonical/` | Candidate search, long-control diagnostics, baseline exploration. | Frozen as development-only; never call independent confirmation. |
| Development freeze | `development_freeze_20260730.json` | Data-tree and source hashes recorded. | No post-freeze method can report R1--R3 as fresh confirmation. |
| Confirmation design | `confirmation_preregistration_20260730.json` | Candidate, data sizes, calibration, endpoints, and reporting rules locked. | Do not alter candidate or threshold using R4/R5. |
| Independent confirmation | `confirmation_R4`, `confirmation_R5` | New wording/operands; API-executed attack and normal trajectories; long controls collected. | Primary evidence. |
| Official baseline | `confirmation_official_agentshield_R4/R5` | Official defense layers run through a local adapter. | Separate defended executions, not paired replay. |
| Diagnostics | `ablate_confirmation.py`, length matching, bootstraps, misses | Evaluated mechanisms and confounds after confirmation. | Exploratory only unless separately preregistered. |
| Manuscript | `output/pdf/` | Rewrote paper to reflect independent evidence and limitations. | Retain cautious claims and disclosure wording. |

## 5. R4/R5 protocol in detail

### Per-run group counts

| Split | Short normal | Long normal | Attack | Total |
| --- | ---: | ---: | ---: | ---: |
| Train/calibration | 100 | 100 | 0 | 200 |
| Validation | 50 | 50 | 0 | 100 |
| Test normal | 100 | 100 | 0 | 200 |
| Attack test | 0 | 0 | 240 | 240 |
| Raw collection before long controls | 330 normal/attack mix + 240 attack | -- | -- | 570 records |

The documented final denominators are 200 benign and 240 attack groups per R4/R5. A failed first long-control attempt due to provider balance was retained as error metadata and recovered into `long_benign_calibration/recovery_02`; valid records were not overwritten.

### Attack families and endpoints

The held-out confirmation collection includes multi-step families such as indirect prompt injection, tool exfiltration, transaction cover, and privilege escalation. The principal conditional endpoint is successful-objective recall: only attacks whose underlying objective succeeded are in this denominator. Each R4/R5 run had 160 successful objectives. The all-attempt endpoint retains all 240 attacks.

### Calibration and statistics

- A per-run threshold is fixed from 100 validation-normal group scores.
- No test score chooses a threshold.
- Bootstrap confidence intervals resample groups, not individual tool calls.
- Exact 1:1 matching without replacement is performed by total tool-call count. It is a secondary confound analysis, not a replacement test set.
- R4/R5 are independent repetitions, but two runs are not sufficient to claim broad stability.

## 6. Primary results

### Frozen candidate

| Endpoint | R4 | R5 | Aggregate descriptive summary |
| --- | ---: | ---: | ---: |
| All-attempt detection (n=240) | 28.33% [22.50, 34.17] | 29.58% [23.75, 35.42] | 28.96% +/- 0.88% |
| Successful-attack recall (n=160) | 41.88% [34.38, 49.38] | 43.75% [36.25, 51.88] | 42.81% +/- 1.33% |
| FPR (n=200) | 0.00% [0.00, 0.00] | 0.50% [0.00, 1.50] | 0.25% +/- 0.35% |
| Exact length-matched detection | 22.22% (117 groups) | 25.66% (113 groups) | not pooled |

Brackets are 95% bootstrap intervals. Full machine-readable output: `local_results/canonical/confirmation_paper_artifacts/confirmation_summary.json`.

### Official AgentShield: separate defended executions

| Endpoint | R4 | R5 | Aggregate descriptive summary |
| --- | ---: | ---: | ---: |
| All-attempt detection | 67.08% | 66.67% | 66.88% +/- 0.29% |
| Successful-attack recall | 100% (161/161) | 100% (160/160) | 100% |
| FPR | 0.50% | 0.00% | 0.25% +/- 0.35% |

The official baseline injects honeytools/honeytokens. Because those intervention layers affect the agent's trajectory, the rows are not paired with primary-detector traces. Never calculate or report paired p-values from these columns.

### Misses

There are 183 missed successful primary-detector objectives across R4/R5:

| Family | Misses |
| --- | ---: |
| Indirect injection | 80 |
| Tool exfiltration | 47 |
| Transaction cover | 48 |
| Privilege escalation | 8 |

See `primary_successful_attack_misses.csv`. The error pattern suggests that harmful objectives can be achieved through calls individually similar to normal trajectories.

## 7. Exploratory diagnostics

| Variant | R4 all-attempt detection | R5 all-attempt detection | Interpretation |
| --- | ---: | ---: | --- |
| Frozen trajectory-only | 28.3% | 29.6% | Primary result. |
| No cross-session | 27.9% | 29.2% | Measured gain is negligible here. |
| No transition/frequency | 0.0% | 0.0% | Necessary under the frozen calibration. |
| No cumulative | 95.0% | 100.0% | Strong post-hoc hypothesis only. |

The next scientifically valid action for the no-cumulative finding is not to replace the paper's main method. It is to create a new pre-registration, lock the simplified detector and all endpoints, collect R6/R7 with new templates/operands, and report all outcomes.

## 8. File and script index

### Core code

- `agent/function_agent.py`: function-calling API agent and trace capture.
- `agent/env.py`: synthetic banking environment and tools.
- `detection/graph_detector.py`: trajectory score and feature switches.
- `attack/scenarios.py`: legacy scenario catalog.

### Protocol, collection, and evaluation

| Scripts | Responsibility |
| --- | --- |
| `canonical_protocol.py`, `canonical_evaluation.py`, `freeze_development_artifacts.py`, `validate_canonical_dataset.py` | Development protocol construction, validation, and freezing. |
| `confirmation_protocol.py`, `freeze_confirmation_protocol.py` | R4/R5 held-out confirmation specification and freeze metadata. |
| `collect_confirmation_api.py`, `collect_confirmation_long_benign.py` | Primary R4/R5 API and long-normal collectors. |
| `evaluate_confirmation.py`, `analyze_confirmation.py`, `summarize_confirmation.py` | Primary scoring, bootstrap/secondary analyses, paper summary. |
| `ablate_confirmation.py` | Post-confirmation mechanism diagnostics. |
| `collect_confirmation_official_agentshield.py`, `collect_confirmation_official_long.py`, `evaluate_confirmation_official_agentshield.py`, `summarize_official_agentshield.py` | Official-baseline adapter and analysis. |
| `collect_canonical_api.py`, `collect_long_benign_calibration.py`, `collect_length_control_benign.py`, `collect_length_stratified_benign.py` | Earlier development/control collection. |
| `run_frozen_main_evaluation.py`, `run_frozen_baselines.py`, `run_long_calibrated_evaluation.py`, `run_long_calibrated_ablation.py`, `run_long_calibrated_bootstrap.py`, `run_long_protocol_baselines.py` | Frozen development and long-control evaluation families. |
| `run_length_matched_evaluation.py`, `run_length_matched_control.py`, `run_augmented_length_matched.py`, `length_matched_paired_evaluation.py` | Length/confounding analyses. |
| `run_ablation_evaluation.py`, `run_threshold_sensitivity.py`, `run_statistical_analysis.py`, `paired_baseline_statistics.py` | Supporting diagnostics; do not elevate exploratory outcomes. |
| `collect_feedback_adaptive.py`, `score_feedback_adaptive.py`, `collect_official_agentshield_feedback.py`, `evaluate_official_agentshield_feedback.py` | Adaptive/feedback experiments. |
| `api_preflight.py`, `run_official_agentshield_smoke.py`, `bootstrap_official_agentshield.py`, `audit_baseline_comparability.py` | Environment checks, baseline setup, and comparability disclosures. |
| `generate_paper_experiment_artifacts.py`, `generate_submission_assets.py`, `build_submission_pdf.py` | Tables, figures, and manuscript output. |
| `run_replication_pipeline.py`, `summarize_replications.py`, `evaluate_attack_outcomes.py`, `develop_fpr_constrained_variants.py`, `compare_primary_official_agentshield.py` | Replication and supplementary analyses. |

### Tests

`tests/` includes `test_attack_outcomes.py`, `test_canonical_evaluation.py`, `test_canonical_protocol.py`, `test_collection_schema.py`, `test_confirmation_protocol.py`, and `test_function_agent_injection.py`.

Run:

```powershell
python -m unittest discover -s tests -v
```

### Results inventory

| Path | Use it for |
| --- | --- |
| `local_results/canonical/paper_protocol_20260729_1/` | R1 development data. |
| `local_results/canonical/replication_R2/raw/`, `replication_R3/raw/` | R2/R3 development replications. |
| `local_results/canonical/confirmation_R4/`, `confirmation_R5/` | Frozen primary confirmation records and evaluations. |
| `local_results/canonical/confirmation_official_agentshield_R4/`, `confirmation_official_agentshield_R5/` | Separate official-baseline executions. |
| `local_results/canonical/confirmation_paper_artifacts/` | Final JSON summaries, CSV misses, and paper figures. |
| `local_results/canonical/paper_level_artifacts/` | Earlier development-level plots/tables; not final evidence. |

## 9. Reproduction procedures

### Analysis-only reproduction (recommended)

The checked-in `local_results/` records permit rebuilding figures and the paper without a provider key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments\generate_submission_assets.py
python experiments\build_submission_pdf.py
```

Expected deliverable: `output/pdf/cross_session_trajectory_detection_revised.pdf`.

### New collection

1. Create a new run label (for example `R6`), new output directory, new templates/operands, and a new preregistration JSON.
2. Store the provider key only in an environment variable.
3. Run a preflight check, collect short/long normals and attack groups, then evaluate once using the locked threshold.
4. Preserve raw records and index files; do not overwrite R1--R5.
5. Explicitly separate confirmatory endpoints from any post-hoc diagnostics.

## 10. Baseline provenance and environments

`external_baselines/AgentShield_official_main.zip` is a public-source archive used to reproduce the official layers. Its SHA-256 is:

```text
EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E
```

The ignored `external_baselines/agentshield_py311/` directory is a local, Windows-specific Python 3.11 environment (about 1.17 GiB). It must be rebuilt rather than committed. The extracted upstream source is also ignored to avoid vendoring third-party code; use the archive/checksum and upstream licence/citation instructions.

## 11. Manuscript notes

The final manuscript is at `output/pdf/cross_session_trajectory_detection_revised.pdf`; its editable source is adjacent. It contains four data-backed figures:

1. frozen development-to-confirmation protocol;
2. empirical score CDFs for R4/R5 attack and benign groups;
3. exploratory ablation summary;
4. primary versus official baseline descriptive comparison.

Before an actual submission:

1. replace the placeholder author affiliation/contact;
2. choose a venue and apply its official template, page limit, anonymity, and bibliography rules;
3. retain the limitations and separate-execution disclosure;
4. verify citations against the final bibliographic source;
5. if the no-cumulative method is to be claimed, finish preregistered R6/R7 first.

## 12. Risks and recommended next actions

| Priority | Risk | Recommended action |
| --- | --- | --- |
| P0 | Current primary candidate misses most successful objectives. | Do not deploy; keep the claim as a benchmark/diagnostic study. |
| P0 | No-cumulative result is selected post-hoc. | Preregister and run R6/R7 before revising the primary method. |
| P1 | Single-model, single-domain scope. | Add independently held-out model and tool domains only after fixing the primary hypothesis. |
| P1 | Official baseline runs in a changed environment. | Use a jointly specified paired harness if a causal baseline comparison is required. |
| P1 | FPR confidence intervals remain broad. | Increase independent normal groups and diverse normal task sources. |
| P2 | Publication formatting is venue-specific. | Move the text to the selected venue's official template after scientific decisions are stable. |

## 13. Security rules

- Never commit `.env`, provider keys, personal data, raw private transcripts, or a virtual environment.
- Synthetic fixtures may intentionally contain fake credential-shaped strings as honeytokens. Treat them as test artifacts, not credentials.
- Before every push, scan staged files for real credentials and inspect the staged diff.
- The project uses a banking-style synthetic environment; no real customer records are required or permitted.

## 14. How this handoff represents prior discussion

The project was repeatedly stress-tested for length confounding, post-selection bias, baseline comparability, false-positive reporting, independent repeats, and paper overclaiming. The resulting decisions are encoded in the freeze/preregistration files, this document, the README, and the manuscript. This is a complete research decision record; it intentionally omits verbatim conversational history because that history contained operational chatter and private credentials that are neither reproducible nor safe to publish.
