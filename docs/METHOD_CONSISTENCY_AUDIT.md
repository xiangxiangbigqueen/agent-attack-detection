# Method and Threshold Consistency Audit

**Audit date:** 2026-08-10
**Scope:** repository implementation, R4/R5 confirmation protocol, experiment outputs, and the revised manuscript source.
**Purpose:** provide an explicit source-of-truth for the method that is actually confirmed, identify conflicts with legacy/full-model code, and give text/formula changes required before submission.

## Executive finding

The repository contains a broad `BehaviorGraph`/`MultiLayerDetector` implementation, but the independently confirmed R4/R5 system is a deliberately reduced **trajectory-only candidate**.  R4/R5 set `use_structure=False`, `use_parameter_rules=False`, and `use_tool_combination_rules=False`; only transition/frequency scores and cumulative scoring are enabled.  Therefore the confirmed result must not be described as an evaluation of the complete BehaviorGraph.  The paper may use BehaviorGraph as the original framework or implementation context, but its primary empirical claim must name the frozen trajectory-only candidate explicitly.

The canonical R4/R5 threshold is also unambiguous: the next representable floating-point value above the maximum of 100 validation-group scores.  Other scripts in the repository (development/legacy evaluators) use a P95 quantile.  Those scripts must be labelled as non-canonical in the manuscript and README; otherwise a reader can reproduce a different threshold and obtain different results.

## Evidence map

| Question | Repository evidence | Consequence |
|---|---|---|
| What is the full implementation? | `detection/graph_detector.py:24-50` (`DetectorConfig`) enables cumulative, transition/frequency, graph structure, parameter rules, and tool-combination rules by default. `BehaviorGraph` begins at `:102` and tracks cross-session edges/decay (`:159-200`). | This is a framework/full model, not automatically the confirmed model. |
| What was confirmed in R4/R5? | `experiments/evaluate_confirmation.py:18-24` defines `CANDIDATE`: structure, parameter, and combination rules are false; transition/frequency and cumulative are true. The same configuration is recorded in `local_results/canonical/confirmation_preregistration_20260730.json` (`candidate`). | R4/R5 results are trajectory-only results. |
| How is the R4/R5 score obtained? | `evaluate_confirmation.py:60-76` fits a benign baseline, preserves the detector across ordered sessions, and records the maximum `cumulative_score` over all calls in a group. | The evaluated group statistic is a maximum cumulative score, not the `DetectionResult.is_attack` decision from `get_decision`. |
| How is the canonical threshold obtained? | `evaluate_confirmation.py:97` uses `np.nextafter(max(validation_scores.values()), np.inf)`; `:110-115` records the rule and applies `score >= threshold`. The preregistration states the same rule. | Use one formula everywhere in the confirmation manuscript. |
| Where do conflicting P95 rules remain? | `experiments/canonical_evaluation.py:121` and `experiments/run_frozen_main_evaluation.py:93` use `np.quantile(..., .95, method="higher")`; multiple older scripts (`run_long_calibrated_evaluation.py`, `step1_real_api.py`, `steps_234.py`, `unified_benchmark.py`) also use P95. | These are development/legacy protocols and must not be mixed with R4/R5 or cited as the confirmation threshold. |
| What is the exploratory ablation rule? | `experiments/ablate_confirmation.py:45-49` states post-confirmation exploratory status and recalibrates each variant with its own validation-maximum threshold. `NoCumulative` uses instant scores (`instant_only=True`). | No-cumulative 95%/100% is post-hoc; it is not the locked primary method. |
| What does the paper currently say? | `output/pdf/cross_session_trajectory_detection_revised.tex:28` describes the reduced candidate; `:94-95` calls cross-session gain negligible and no-cumulative post-hoc; `:115` requires R6/R7 before a method claim. | The revised source is directionally consistent, but the full/reduced distinction and exact score/threshold definition should be made explicit in Methods and a table. |

## Full BehaviorGraph versus frozen trajectory-only candidate

### Full implementation (framework, not the R4/R5 primary result)

The full code path contains the following components:

1. benign tool-frequency and transition baselines (`BehavioralBaseline`);
2. a directed graph with within-session and cross-session edges, edge decay, density, diversity, entropy, and novelty features (`BehaviorGraph`);
3. parameter anomaly rules (for example, external recipients and unusual accounts);
4. tool-combination rules (for example, export followed by email);
5. a weighted per-call score and a cumulative EWMA score;
6. an operational `get_decision` path with an instant-signal rule and a cumulative alert threshold.

These defaults are visible in `DetectorConfig` (`detection/graph_detector.py:24-50`) and the component scoring in `GraphAnomalyScorer.score_call` (`:348-401`). The default configuration has all optional components enabled. A paper sentence such as “our detector uses the BehaviorGraph, parameter checks, combination rules, and cumulative EWMA” is therefore a description of the full implementation, not of the confirmed R4/R5 candidate.

### Frozen R4/R5 candidate (the only primary confirmation claim)

The pre-registered candidate is exactly:

```text
use_structure = false
use_parameter_rules = false
use_tool_combination_rules = false
use_transition_frequency = true
use_cumulative = true
```

It still uses the shared `BehaviorGraph` object to maintain ordered calls and cross-session state, but the graph-structure score is excluded from the weighted anomaly score.  The candidate's measured signal is therefore decayed baseline transition/frequency deviation accumulated over the ordered trajectory.  This distinction must appear in the abstract, Methods, Results table headings, all figure captions, and the limitations section.

Recommended notation:

```text
Full-BG: the complete DetectorConfig with all feature families enabled.
TO-CF: the pre-registered trajectory-only confirmation candidate used in R4/R5.
```

Do not call TO-CF “the full BehaviorGraph detector,” and do not infer that the graph-structure component caused the reported recall.  The R4/R5 ablation measured less than one percentage point change when cross-session structure was removed, while disabling transition/frequency eliminated detection under that calibration.

## Canonical score and threshold definition

The manuscript should define the statistic before presenting any result.  For run (r), candidate configuration (C), and ordered group (g), let

\[
S_{r,C}(g) = \max_{c \in g} q_{r,C}(c),
\]

where (q_{r,C}(c)) is the cumulative EWMA score after call (c).  For a multi-session group, the detector state is retained between sessions and `reset_session()` is called at each boundary; the group maximum spans all sessions.  The long-control records are one-session groups, so their score is the maximum over that single record.

For the canonical TO-CF confirmation, (V_r) contains 100 benign validation groups (50 short protocol groups plus 50 independently collected long benign controls).  The only valid threshold is

\[
\theta_r = \operatorname{nextafter}\left(\max_{g \in V_r} S_{r,\mathrm{TO\text{-}CF}}(g), +\infty\right),
\]

and the group-level alert is

\[
A_{r}(g) = \mathbb{1}\left[S_{r,\mathrm{TO\text{-}CF}}(g) \geq \theta_r\right].
\]

No test score or test label selects (	heta_r).  The `nextafter` operation matters: it guarantees that a validation maximum itself is not counted as an alert under the `>=` comparison.  Do not replace this rule with “P95” in the R4/R5 paper text.

### Operational decision-path caveat

`GraphAnomalyScorer.get_decision` (`detection/graph_detector.py:410-429`) also contains an instant-signal alert (`instant >= 0.7`) and a configurable cumulative alert threshold.  The confirmation evaluator intentionally bypasses that decision path: it sets `alert_threshold=1e9`, collects group maxima, and applies the preregistered external threshold (`evaluate_confirmation.py:64-75, 97`).  The paper must say that R4/R5 evaluate a **locked score-and-threshold protocol**, not the default `get_decision` thresholds.  If the goal is to claim online detector decisions rather than score classification, a separate end-to-end decision-path experiment is required.

## Required manuscript changes

1. **Methods, first paragraph:** define Full-BG and TO-CF as separate configurations and state that only TO-CF is independently confirmed in R4/R5.
2. **Algorithm box:** show baseline fitting, ordered-session state retention, cumulative score update, group maximum, and pre-test threshold calibration.  Include the exact `nextafter(max(validation), +inf)` rule.
3. **Configuration table:** list all five Boolean feature switches and their values for Full-BG, TO-CF, and each exploratory ablation.  Include `decay_factor`, baseline smoothing, and `min_baseline_samples`.
4. **Results table/captions:** prefix primary values with “TO-CF (R4/R5)” and label no-cumulative/no-cross-session rows “exploratory post-confirmation.”
5. **Threshold paragraph:** remove “P95” wherever it refers to R4/R5.  P95 may remain only in a clearly labelled development/legacy appendix.
6. **Decision semantics:** distinguish the score protocol (`max cumulative score >= theta`) from `MultiLayerDetector.get_decision`.  Do not imply that the reported FPR/recall came from the default `alert_threshold=6.0`.
7. **Claim language:** replace “BehaviorGraph achieves …” with “TO-CF, a reduced candidate derived from the BehaviorGraph framework, achieved … in two independent DeepSeek confirmation runs.”
8. **Cross-session claim:** state that cross-session state is retained in the TO-CF evaluator, but the measured incremental contribution was negligible in these runs; no graph-structure superiority claim is supported.
9. **Ablation status:** explicitly say that no-cumulative uses a separately recalibrated instant-score threshold and was selected after observing R4/R5; it requires pre-registered R6/R7 before becoming a method claim.
10. **Reproducibility appendix:** link to `confirmation_preregistration_20260730.json`, `evaluate_confirmation.py`, and the immutable raw-data manifests.  Include the per-run thresholds and validation/test denominators.

## Required code/documentation hygiene (without changing results)

- Mark `canonical_evaluation.py`, `run_frozen_main_evaluation.py`, and older P95 scripts as **development/legacy** in their docstrings or README section.  They should not be presented as the R4/R5 evaluator.
- Add a single named constant or manifest field for the confirmation threshold protocol so future scripts cannot silently substitute P95.
- Add an evaluator assertion that the candidate flags equal the preregistered configuration and that validation count equals 100.
- Add a test that verifies the threshold is strictly greater than the validation maximum and that no test records are read during calibration.
- If Full-BG is retained in the title or contribution claim, preregister and run independent Full-BG confirmation (for example R6/R7).  Otherwise use “trajectory-only candidate” in the main title/abstract.

## Minimal recommended Methods text

> We distinguish the complete BehaviorGraph implementation (Full-BG) from the candidate evaluated in the independent confirmation.  The pre-registered trajectory-only candidate (TO-CF) disables parameter and tool-combination rules and the graph-structure score, while retaining benign transition/frequency deviations and cumulative scoring.  For each run, we fit the baseline on training normals, score every ordered group, and define the group statistic as the maximum cumulative EWMA score over all calls and sessions.  The threshold is the next representable floating-point value above the maximum of 100 benign validation-group scores (50 short and 50 long controls); test labels are never used for calibration.  A group is alerted iff its maximum score is greater than or equal to this frozen threshold.  Thus the R4/R5 results quantify TO-CF under a locked score protocol and should not be interpreted as independent confirmation of every Full-BG feature.

## Submission gate

The method section is internally consistent only when all of the following are true:

- every primary result names TO-CF (or explicitly names Full-BG if new confirmation has been run);
- the threshold formula is the single `nextafter` rule for R4/R5;
- P95 scripts are labelled development/legacy and are not used to populate confirmation tables;
- score-protocol alerts are not conflated with `get_decision` alerts;
- figure legends, captions, README, and machine-readable manifests use the same candidate name and feature switches;
- no-cumulative and no-cross-session numbers remain exploratory unless a new preregistration and independent confirmation are completed.
