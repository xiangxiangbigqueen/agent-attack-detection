# Paper Repositioning and CCF-C Submission Plan

**Status:** execution plan grounded in the current R4/R5 evidence
**Date:** 2026-08-10
**Scope:** redesign of the manuscript and experiment package; no new result is claimed unless it is produced by a preregistered run.

## 1. Decision and positioning

The current evidence does not justify presenting the system as a strongly validated cross-session detector. The independently confirmed system is the reduced **trajectory-only confirmation candidate (TO-CF)** derived from the broader BehaviorGraph implementation. Cross-session state is retained by the evaluator, but the observed removal effect is small (0.4 percentage points in both confirmation comparisons). The paper should therefore be positioned as a leakage-controlled measurement and boundary study unless a new, preregistered cross-session experiment demonstrates a statistically reliable gain.

### Recommended title

**How Much Can Tool Trajectories Reveal? A Leakage-Controlled Study of Cross-Session Attacks on LLM Agents**

### Prohibited title/claim until new confirmation

Do not use a title or abstract that states or implies that the complete BehaviorGraph is a superior cross-session detector, that cross-session memory is the source of the observed recall, or that the method is production-ready.

## 2. Claims supported by current evidence

The following numbers are the only primary confirmation results currently available:

| Quantity | R4 | R5 |
|---|---:|---:|
| Frozen threshold | 5.0040541339 | 4.9817326042 |
| Validation alerts | 0/100 | 0/100 |
| Benign test alerts | 0/200 | 1/200 |
| All-attempt attack alerts | 68/240 (28.33%) | 71/240 (29.58%) |
| Successful-objective alerts | 67/160 (41.88%) | 70/160 (43.75%) |
| Exact total-tool-call-count matched alerts | 26/117 (22.22%) | 29/113 (25.66%) |

Pooled descriptive values are 139/480 (29.0%) all-attempt detection, 137/320 (42.8%) successful-objective recall, and 1/400 (0.25%) benign FPR. These are benchmark observations, not population guarantees. Confidence intervals and paired differences must be computed and inserted into the manuscript before submission; they must not be invented from the rounded pooled values.

### Current family-level evidence

All-attempt detection by family is heterogeneous. R4/R5 respectively report delayed 0/40 and 0/40, indirect 0/40 and 0/40, persistent 1/40 and 1/40, privilege 35/40 and 37/40, tool-chain 16/40 and 17/40, and transaction 16/40 and 16/40. This supports a boundary claim: the observed signal is concentrated in attacks that alter tool transitions or frequencies, while semantically harmful attacks with ordinary-looking trajectories remain difficult to detect.

### Claims that must remain exploratory or conditional

- The no-cumulative ablation (95%/100% in the exploratory run) is post-confirmation and separately recalibrated; it is not a confirmed method result.
- The no-cross-session comparison changes detection by about 0.4 percentage points and does not establish a cross-session advantage.
- AgentShield was run in a separate execution environment and is descriptive context, not a paired same-trajectory baseline.
- No cross-model generalization claim is allowed without another model and a preregistered protocol.

## 3. Research questions

Use explicit research questions to make the contribution falsifiable:

- **RQ1 — Detectability:** How much attack behavior can be detected from tool-call trajectories alone under a leakage-controlled protocol?
- **RQ2 — Cross-session value:** Does retaining state across sessions improve detection over a strict session-only detector on attacks designed to require cross-session context?
- **RQ3 — Family heterogeneity:** Which attack families produce detectable transition/frequency deviations, and which remain trajectory-indistinguishable from benign behavior?
- **RQ4 — Confounding:** How much of the apparent recall survives exact total-tool-call-count matching and other controls for execution length and tool identity?
- **RQ5 — Baseline sufficiency:** Does TO-CF outperform simple frequency/Markov/feature-space anomaly detectors under identical splits, calibration, and denominators?

## 4. Contributions to claim in the paper

After the additional baseline and cross-session experiments are complete, the paper may claim:

1. A reproducible, leakage-controlled evaluation protocol separating development, frozen calibration, and independent confirmation runs.
2. A precise operationalization of trajectory-only detection (TO-CF), including ordered-session state, cumulative scoring, group-maximum statistics, and validation-maximum thresholding.
3. A family-level measurement of where trajectory signals work and where they fail, including length-matched results and failure cases.
4. A controlled test of whether cross-session state contributes beyond session-local transitions.
5. An empirical comparison against same-trajectory statistical and sequence baselines, with low-FPR confidence intervals and paired uncertainty.

Do not claim a new universal detector, production security guarantee, or cross-model robustness unless separately demonstrated.

## 5. Manuscript section map

The complete manuscript should be reorganized as follows:

1. **Introduction.** State the security problem, the measurement gap, the leakage concern, and the five RQs. End with the boundary-study thesis rather than a superiority claim.
2. **Problem Definition and Threat Model.** Define sessions, groups, benign behavior, attack attempts, successful objectives, tool trajectories, cross-session state, and the attacker's observability/adaptation assumptions.
3. **Related Work.** Cover indirect prompt injection, tool-using agent security, memory/cross-session attacks, behavioral and graph anomaly detection, sequence anomaly detection, agent benchmarks, and calibration/low-FPR evaluation.
4. **Detection Candidates.** Separate Full-BG from TO-CF. Define nodes, normalized tool/parameter tokens, transition/frequency baseline, optional graph/parameter/combination components, cumulative score, and complexity.
5. **Leakage-Controlled Evaluation Protocol.** Give the development/validation/confirmation boundary, R4/R5 manifests, exact threshold rule, ordered sessions, test denominators, and the length-matching procedure.
6. **Baselines and Statistical Analysis.** Specify all same-trajectory baselines, fitting/calibration isolation, bootstrap unit, exact/binomial FPR interval, paired differences, multiplicity handling, and seed/API reproducibility.
7. **Main Results.** Present TO-CF R4/R5 numbers with CIs directly in tables; include family-level results and length-matched results.
8. **Cross-Session Causal Test.** Report R6/R7 only after preregistration. Include session-only, cross-session, state-reset, order-shuffle, and matched controls.
9. **Failure Analysis and Boundary Findings.** Explain zero-detection families, unsuccessful attacks, benign false positives, and examples of trajectory-indistinguishable attacks.
10. **Threats to Validity, Reproducibility, and Ethics.** Cover model/domain scope, synthetic generation, API stochasticity, attack-success labels, adaptive attackers, cost, disclosure, and artifact release.
11. **Conclusion.** State exactly what trajectory monitoring can and cannot establish.

The audit/rebuild checklist should move to supplementary material or an artifact appendix so the main text can spend space on method definitions, baselines, and causal cross-session analysis.

## 6. Required experiment package

### 6.1 Same-trajectory baselines (mandatory)

Run every baseline on the identical groups, ordered calls, feature normalization, train/validation/test split, threshold rule, and denominator definitions used for TO-CF:

- tool-frequency z-score;
- first-order Markov transition likelihood;
- n-gram transition likelihood;
- Mahalanobis distance over trajectory features;
- Isolation Forest;
- One-Class SVM;
- sequence autoencoder (GRU or Transformer, with architecture and early-stopping rule frozen before confirmation);
- session-only TO-CF;
- cross-session TO-CF.

The primary table must include all-attempt detection, successful-objective recall, FPR, exact count-matched recall, 95% CIs, paired differences against TO-CF, and inference cost. AgentShield remains a separate-execution descriptive reference unless a same-environment reproduction becomes available.

### 6.2 R6/R7 cross-session experiment (mandatory for a cross-session title)

Pre-register attack templates in which each individual session is benign-looking but the ordered combination is malicious or anomalous. Include:

- session-only detector;
- cross-session detector;
- explicit state reset;
- session-order shuffle;
- same-tool and same-call-count controls;
- benign multi-session controls with matching inter-session gaps.

Primary estimand:

`Delta TPR = TPR(cross-session) - TPR(session-only)`

Report paired bootstrap CI and per-family results. A positive and practically meaningful interval is required before retaining a strong cross-session contribution. If the interval includes zero or the effect is negligible, revise the title and contribution to a trajectory-only boundary study.

### 6.3 Robustness and reproducibility

- At least three independent confirmation seeds/API trajectory samples, where feasible.
- Frozen model name/version, temperature, system prompt, tool schemas, and retry policy.
- Exact attack-template and operand-generation manifests.
- Human audit of a sample of attack-success and benign labels.
- Adaptive-evasion stress test in which the attacker preserves tool identity and approximate call counts.
- Cost/latency/memory measurements for each detector.

## 7. Mandatory disclosure checklist

The Methods or appendix must answer each item explicitly:

- model provider, exact model identifier/version, API date, temperature, max tokens, system prompt;
- tool domain, schemas, argument normalization, and execution sandbox;
- number of sessions per group and inter-session interval definition;
- training, validation, R4, R5, and future R6/R7 record counts;
- how new wording/operands are generated and separated from development data;
- attack-success definition, automatic checks, and human adjudication protocol;
- reasons for unsuccessful attacks and whether early stopping changes trajectory length;
- benign/attack tool-identity and call-count distributions;
- threshold calibration and proof that no test record is read during calibration;
- bootstrap resampling unit and all multiplicity/interval conventions;
- random seeds, retries, missing API responses, and exclusion rules;
- security/ethics safeguards and non-deployment of attack payloads.

## 8. Literature expansion map (25–40 references)

Build the bibliography from primary papers and official standards, organized into the following slots. Do not add citations merely to increase the count; each paper must be compared in a table by signal, execution setting, cross-session scope, data scale, and FPR reporting.

1. Indirect prompt injection and instruction-conflict attacks (4–6 papers).
2. Tool-using LLM agents, function-calling security, and agent hijacking (4–6).
3. Agent memory, persistent state, and cross-session attacks (3–5).
4. Behavioral anomaly detection for software or cyber-physical systems (3–5).
5. Sequence, Markov, n-gram, and graph anomaly detection (4–6).
6. LLM-agent security benchmarks and evaluation protocols (3–5).
7. Calibration, selective prediction, low-FPR evaluation, and bootstrap inference (3–5).
8. Defenses and monitoring systems, including AgentShield-style work (2–4).

The related-work table should make clear that this paper differs by its leakage-controlled independent confirmation, explicit denominator separation, exact length matching, family-level failure analysis, and controlled cross-session comparison—not merely by renaming a transition detector.

## 9. Double-blind and venue-template gate

Before submission:

- Select the target CCF-C venue and use only its official IEEE/ACM/ LNCS template.
- Remove author names, affiliations, acknowledgments, repository URLs, self-identifying filenames, and metadata from the anonymous version.
- Check PDF metadata, embedded comments, source archive names, and figure paths for identity leakage.
- Confirm page limit, font size, column format, reference style, and supplementary-material rules.
- Ensure every figure remains readable at the venue's two-column width; do not shrink labels to fit.
- Keep method pseudocode, CI tables, and baseline definitions in the main paper; move only audit/rebuild details to supplement.
- Run a final text search for stale phrases such as `Title Suppressed Due to Excessive Length`, `P95` when referring to R4/R5, `production-ready`, and unqualified `BehaviorGraph achieves`.

## 10. Submission decision rule

The manuscript is submission-ready only when all mandatory experiments are complete, all primary tables include uncertainty, the method is named consistently (Full-BG versus TO-CF), and the title matches the measured cross-session effect. If R6/R7 does not establish a reliable gain, submit the measurement-study title and remove superiority language. This rule prevents the paper from making a stronger claim than its own evidence supports.
