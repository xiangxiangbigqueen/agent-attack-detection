# Final Review and Revision Log

**Review date:** 2026-08-10
**Input reviewed:** `ars_paper.pdf` supplied by the author
**Final output:** `output/pdf/behaviorgraph_submission_complete.pdf` (complete-length manuscript)

The shorter `output/pdf/behaviorgraph_submission_final.pdf` remains available as a provenance artifact; it is not the primary deliverable.

## Problems found in the supplied PDF

The supplied 11-page PDF contained stale or mismatched figures. In particular, the protocol caption was attached to an attack-taxonomy image; the EWMA caption was attached to an unrelated output-length plot; the primary-results and ablation captions were interchanged; and residual latent-dimension/output-length figures remained in the manuscript. Several pages also showed `Title Suppressed Due to Excessive Length`, large unused areas, and figures too small for reliable reading.

The methods text mixed a full feature-rich BehaviorGraph with the reduced candidate actually evaluated in R4/R5. The threshold was described as P95 in the prose but is `nextafter(max(validation maximum), +inf)` in the canonical evaluator. The paper also needed to distinguish score-protocol evaluation from the detector's default online `get_decision` path.

## Changes completed

1. Created `docs/METHOD_CONSISTENCY_AUDIT.md` with source-line evidence, formulas, candidate switches, operational caveats, and a submission gate.
2. Created `experiments/audit_submission_consistency.py` and `output/audit/submission_consistency.json`. The audit independently recomputes thresholds, group alerts, attack-type counts, successful-objective denominators, misses, and exploratory ablations. `passed=true` for both R4 and R5.
3. Created six data-backed corrected figures in `output/pdf/assets_corrected/`:
   - `figure1_evaluation_protocol`: development/confirmation leakage boundary;
   - `figure2_score_distributions`: attack/benign score distributions and run-specific thresholds;
   - `figure3_primary_confirmation`: all-attempt, successful-recall, and exact count-matched results;
   - `figure4_exploratory_ablation`: explicitly post-confirmation ablations;
   - `figure5_attack_taxonomy`: all-attempt detection by all six attack families;
   - `figure6_separate_baseline`: descriptive AgentShield separate-execution comparison.
4. Created `experiments/build_final_submission_pdf.py` and rebuilt the short provenance PDF from the corrected assets.
5. Rewrote the final paper's title and abstract to identify the confirmed model as TO-CF, state the exact threshold rule and thresholds, remove production-readiness language, and disclose the separate baseline execution.
6. Added a per-attack-family table, exact thresholds, configuration boundary table, length-matching terminology, failure counts, and the R6/R7 requirement for the no-cumulative hypothesis.
7. Created `experiments/build_complete_submission_pdf.py` and rebuilt the complete-length manuscript with the full method, protocol, results, baseline, threats-to-validity, reproducibility, references, and audit appendix sections.
7. Rendered all four final pages to PNG and visually checked that captions stay with their figures, no old figures remain, no plot overlaps text, and the short running header is clean.

## Canonical numbers used in the final paper

| Quantity | R4 | R5 |
| --- | ---: | ---: |
| Threshold | 5.0040541339 | 4.9817326042 |
| Validation alerts | 0/100 | 0/100 |
| Benign test alerts | 0/200 | 1/200 |
| All-attempt attack alerts | 68/240 | 71/240 |
| Successful-objective alerts | 67/160 | 70/160 |
| Exact total-tool-call-count matched alerts | 26/117 | 29/113 |

The aggregate descriptive values are 29.0% all-attempt detection, 42.8% successful-objective recall, and 0.25% benign FPR. The 95%-100% no-cumulative result remains exploratory and is not promoted to a confirmed method claim.

## Final submission gate

- [x] Every figure corresponds to its caption and source data.
- [x] Full-BG and TO-CF are explicitly separated.
- [x] R4/R5 use the single canonical `nextafter(max(validation maximum), +inf)` rule.
- [x] P95 is not used to describe the confirmation threshold.
- [x] Score-protocol alerts are not conflated with `get_decision` alerts.
- [x] Length matching is described as exact total-tool-call-count matching.
- [x] AgentShield is labelled separate execution and descriptive only.
- [x] No-cumulative is labelled post-confirmation exploratory.
- [x] Final PDF has no stale legacy plots or excessive-length running header.
- [x] Author affiliation/contact remains the only submission metadata placeholder.

## Remaining scientific work before stronger claims

The final PDF is internally consistent, but stronger method claims still require preregistered R6/R7 no-cumulative replication, independent Full-BG confirmation if the full framework is to be claimed, broader model/domain coverage, adaptive attacks, and deployment-cost measurements.
