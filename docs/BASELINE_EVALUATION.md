# Same-trajectory baseline evaluation

`experiments/run_baseline_suite.py` evaluates statistical detectors on the
stored R4/R5 confirmation traces.  It does not call an API and does not alter
the frozen candidate, the manuscript PDF, or the confirmation records.

## Protocol

For each run, the script uses exactly the groups selected by
`experiments/evaluate_confirmation.py`:

* training: 100 short benign groups (one session each) plus 100 long benign
  calibration groups;
* validation: 50 short benign groups plus 50 long benign groups;
* test: 100 short benign groups plus 100 long benign groups;
* attack test: 240 attack groups, preserving all sessions in each attack group.

The detector is fitted only on training sessions.  A group score is the maximum
score over its constituent sessions.  Each method's threshold is
`nextafter(max(validation group score), +inf)`, so no test record selects a
threshold and the validation alert count is exactly zero unless a numerical
serialization changes the score.  Attack success labels are taken from the
pre-registered `attack_success` rules; they are not inferred from detector
scores.

## Implemented methods

1. **Tool-frequency z-score.** Per-session tool counts are normalized by the
   number of calls.  The score is the largest absolute standardized deviation
   from benign training frequencies.  This is a frequency-only baseline and
   deliberately does not inspect parameters or natural-language content.
2. **First-order Markov transition surprise.** Tool-name transitions are fit
   with additive smoothing (`alpha=0.5`).  The score is the largest negative
   log2 transition probability in a session.
3. **Regularized Mahalanobis.** The feature vector concatenates tool and
   transition proportions, including explicit unknown-tool/unknown-transition
   buckets.  The covariance is ridge-regularized before pseudoinversion so the
   detector is defined for short, collinear banking traces.
4. **Isolation Forest (optional).** The same feature vector is passed to
   scikit-learn's `IsolationForest` (256 trees, `random_state=20260729`).  If
   scikit-learn is unavailable, the output records `status: skipped` and the
   reason; no substitute number is fabricated.

## Reported outputs

Each `output/baselines/{r4,r5}_baseline_suite.json` contains the protocol
hashes, git revision, threshold, validation/benign/attack metrics, Wilson 95%
intervals, successful-attack recall, per-attack-family results, exact
tool-call-length-matched strata, and raw group scores. `summary.json` is a
compact machine-readable index.

The length-matched analysis is deterministic: within each exact total
tool-call-count stratum, it chooses the lexicographically first equal number of
attack and benign groups without replacement.  This is a secondary diagnostic,
not a replacement for the pre-registered all-group result.

## Interpretation and limitations

These baselines are trajectory-only comparators, not claimed reproductions of
AgentShield or any other published implementation.  They share the same model,
tool sandbox, records, splits, and denominators as TO-CF, making paired
comparisons meaningful.  They do not establish cross-model generalization,
semantic detection, or causal cross-session benefit.  The validation-maximum
threshold controls this benchmark's observed validation alerts; it is not a
finite-sample guarantee of a population FPR.  Because the confirmation data are
finite and attack families are heterogeneous, intervals and family-level zeros
must be reported rather than hidden behind aggregate averages.

## Rebuild

From the repository root:

```powershell
python experiments/run_baseline_suite.py --runs R4 R5
```

To run only baselines available in the current Python environment:

```powershell
python experiments/run_baseline_suite.py --methods tool_frequency_zscore markov_transition mahalanobis
```
