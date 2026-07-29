# Paper-level experiment summary

## Aggregate

| Metric | Mean | Sample SD |
|---|---:|---:|
| primary_dr | 0.6792 | 0.0423 |
| primary_fpr | 0.1250 | 0.0000 |
| official_dr | 0.6028 | 0.0024 |
| official_fpr | 0.0000 | 0.0000 |
| paired_attack_delta_primary_minus_official | 0.0764 | 0.0405 |
| paired_benign_delta_primary_minus_official | 0.1250 | 0.0000 |
| length_matched_attack_delta_primary_minus_official | 0.0393 | 0.0224 |
| feedback_success_rate | 0.4722 | 0.0096 |

## Required disclosure

All primary-vs-official comparisons are paired offline evaluations on identical official-AgentShield-augmented stored trajectories. The official source is a pinned GitHub archive (SHA-256 EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E) with a pre-registered local banking interface adapter. Feedback is one-step frozen-detector candidate screening, not an online adaptive policy.

Primary-detector successful-attack misses: 50 rows in `primary_successful_attack_misses.csv`.
