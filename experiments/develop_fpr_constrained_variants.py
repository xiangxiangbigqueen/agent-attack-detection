"""Development-only diagnostic for FPR-constrained detector candidates.

R1--R3 test values are intentionally labelled diagnostic: this script may help
select a protocol, but it must never be cited as independent confirmation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection.graph_detector import DetectorConfig, MultiLayerDetector
from experiments.canonical_protocol import all_episodes
from experiments.evaluate_attack_outcomes import successful
from experiments.run_frozen_main_evaluation import calls, selected_records

RUNS = {
    "R1": (ROOT / "local_results" / "canonical" / "paper_protocol_20260729_1", ROOT / "local_results" / "canonical" / "paper_protocol_20260729_1" / "long_benign_calibration"),
    "R2": (ROOT / "local_results" / "canonical" / "replication_R2" / "raw", ROOT / "local_results" / "canonical" / "replication_R2" / "long_benign_calibration"),
    "R3": (ROOT / "local_results" / "canonical" / "replication_R3" / "raw", ROOT / "local_results" / "canonical" / "replication_R3" / "long_benign_calibration"),
}
VARIANTS = {
    "full": {},
    "no_structure": {"use_structure": False},
    "no_structure_no_parameter": {"use_structure": False, "use_parameter_rules": False},
    "trajectory_only": {"use_structure": False, "use_parameter_rules": False, "use_tool_combination_rules": False},
}


def long_rows(root: Path, split: str) -> list[dict]:
    rows = []
    for path in (root / "records").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("collector_status") == "ok" and row.get("split") == split:
            rows.append(row)
    return rows


def score_group(records: list[dict], train: list, config: DetectorConfig) -> float:
    detector = MultiLayerDetector(config)
    detector.set_training(True)
    for session in train:
        detector.train_on(session)
    detector.set_training(False)
    maximum = 0.0
    for index, record in enumerate(records):
        if index:
            detector.reset_session()
        for call in calls(record):
            maximum = max(maximum, float(detector.analyze_call(call).layer_results["cumulative_score"]))
    return maximum


def assess(raw_root: Path, long_root: Path, kwargs: dict) -> dict:
    data = selected_records(raw_root)
    episodes = {episode.group_id: episode for episode in all_episodes()}
    config = DetectorConfig(min_baseline_samples=3, alert_threshold=1e9, **kwargs)
    train = [calls(data[key][0]) for key, episode in episodes.items() if episode.split == "train"]
    train += [calls(row) for row in long_rows(long_root, "train")]
    validation = [data[key] for key, episode in episodes.items() if episode.split == "validation"]
    validation += [[row] for row in long_rows(long_root, "validation")]
    benign = [data[key] for key, episode in episodes.items() if episode.label == "benign" and episode.split == "test"]
    benign += [[row] for row in long_rows(long_root, "test")]
    attacks = {key: data[key] for key, episode in episodes.items() if episode.label == "attack"}
    validation_scores = [score_group(group, train, config) for group in validation]
    # A predeclared zero-observed-validation-FP calibration.  It is deliberately
    # strict; R4/R5 will determine whether the FPR cap generalizes.
    threshold = float(np.nextafter(max(validation_scores), np.inf))
    benign_scores = [score_group(group, train, config) for group in benign]
    attack_scores = {key: score_group(group, train, config) for key, group in attacks.items()}
    success = []
    for key, group in attacks.items():
        episode = episodes[key]
        trace = [call for record in group for call in record["tools"]]
        if successful(episode.attack_type, trace):
            success.append(key)
    return {
        "threshold": threshold,
        "validation_groups": len(validation_scores),
        "validation_alerts": sum(value >= threshold for value in validation_scores),
        "benign_test": {"total": len(benign_scores), "alerts": sum(value >= threshold for value in benign_scores)},
        "attack_test": {"total": len(attack_scores), "alerts": sum(value >= threshold for value in attack_scores.values())},
        "successful_attack_test": {"total": len(success), "alerts": sum(attack_scores[key] >= threshold for key in success)},
    }


def main() -> None:
    result = {
        "status": "development_only_not_independent_confirmation",
        "selection_rule": "R4/R5 candidate: trajectory_only, threshold = next representable float above the maximum R4/R5 validation score; no instant rule alerts.",
        "calibration_rule": "Observed validation alerts must be zero; no test score can select a threshold.",
        "variants": {},
    }
    for name, kwargs in VARIANTS.items():
        result["variants"][name] = {run: assess(raw, long, kwargs) for run, (raw, long) in RUNS.items()}
    output = ROOT / "local_results" / "canonical" / "development_fpr_constrained_diagnostic.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
