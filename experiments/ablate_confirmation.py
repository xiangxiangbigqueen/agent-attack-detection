"""Exploratory post-confirmation mechanism ablations on frozen R4/R5 traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection.graph_detector import DetectorConfig, MultiLayerDetector
from experiments.evaluate_confirmation import calls, select_long, select_raw
from experiments.confirmation_protocol import all_episodes

BASE = {"use_structure": False, "use_parameter_rules": False, "use_tool_combination_rules": False, "use_transition_frequency": True, "use_cumulative": True}
VARIANTS = {"TrajectoryOnly": (BASE, True, False), "NoCrossSession": (BASE, False, False), "NoTransitionFrequency": ({**BASE, "use_transition_frequency": False}, True, False), "NoCumulative": (BASE, True, True)}


def score(records, training, kwargs, preserve_cross_session, instant_only):
    def detector():
        value = MultiLayerDetector(DetectorConfig(min_baseline_samples=3, alert_threshold=1e9, **kwargs)); value.set_training(True)
        for session in training: value.train_on(session)
        value.set_training(False); return value
    current = detector(); maximum = 0.0
    for session_index, record in enumerate(records):
        if session_index:
            if preserve_cross_session: current.reset_session()
            else: current = detector()
        for call in calls(record):
            result = current.analyze_call(call)
            maximum = max(maximum, float(result.layer_results["instant_anomaly_score"] if instant_only else result.layer_results["cumulative_score"]))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--raw-root", type=Path, required=True); parser.add_argument("--long-root", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args()
    raw = select_raw(args.raw_root); episodes = {episode.group_id: episode for episode in all_episodes()}
    long_train, long_validation, long_test = select_long(args.long_root, "train", 100), select_long(args.long_root, "validation", 50), select_long(args.long_root, "test", 100)
    training = [calls(raw[key][0]) for key, episode in episodes.items() if episode.split == "train"] + [calls(row) for row in long_train]
    validation = [raw[key] for key, episode in episodes.items() if episode.split == "validation"] + [[row] for row in long_validation]
    benign = [raw[key] for key, episode in episodes.items() if episode.label == "benign" and episode.split == "test"] + [[row] for row in long_test]
    attacks = [raw[key] for key, episode in episodes.items() if episode.label == "attack"]
    output = {"disclosure": "Exploratory post-confirmation ablations; each variant uses its own validation-maximum threshold.", "variants": {}}
    for name, (kwargs, preserve, instant) in VARIANTS.items():
        threshold = float(np.nextafter(max(score(group, training, kwargs, preserve, instant) for group in validation), np.inf))
        benign_scores = [score(group, training, kwargs, preserve, instant) for group in benign]; attack_scores = [score(group, training, kwargs, preserve, instant) for group in attacks]
        output["variants"][name] = {"threshold": threshold, "fpr": sum(value >= threshold for value in benign_scores) / len(benign_scores), "dr": sum(value >= threshold for value in attack_scores) / len(attack_scores), "fp": sum(value >= threshold for value in benign_scores), "tp": sum(value >= threshold for value in attack_scores)}
    destination = args.output or args.raw_root / "confirmation_ablation_exploratory.json"; destination.write_text(json.dumps(output, indent=2), encoding="utf-8"); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
