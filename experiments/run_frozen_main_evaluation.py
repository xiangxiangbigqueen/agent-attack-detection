"""Frozen main evaluation over the audited canonical raw trajectories."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.types import ToolCall
from detection.graph_detector import DetectorConfig, MultiLayerDetector
from experiments.canonical_protocol import all_episodes
from experiments.validate_canonical_dataset import audit, index_attempt, is_complete_ok

BASE = ROOT / "local_results" / "canonical" / "paper_protocol_20260729_1"


def selected_records(base: Path) -> dict[str, list[dict[str, Any]]]:
    result = audit(base)
    if not result["valid"]:
        raise ValueError("Data audit failed; main evaluation is blocked")
    attempts = [base] + sorted(path for path in base.glob("recovery_*") if path.is_dir())
    indexed = [(path, index_attempt(path)) for path in attempts]
    selected: dict[str, list[dict[str, Any]]] = {}
    for episode in all_episodes():
        for _, rows in indexed:
            candidate = rows.get(episode.group_id, [])
            if is_complete_ok(candidate, len(episode.sessions)):
                selected[episode.group_id] = sorted(candidate, key=lambda item: item["session_index"])
    return selected


def calls(record: dict[str, Any]) -> list[ToolCall]:
    return [ToolCall(session_id=f"{record['attack_id']}-s{record['session_index']}", turn_id=index,
                     tool_name=item["name"], parameters=item.get("params", {}), timestamp=float(index))
            for index, item in enumerate(record["tools"])]


def detector_with_baseline(train: list[list[ToolCall]]) -> MultiLayerDetector:
    detector = MultiLayerDetector(DetectorConfig(min_baseline_samples=3, alert_threshold=1e9))
    detector.set_training(True)
    for session in train:
        detector.train_on(session)
    detector.set_training(False)
    return detector


def group_score(records: list[dict[str, Any]], train: list[list[ToolCall]]) -> float:
    detector = detector_with_baseline(train)
    maximum = 0.0
    for session_index, record in enumerate(records):
        if session_index:
            detector.reset_session()
        for call in calls(record):
            result = detector.analyze_call(call)
            maximum = max(maximum, float(result.layer_results["cumulative_score"]))
    return maximum


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z, p = 1.96, successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def main() -> None:
    selected = selected_records(BASE)
    episodes = {episode.group_id: episode for episode in all_episodes()}
    train = [calls(selected[group_id][0]) for group_id, episode in episodes.items() if episode.split == "train"]
    validation_ids = [group_id for group_id, episode in episodes.items() if episode.split == "validation"]
    test_benign_ids = [group_id for group_id, episode in episodes.items() if episode.label == "benign" and episode.split == "test"]
    attack_ids = [group_id for group_id, episode in episodes.items() if episode.label == "attack"]

    validation_scores = {group_id: group_score(selected[group_id], train) for group_id in validation_ids}
    threshold = float(np.quantile(list(validation_scores.values()), 0.95, method="higher"))
    benign_scores = {group_id: group_score(selected[group_id], train) for group_id in test_benign_ids}
    attack_scores = {group_id: group_score(selected[group_id], train) for group_id in attack_ids}
    fp = sum(value >= threshold for value in benign_scores.values())
    tp = sum(value >= threshold for value in attack_scores.values())
    by_type: dict[str, dict[str, Any]] = {}
    for attack_type in sorted({episodes[item].attack_type for item in attack_ids}):
        values = [attack_scores[item] for item in attack_ids if episodes[item].attack_type == attack_type]
        detected = sum(value >= threshold for value in values)
        by_type[attack_type] = {"detected": detected, "total": len(values), "dr": detected / len(values), "wilson_95": wilson(detected, len(values))}
    by_adaptation: dict[str, dict[str, Any]] = {}
    for level in ("A0", "A1", "A2", "A3"):
        values = [attack_scores[item] for item in attack_ids if episodes[item].adaptation_level == level]
        detected = sum(value >= threshold for value in values)
        by_adaptation[level] = {"detected": detected, "total": len(values), "dr": detected / len(values), "wilson_95": wilson(detected, len(values))}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_audit": "data_audit.json", "threshold_protocol": "P95 validation benign group maximum",
        "threshold": threshold, "counts": {"train": len(train), "validation": len(validation_ids), "benign_test": len(test_benign_ids), "attack_groups": len(attack_ids)},
        "metrics": {"dr": tp / len(attack_ids), "dr_wilson_95": wilson(tp, len(attack_ids)), "fpr": fp / len(test_benign_ids), "fpr_wilson_95": wilson(fp, len(test_benign_ids)), "tp": tp, "fp": fp, "by_attack_type": by_type, "by_adaptation": by_adaptation},
        "scores": {"validation": validation_scores, "benign_test": benign_scores, "attack_test": attack_scores},
    }
    destination = BASE / "main_evaluation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
