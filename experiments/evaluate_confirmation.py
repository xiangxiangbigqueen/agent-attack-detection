"""Frozen R4/R5 evaluation for the pre-registered trajectory-only candidate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.types import ToolCall
from detection.graph_detector import DetectorConfig, MultiLayerDetector
from experiments.confirmation_protocol import PROTOCOL_VERSION, all_episodes, attack_success

CANDIDATE = {
    "use_structure": False,
    "use_parameter_rules": False,
    "use_tool_combination_rules": False,
    "use_transition_frequency": True,
    "use_cumulative": True,
}


def index(root: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in (root / "records").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["attack_id"]].append(row)
    return rows


def select_raw(root: Path) -> dict[str, list[dict]]:
    indexed = index(root)
    selected = {}
    for episode in all_episodes():
        rows = sorted(indexed.get(episode.group_id, []), key=lambda row: row["session_index"])
        if len(rows) != len(episode.sessions) or [row["session_index"] for row in rows] != list(range(len(rows))) or any(row.get("collector_status") != "ok" for row in rows):
            raise RuntimeError(f"incomplete or failed group: {episode.group_id}")
        selected[episode.group_id] = rows
    return selected


def select_long(root: Path, split: str, expected: int) -> list[dict]:
    selected: dict[str, dict] = {}
    for attempt in [root] + sorted(path for path in root.glob("recovery_*") if path.is_dir()):
        for group, values in index(attempt).items():
            rows = [row for row in values if row.get("split") == split and row.get("collector_status") == "ok"]
            if len(rows) == 1:
                selected[group] = rows[0]
    rows = list(selected.values())
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} long benign {split} records, found {len(rows)}")
    return rows


def calls(record: dict) -> list[ToolCall]:
    return [ToolCall(session_id=f"{record['attack_id']}-s{record['session_index']}", turn_id=index, tool_name=item["name"], parameters=item.get("params", {}), timestamp=float(index)) for index, item in enumerate(record["tools"])]


def score_group(records: list[dict], training: list[list[ToolCall]]) -> float:
    detector = MultiLayerDetector(DetectorConfig(min_baseline_samples=3, alert_threshold=1e9, **CANDIDATE))
    detector.set_training(True)
    for session in training:
        detector.train_on(session)
    detector.set_training(False)
    maximum = 0.0
    for session_index, record in enumerate(records):
        if session_index:
            detector.reset_session()
        for call in calls(record):
            maximum = max(maximum, float(detector.analyze_call(call).layer_results["cumulative_score"]))
    return maximum


def metric(ids: list[str], scores: dict[str, float], threshold: float) -> dict:
    alerted = sum(scores[item] >= threshold for item in ids)
    return {"total": len(ids), "alerts": alerted, "rate": alerted / len(ids) if ids else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = select_raw(args.raw_root)
    episodes = {episode.group_id: episode for episode in all_episodes()}
    long_train = select_long(args.long_root, "train", 100)
    long_validation = select_long(args.long_root, "validation", 50)
    long_test = select_long(args.long_root, "test", 100)
    training = [calls(raw[key][0]) for key, episode in episodes.items() if episode.split == "train"] + [calls(row) for row in long_train]
    validation = [raw[key] for key, episode in episodes.items() if episode.split == "validation"] + [[row] for row in long_validation]
    validation_scores = {f"validation-{index:03d}": score_group(group, training) for index, group in enumerate(validation)}
    threshold = float(np.nextafter(max(validation_scores.values()), np.inf))
    benign = {key: raw[key] for key, episode in episodes.items() if episode.label == "benign" and episode.split == "test"}
    benign.update({row["attack_id"]: [row] for row in long_test})
    attacks = {key: raw[key] for key, episode in episodes.items() if episode.label == "attack"}
    benign_scores = {key: score_group(group, training) for key, group in benign.items()}
    attack_scores = {key: score_group(group, training) for key, group in attacks.items()}
    successful = [key for key, episode in episodes.items() if episode.label == "attack" and attack_success(episode.attack_type, [call for record in attacks[key] for call in record["tools"]])]
    by_type = {}
    for kind in sorted({episode.attack_type for episode in episodes.values() if episode.attack_type}):
        ids = [key for key, episode in episodes.items() if episode.attack_type == kind]
        by_type[kind] = metric(ids, attack_scores, threshold)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": "trajectory_only", "candidate_config": CANDIDATE,
        "threshold_rule": "next representable float above maximum of 100 independent short+long benign validation group scores",
        "threshold": threshold, "validation": metric(list(validation_scores), validation_scores, threshold),
        "benign_test": metric(list(benign_scores), benign_scores, threshold),
        "attack_test": metric(list(attack_scores), attack_scores, threshold),
        "successful_attack_test": metric(successful, attack_scores, threshold),
        "by_attack_type": by_type,
        "scores": {"validation": validation_scores, "benign": benign_scores, "attack": attack_scores},
    }
    output = args.output or args.raw_root / "confirmation_evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("validation", "benign_test", "attack_test", "successful_attack_test")}, indent=2))


if __name__ == "__main__":
    main()
