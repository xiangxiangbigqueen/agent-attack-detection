"""Exact tool-call-length matched evaluation over frozen group-level scores."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import BASE, selected_records

SEED = 20260729
REPETITIONS = 1000


def main() -> None:
    scores = json.loads((BASE / "main_evaluation.json").read_text(encoding="utf-8"))["scores"]
    episodes = {item.group_id: item for item in all_episodes()}
    records = selected_records(BASE)
    threshold = json.loads((BASE / "main_evaluation.json").read_text(encoding="utf-8"))["threshold"]
    rows = []
    for group_id, episode in episodes.items():
        if episode.split != "test":
            continue
        score = scores["benign_test"].get(group_id, scores["attack_test"].get(group_id))
        rows.append({"group_id": group_id, "attack": episode.label == "attack", "length": sum(item["n_calls"] for item in records[group_id]), "score": score})
    benign, attack = defaultdict(list), defaultdict(list)
    for row in rows:
        (attack if row["attack"] else benign)[row["length"]].append(row)
    common = sorted(set(benign) & set(attack))
    support = {str(length): {"benign": len(benign[length]), "attack": len(attack[length]), "matched_per_class": min(len(benign[length]), len(attack[length]))} for length in common}
    rng = random.Random(SEED)
    outcomes = []
    for _ in range(REPETITIONS):
        matched_benign, matched_attack = [], []
        for length in common:
            count = min(len(benign[length]), len(attack[length]))
            matched_benign.extend(rng.sample(benign[length], count))
            matched_attack.extend(rng.sample(attack[length], count))
        fp = sum(row["score"] >= threshold for row in matched_benign)
        tp = sum(row["score"] >= threshold for row in matched_attack)
        precision = tp / (tp + fp) if tp + fp else 0.0
        dr, fpr = tp / len(matched_attack), fp / len(matched_benign)
        outcomes.append({"dr": dr, "fpr": fpr, "f1": 2 * precision * dr / (precision + dr) if precision + dr else 0.0})
    def interval(name):
        values = sorted(item[name] for item in outcomes)
        return {"mean": sum(values) / len(values), "ci95": [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]}
    result = {"seed": SEED, "repetitions": REPETITIONS, "threshold": threshold, "common_length_support": support, "excluded_lengths": {"benign_only": sorted(set(benign)-set(attack)), "attack_only": sorted(set(attack)-set(benign))}, "effective_groups_per_class": sum(value["matched_per_class"] for value in support.values()), "metrics": {key: interval(key) for key in ("dr", "fpr", "f1")}}
    path = BASE / "length_matched_evaluation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
