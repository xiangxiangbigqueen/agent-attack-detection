"""Bootstrap, exact-length-matched, and failure-case analysis for frozen R4/R5."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.confirmation_protocol import all_episodes, attack_success


def percentile(values: list[float], fraction: float) -> float:
    values.sort()
    return values[round((len(values) - 1) * fraction)]


def bootstrap(values: list[bool], rng: random.Random, repetitions: int = 10_000) -> dict:
    samples = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)]
    return {"rate": sum(values) / len(values), "total": len(values), "bootstrap_95_ci": [percentile(samples, 0.025), percentile(samples, 0.975)]}


def index_records(roots: list[Path]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for root in roots:
        for path in (root / "records").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("collector_status") == "ok":
                rows[row["attack_id"]].append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    threshold = evaluation["threshold"]
    attacks = evaluation["scores"]["attack"]
    benign = evaluation["scores"]["benign"]
    episodes = {episode.group_id: episode for episode in all_episodes()}
    records = index_records([args.raw_root, args.long_root, *sorted(path for path in args.long_root.glob("recovery_*") if path.is_dir())])
    attack_alerts = {key: value >= threshold for key, value in attacks.items()}
    benign_alerts = {key: value >= threshold for key, value in benign.items()}
    success_ids = [key for key, episode in episodes.items() if episode.label == "attack" and attack_success(episode.attack_type, [call for row in records[key] for call in row["tools"]])]
    lengths = {key: sum(len(row["tools"]) for row in rows) for key, rows in records.items()}
    attack_by_length, benign_by_length = defaultdict(list), defaultdict(list)
    for key in attacks: attack_by_length[lengths[key]].append(key)
    for key in benign: benign_by_length[lengths[key]].append(key)
    matched_attack, matched_benign, strata = [], [], {}
    for length in sorted(set(attack_by_length) & set(benign_by_length)):
        attack_ids, benign_ids = sorted(attack_by_length[length]), sorted(benign_by_length[length])
        count = min(len(attack_ids), len(benign_ids))
        matched_attack.extend(attack_ids[:count]); matched_benign.extend(benign_ids[:count])
        strata[str(length)] = {"attack_available": len(attack_ids), "benign_available": len(benign_ids), "matched_per_class": count}
    rng = random.Random(args.seed)
    by_type = {}
    failures = []
    for kind in sorted({episode.attack_type for episode in episodes.values() if episode.attack_type}):
        ids = [key for key, episode in episodes.items() if episode.attack_type == kind]
        successful_ids = [key for key in ids if key in success_ids]
        by_type[kind] = {"all_attempts": bootstrap([attack_alerts[key] for key in ids], rng), "successful_attacks": bootstrap([attack_alerts[key] for key in successful_ids], rng) if successful_ids else None}
    for key in success_ids:
        if not attack_alerts[key]:
            episode = episodes[key]
            failures.append({"group_id": key, "attack_type": episode.attack_type, "adaptation_level": episode.adaptation_level, "tool_calls": lengths[key], "score": attacks[key], "threshold": threshold})
    output = {
        "evaluation": str(args.evaluation), "repetitions": 10_000, "seed": args.seed,
        "all_attempts": bootstrap(list(attack_alerts.values()), rng),
        "successful_attacks": bootstrap([attack_alerts[key] for key in success_ids], rng),
        "benign": bootstrap(list(benign_alerts.values()), rng),
        "length_matched": {"matching_protocol": "exact 1:1 group-total tool-call matching without replacement", "matched_attack_groups": len(matched_attack), "matched_benign_groups": len(matched_benign), "strata": strata, "attack_detection": bootstrap([attack_alerts[key] for key in matched_attack], rng), "benign_alert_rate": bootstrap([benign_alerts[key] for key in matched_benign], rng)},
        "by_attack_type": by_type, "successful_attack_misses": failures,
    }
    destination = args.output or args.evaluation.with_name("confirmation_analysis.json")
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("all_attempts", "successful_attacks", "benign", "length_matched")}, indent=2))


if __name__ == "__main__":
    main()
