"""Exact tool-call-length-matched paired detector evaluation on stored traces."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compare_primary_official_agentshield import paired_summary


def group_lengths(roots: list[Path]) -> tuple[dict[str, int], dict[str, int]]:
    attacks, benign = defaultdict(int), defaultdict(int)
    for root in roots:
        for path in (root / "records").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("collector_status") != "ok":
                continue
            target = attacks if row.get("label") == "attack" else benign if row.get("label") == "benign" and row.get("split") == "test" else None
            if target is not None:
                target[row["attack_id"]] += len(row["tools"])
    return dict(attacks), dict(benign)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    official = json.loads(args.official.read_text(encoding="utf-8"))
    threshold = primary["threshold"]
    primary_attack = {key: value >= threshold for key, value in primary["scores"]["attack_by_group"].items()}
    primary_benign = {key: value >= threshold for key, value in primary["scores"]["benign_by_group"].items()}
    official_all = official["variants"]["Full"]["predictions"]
    official_attack = {key: official_all[key] for key in primary_attack}
    official_benign = {key: official_all[key] for key in primary_benign}
    attack_length, benign_length = group_lengths([args.raw_root, args.long_root])
    attacks_by_length, benign_by_length = defaultdict(list), defaultdict(list)
    for key in primary_attack:
        attacks_by_length[attack_length[key]].append(key)
    for key in primary_benign:
        benign_by_length[benign_length[key]].append(key)
    chosen_attack, chosen_benign, strata = [], [], {}
    for length in sorted(set(attacks_by_length) & set(benign_by_length)):
        a, b = sorted(attacks_by_length[length]), sorted(benign_by_length[length])
        count = min(len(a), len(b))
        if count:
            chosen_attack.extend(a[:count]); chosen_benign.extend(b[:count])
            strata[str(length)] = {"attack_available": len(a), "benign_available": len(b), "matched_per_class": count}
    import random
    rng = random.Random(args.seed)
    result = {
        "raw_root": str(args.raw_root),
        "long_root": str(args.long_root),
        "matching_protocol": "Exact 1:1 matching without replacement on group-total tool-call count; lexicographic deterministic selection within each length stratum.",
        "threshold": threshold,
        "matched_attack_groups": len(chosen_attack),
        "matched_benign_groups": len(chosen_benign),
        "strata": strata,
        "attack_detection": paired_summary(primary_attack, official_attack, chosen_attack, args.repetitions, rng),
        "benign_alert_rate": paired_summary(primary_benign, official_benign, chosen_benign, args.repetitions, rng),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("matched_attack_groups", "matched_benign_groups", "attack_detection", "benign_alert_rate")}, indent=2))


if __name__ == "__main__":
    main()
