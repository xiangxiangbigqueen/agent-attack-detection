"""Paired comparison of the primary detector and official AgentShield on identical traces."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def exact_mcnemar(primary_only: int, official_only: int) -> float:
    """Two-sided exact McNemar p-value for the discordant pairs."""
    discordant = primary_only + official_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(primary_only, official_only) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], fraction: float) -> float:
    values.sort()
    return values[round((len(values) - 1) * fraction)]


def paired_summary(primary: dict[str, bool], official: dict[str, bool], ids: list[str], repetitions: int, rng: random.Random) -> dict:
    if set(ids) - set(primary) or set(ids) - set(official):
        raise RuntimeError("prediction identifiers differ between the two evaluators")
    pairs = [(primary[item], official[item]) for item in ids]
    both = sum(a and b for a, b in pairs)
    primary_only = sum(a and not b for a, b in pairs)
    official_only = sum(not a and b for a, b in pairs)
    neither = len(pairs) - both - primary_only - official_only
    differences = []
    for _ in range(repetitions):
        sample = [rng.choice(pairs) for _ in pairs]
        differences.append(sum(a for a, _ in sample) / len(sample) - sum(b for _, b in sample) / len(sample))
    return {
        "total": len(pairs),
        "primary_rate": sum(a for a, _ in pairs) / len(pairs),
        "official_agentshield_rate": sum(b for _, b in pairs) / len(pairs),
        "primary_minus_official": (primary_only - official_only) / len(pairs),
        "paired_bootstrap_95_ci": [percentile(differences, 0.025), percentile(differences, 0.975)],
        "contingency": {"both_alert": both, "primary_only": primary_only, "official_only": official_only, "neither_alert": neither},
        "mcnemar_exact_two_sided_p": exact_mcnemar(primary_only, official_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    official = json.loads(args.official.read_text(encoding="utf-8"))
    threshold = primary["threshold"]
    primary_attacks = {key: value >= threshold for key, value in primary["scores"]["attack_by_group"].items()}
    primary_benign = {key: value >= threshold for key, value in primary["scores"]["benign_by_group"].items()}
    official_predictions = official["variants"]["Full"]["predictions"]
    official_attacks = {key: official_predictions[key] for key in primary_attacks}
    official_benign = {key: official_predictions[key] for key in primary_benign}
    rng = random.Random(args.seed)

    by_type = {}
    for attack_type, metric in primary["by_attack_type"].items():
        ids = [key for key in primary_attacks if f"-{attack_type}-" in key]
        by_type[attack_type] = paired_summary(primary_attacks, official_attacks, ids, args.repetitions, rng)
    result = {
        "primary_evaluation": str(args.primary),
        "official_evaluation": str(args.official),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "protocol_disclosure": "Both methods are evaluated offline on the identical official-AgentShield augmented trajectories and identical benign controls. McNemar tests are paired; per-attack-type p-values are exploratory and unadjusted.",
        "attack_detection": paired_summary(primary_attacks, official_attacks, list(primary_attacks), args.repetitions, rng),
        "benign_alert_rate": paired_summary(primary_benign, official_benign, list(primary_benign), args.repetitions, rng),
        "by_attack_type": by_type,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("attack_detection", "benign_alert_rate")}, indent=2))


if __name__ == "__main__":
    main()
