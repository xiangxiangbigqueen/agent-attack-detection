"""Bootstrap confidence intervals for the official AgentShield evaluation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    values.sort()
    return values[round((len(values) - 1) * fraction)]


def interval(values: list[bool], repetitions: int, rng: random.Random) -> dict:
    samples = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)]
    return {"rate": sum(values) / len(values), "total": len(values), "bootstrap_95_ci": [percentile(samples, 0.025), percentile(samples, 0.975)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    result = json.loads(args.evaluation.read_text(encoding="utf-8"))
    predictions = result["variants"]["Full"]["predictions"]
    attack_ids = [key for key in predictions if key.startswith("attack-")]
    benign_ids = [key for key in predictions if key.startswith("benign-test-") or key.startswith("long-test-")]
    rng = random.Random(args.seed)
    by_type = {}
    for attack_type, metric in result["full_by_attack_type"].items():
        ids = [key for key in attack_ids if f"-{attack_type}-" in key]
        by_type[attack_type] = interval([predictions[key] for key in ids], args.repetitions, rng)
    output = {
        "evaluation": str(args.evaluation),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "attack_detection": interval([predictions[key] for key in attack_ids], args.repetitions, rng),
        "benign_alert_rate": interval([predictions[key] for key in benign_ids], args.repetitions, rng),
        "by_attack_type": by_type,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("attack_detection", "benign_alert_rate")}, indent=2))


if __name__ == "__main__":
    main()
