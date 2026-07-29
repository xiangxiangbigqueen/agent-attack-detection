"""Generate final R4/R5 confirmation tables, bootstrap intervals, and figures."""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_results" / "canonical" / "confirmation_paper_artifacts"


def read(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def ci(values: list[bool], seed: int) -> dict:
    rng = random.Random(seed); samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(10_000))
    return {"rate": sum(values) / len(values), "total": len(values), "bootstrap_95_ci": [samples[250], samples[9749]]}
def mean_sd(values: list[float]) -> dict: return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    rows = []; failures = []
    for offset, run in enumerate(("R4", "R5")):
        primary_root = ROOT / "local_results" / "canonical" / f"confirmation_{run}"
        official_root = ROOT / "local_results" / "canonical" / f"confirmation_official_agentshield_{run}"
        evaluation = read(primary_root / "confirmation_evaluation.json")
        analysis = read(primary_root / "confirmation_analysis.json")
        official = read(official_root / "official_agentshield_confirmation_evaluation.json")
        predictions = official["variants"]["Full"]["predictions"]
        official_attack = [predictions[key] for key in predictions if key.startswith("confirm-attack-")]
        official_benign = [predictions[key] for key in predictions if key.startswith("confirm-benign-test-") or key.startswith("confirm-long-test-")]
        rows.append({
            "run": run,
            "primary": {"all_attempts": analysis["all_attempts"], "successful_attacks": analysis["successful_attacks"], "benign": analysis["benign"], "length_matched": analysis["length_matched"]},
            "official_agentshield_separate_execution": {"all_attempts": ci(official_attack, 20260730 + offset), "successful_attacks": official["successful_attack_subset"], "benign": ci(official_benign, 20260740 + offset)},
        })
        for failure in analysis["successful_attack_misses"]:
            failures.append({"run": run, **failure})
    aggregate = {
        "primary_all_attempt_dr": mean_sd([row["primary"]["all_attempts"]["rate"] for row in rows]),
        "primary_successful_attack_recall": mean_sd([row["primary"]["successful_attacks"]["rate"] for row in rows]),
        "primary_fpr": mean_sd([row["primary"]["benign"]["rate"] for row in rows]),
        "official_all_attempt_dr_separate_execution": mean_sd([row["official_agentshield_separate_execution"]["all_attempts"]["rate"] for row in rows]),
        "official_successful_attack_recall_separate_execution": mean_sd([row["official_agentshield_separate_execution"]["successful_attacks"]["rate"] for row in rows]),
        "official_fpr_separate_execution": mean_sd([row["official_agentshield_separate_execution"]["benign"]["rate"] for row in rows]),
    }
    output = {"disclosure": "Primary detector and official AgentShield were executed in separate environments because the official baseline requires honeytools and honeytokens. These figures are descriptive separate-execution comparisons, not paired significance tests.", "runs": rows, "aggregate": aggregate, "successful_attack_miss_count": len(failures)}
    (OUT / "confirmation_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (OUT / "primary_successful_attack_misses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "group_id", "attack_type", "adaptation_level", "tool_calls", "score", "threshold"]); writer.writeheader(); writer.writerows(failures)
    labels = [row["run"] for row in rows]; x = range(len(labels)); width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for axis, key, title in zip(axes, ("successful_attacks", "benign"), ("Successful-attack recall", "False-positive rate")):
        primary = [row["primary"][key]["rate"] for row in rows]; official = [row["official_agentshield_separate_execution"][key]["rate"] for row in rows]
        primary_ci = [row["primary"][key]["bootstrap_95_ci"] for row in rows]
        official_ci = [row["official_agentshield_separate_execution"][key].get("bootstrap_95_ci", [official[i], official[i]]) for i, row in enumerate(rows)]
        p_err = [[value - bounds[0] for value, bounds in zip(primary, primary_ci)], [bounds[1] - value for value, bounds in zip(primary, primary_ci)]]
        o_err = [[value - bounds[0] for value, bounds in zip(official, official_ci)], [bounds[1] - value for value, bounds in zip(official, official_ci)]]
        axis.bar([i - width / 2 for i in x], primary, width, yerr=p_err, capsize=3, label="Trajectory-only candidate")
        axis.bar([i + width / 2 for i in x], official, width, yerr=o_err, capsize=3, label="Official AgentShield")
        axis.set_ylim(0, 0.03 if key == "benign" else 1.05); axis.set_xticks(list(x), labels); axis.set_ylabel("Rate"); axis.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "figures" / "confirmation_recall_fpr_ci.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"output": str(OUT), "aggregate": aggregate, "misses": len(failures)}, indent=2))


if __name__ == "__main__": main()
