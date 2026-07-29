"""Create consolidated paper-level tables, figures, and failure-case records."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.evaluate_attack_outcomes import successful
from experiments.run_frozen_main_evaluation import selected_records

RUNS = ("R1", "R2", "R3")
OUT = ROOT / "local_results" / "canonical" / "paper_level_artifacts"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    runs = []
    ablations: dict[str, list[dict]] = {}
    feedback = []
    failure_rows = []
    episodes = {episode.group_id: episode for episode in all_episodes()}
    for run in RUNS:
        root = ROOT / "local_results" / "canonical" / f"official_agentshield_{run}"
        primary = read(root / "primary_detector_long_calibrated_evaluation.json")
        official = read(root / "official_agentshield_evaluation_long.json")
        primary_boot = read(root / "primary_detector_long_calibrated_bootstrap.json")
        official_boot = read(root / "official_agentshield_bootstrap.json")
        paired = read(root / "paired_primary_vs_official_agentshield.json")
        matched = read(root / "length_matched_paired_evaluation.json")
        ablation = read(root / "primary_detector_long_calibrated_ablation.json")
        fb = read(ROOT / "local_results" / "canonical" / f"official_agentshield_feedback_{run}" / "official_agentshield_feedback_evaluation.json")
        official_full = official["variants"]["Full"]
        runs.append({
            "run": run,
            "primary": {"dr": primary["dr"], "fpr": primary["fpr"], "dr_bootstrap_95_ci": primary_boot["dr_ci95"], "fpr_bootstrap_95_ci": primary_boot["fpr_ci95"]},
            "official_agentshield": {"dr": official_full["attack_detection"]["dr"], "fpr": official_full["benign_alert_rate"], "dr_bootstrap_95_ci": official_boot["attack_detection"]["bootstrap_95_ci"], "fpr_bootstrap_95_ci": official_boot["benign_alert_rate"]["bootstrap_95_ci"]},
            "paired": paired,
            "length_matched": matched,
        })
        feedback.append({"run": run, **{key: fb[key] for key in ("selected_alert_rate", "selected_objective_successes", "selected_objective_success_rate", "selected_successful_attack_detection_rate")}})
        for name, values in ablation["variants"].items():
            ablations.setdefault(name, []).append({"run": run, "dr": values["dr"], "fpr": values["fpr"]})
        data = selected_records(root)
        for group, score in primary["scores"]["attack_by_group"].items():
            episode = episodes[group]
            calls = [call for record in data[group] for call in record["tools"]]
            if successful(episode.attack_type, calls) and score < primary["threshold"]:
                failure_rows.append({"run": run, "group_id": group, "attack_type": episode.attack_type, "adaptation_level": episode.adaptation_level, "tool_calls": len(calls), "primary_score": score, "threshold": primary["threshold"], "official_agentshield_alert": official_full["predictions"][group]})

    aggregate = {
        "primary_dr": mean_sd([item["primary"]["dr"] for item in runs]),
        "primary_fpr": mean_sd([item["primary"]["fpr"] for item in runs]),
        "official_dr": mean_sd([item["official_agentshield"]["dr"] for item in runs]),
        "official_fpr": mean_sd([item["official_agentshield"]["fpr"] for item in runs]),
        "paired_attack_delta_primary_minus_official": mean_sd([item["paired"]["attack_detection"]["primary_minus_official"] for item in runs]),
        "paired_benign_delta_primary_minus_official": mean_sd([item["paired"]["benign_alert_rate"]["primary_minus_official"] for item in runs]),
        "length_matched_attack_delta_primary_minus_official": mean_sd([item["length_matched"]["attack_detection"]["primary_minus_official"] for item in runs]),
        "feedback_success_rate": mean_sd([item["selected_objective_success_rate"] for item in feedback]),
    }
    ablation_summary = {name: {"dr": mean_sd([item["dr"] for item in values]), "fpr": mean_sd([item["fpr"] for item in values])} for name, values in ablations.items()}
    output = {
        "protocol_disclosure": "All primary-vs-official comparisons are paired offline evaluations on identical official-AgentShield-augmented stored trajectories. The official source is a pinned GitHub archive (SHA-256 EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E) with a pre-registered local banking interface adapter. Feedback is one-step frozen-detector candidate screening, not an online adaptive policy.",
        "runs": runs,
        "aggregate": aggregate,
        "primary_ablation": ablation_summary,
        "feedback": feedback,
        "failure_case_count": len(failure_rows),
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (OUT / "primary_successful_attack_misses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "group_id", "attack_type", "adaptation_level", "tool_calls", "primary_score", "threshold", "official_agentshield_alert"])
        writer.writeheader(); writer.writerows(failure_rows)

    labels = [item["run"] for item in runs]
    x = list(range(len(labels)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("dr", "fpr"), ("Attack detection rate", "False-positive rate")):
        primary_values = [item["primary"][metric] for item in runs]
        official_values = [item["official_agentshield"][metric] for item in runs]
        primary_ci_key = f"{metric}_bootstrap_95_ci"
        official_ci_key = f"{metric}_bootstrap_95_ci"
        primary_err = [[value - item["primary"][primary_ci_key][0] for item, value in zip(runs, primary_values)], [item["primary"][primary_ci_key][1] - value for item, value in zip(runs, primary_values)]]
        official_err = [[value - item["official_agentshield"][official_ci_key][0] for item, value in zip(runs, official_values)], [item["official_agentshield"][official_ci_key][1] - value for item, value in zip(runs, official_values)]]
        axis.bar([value - width / 2 for value in x], primary_values, width, yerr=primary_err, capsize=3, label="Primary detector")
        axis.bar([value + width / 2 for value in x], official_values, width, yerr=official_err, capsize=3, label="Official AgentShield")
        axis.set_xticks(x, labels); axis.set_ylim(0, 1.05); axis.set_title(title); axis.set_ylabel("Rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "paired_detection_and_fpr_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    names = list(ablation_summary)
    drs = [ablation_summary[name]["dr"]["mean"] for name in names]
    errs = [ablation_summary[name]["dr"]["sample_sd"] for name in names]
    fig, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    axis.bar(range(len(names)), drs, yerr=errs, capsize=3)
    axis.set_xticks(range(len(names)), names, rotation=24, ha="right")
    axis.set_ylim(0, 1.05); axis.set_ylabel("Attack detection rate"); axis.set_title("Primary-detector component ablation (mean ± sample SD, R1–R3)")
    fig.savefig(figures / "primary_detector_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Paper-level experiment summary", "", "## Aggregate", "", "| Metric | Mean | Sample SD |", "|---|---:|---:|"]
    for name, values in aggregate.items():
        lines.append(f"| {name} | {values['mean']:.4f} | {values['sample_sd']:.4f} |")
    lines.extend(["", "## Required disclosure", "", output["protocol_disclosure"], "", f"Primary-detector successful-attack misses: {len(failure_rows)} rows in `primary_successful_attack_misses.csv`."])
    (OUT / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "aggregate": aggregate, "failure_case_count": len(failure_rows)}, indent=2))


if __name__ == "__main__":
    main()
