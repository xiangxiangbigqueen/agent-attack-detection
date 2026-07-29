"""Summarize R1/R2/R3 official AgentShield evaluations and layer ablations."""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
roots = [ROOT / "local_results" / "canonical" / f"official_agentshield_{run}" for run in ("R1", "R2", "R3")]


def main() -> None:
    runs = []
    for root in roots:
        result = json.loads((root / "official_agentshield_evaluation_long.json").read_text(encoding="utf-8"))
        full = result["variants"]["Full"]
        runs.append({"root": str(root), "dr": full["attack_detection"]["dr"], "fpr": full["benign_alert_rate"], "successful_attack_dr": result["successful_attack_subset"]["dr"], "variants": {name: {"dr": item["attack_detection"]["dr"], "fpr": item["benign_alert_rate"]} for name, item in result["variants"].items()}})
    metric = lambda key: [run[key] for run in runs]
    output = {"disclosure": "Official AgentShield source, fixed archive SHA-256 EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E; pre-registered local banking interface adapter.", "runs": runs, "aggregate": {"dr_mean": statistics.mean(metric("dr")), "dr_sample_sd": statistics.stdev(metric("dr")), "fpr_mean": statistics.mean(metric("fpr")), "fpr_sample_sd": statistics.stdev(metric("fpr")), "successful_attack_dr_mean": statistics.mean(metric("successful_attack_dr")), "successful_attack_dr_sample_sd": statistics.stdev(metric("successful_attack_dr"))}}
    destination = ROOT / "local_results" / "canonical" / "official_agentshield_replication_summary.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
