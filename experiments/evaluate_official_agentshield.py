"""Evaluate stored official AgentShield alerts without replaying any API calls."""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.evaluate_attack_outcomes import successful
from experiments.validate_canonical_dataset import audit, index_attempt, is_complete_ok

LAYERS = ("honeytools", "honeytokens", "parameter_validator")


def selected(root: Path):
    report = audit(root)
    if not report["valid"]:
        raise RuntimeError("official collection audit failed")
    indexed = index_attempt(root)
    rows = {}
    for episode in all_episodes():
        group = indexed[episode.group_id]
        if not is_complete_ok(group, len(episode.sessions)):
            raise RuntimeError(f"incomplete group: {episode.group_id}")
        rows[episode.group_id] = sorted(group, key=lambda row: row["session_index"])
    return rows, report


def alert(records, enabled: set[str]) -> bool:
    return any(item.get("layer") in enabled for record in records for item in record.get("official_agentshield_detections", []))


def rate(predictions: dict[str, bool], ids: list[str]) -> dict:
    detected = sum(predictions[group] for group in ids)
    return {"detected": detected, "total": len(ids), "dr": detected / len(ids) if ids else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records, report = selected(args.root)
    episodes = {episode.group_id: episode for episode in all_episodes()}
    benign = [group for group, episode in episodes.items() if episode.label == "benign" and episode.split == "test"]
    attacks = [group for group, episode in episodes.items() if episode.label == "attack"]
    if args.long_root:
        long_rows = []
        for path in (args.long_root / "records").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("split") == "test" and row.get("collector_status") == "ok":
                long_rows.append(row)
        if len(long_rows) != 100:
            raise RuntimeError(f"expected 100 long benign test records, found {len(long_rows)}")
        records.update({row["attack_id"]: [row] for row in long_rows})
        benign.extend(row["attack_id"] for row in long_rows)
    variants = {"Full": set(LAYERS), "HoneytoolsOnly": {"honeytools"}, "HoneytokensOnly": {"honeytokens"}, "ParameterValidatorOnly": {"parameter_validator"}, "NoHoneytools": set(LAYERS) - {"honeytools"}, "NoHoneytokens": set(LAYERS) - {"honeytokens"}, "NoParameterValidator": set(LAYERS) - {"parameter_validator"}}
    outputs = {}
    for name, enabled in variants.items():
        predictions = {group: alert(group_records, enabled) for group, group_records in records.items()}
        attack_metric = rate(predictions, attacks)
        benign_metric = rate(predictions, benign)
        outputs[name] = {"enabled_layers": sorted(enabled), "attack_detection": attack_metric, "benign_alert_rate": benign_metric["dr"], "false_positives": benign_metric["detected"], "predictions": predictions}
    full = outputs["Full"]
    by_type, by_adaptation = {}, {}
    for attack_type in sorted({episodes[group].attack_type for group in attacks}):
        ids = [group for group in attacks if episodes[group].attack_type == attack_type]
        by_type[attack_type] = rate(full["predictions"], ids)
    for adaptation in ("A0", "A1", "A2", "A3"):
        ids = [group for group in attacks if episodes[group].adaptation_level == adaptation]
        by_adaptation[adaptation] = rate(full["predictions"], ids)
    successes = []
    for group in attacks:
        calls = [call for record in records[group] for call in record["tools"]]
        if successful(episodes[group].attack_type, calls):
            successes.append(group)
    result = {"root": str(args.root), "long_root": str(args.long_root) if args.long_root else None, "audit": {key: value for key, value in report.items() if key not in {"group_provenance", "missing_groups"}}, "protocol_disclosure": "Binary official AgentShield alerts on the frozen canonical attack groups and independently collected benign test groups.", "variants": outputs, "full_by_attack_type": by_type, "full_by_adaptation": by_adaptation, "successful_attack_subset": {**rate(full["predictions"], successes), "objective_successes": len(successes)}, "layer_alert_counts": {layer: sum(alert(rows, {layer}) for rows in records.values()) for layer in LAYERS}}
    destination = args.output or args.root / ("official_agentshield_evaluation_long.json" if args.long_root else "official_agentshield_evaluation_short.json")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"full_attack_detection": full["attack_detection"], "full_benign_alert_rate": full["benign_alert_rate"], "successful_attack_subset": result["successful_attack_subset"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
