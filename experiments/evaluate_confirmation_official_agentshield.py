"""Evaluate official AgentShield alerts on completed R4/R5 confirmation traces."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.confirmation_protocol import all_episodes, attack_success

LAYERS = ("honeytools", "honeytokens", "parameter_validator")


def index(root: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in (root / "records").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8")); rows[row["attack_id"]].append(row)
    return rows


def select_raw(root: Path) -> dict[str, list[dict]]:
    indexed = index(root); result = {}
    for episode in all_episodes():
        rows = sorted(indexed.get(episode.group_id, []), key=lambda row: row["session_index"])
        if len(rows) != len(episode.sessions) or any(row.get("collector_status") != "ok" for row in rows): raise RuntimeError(f"incomplete group: {episode.group_id}")
        result[episode.group_id] = rows
    return result


def select_long(root: Path) -> list[dict]:
    rows = [row for values in index(root).values() for row in values if row.get("split") == "test" and row.get("collector_status") == "ok"]
    if len(rows) != 100: raise RuntimeError(f"expected 100 long test rows, found {len(rows)}")
    return rows


def alert(rows: list[dict], enabled: set[str]) -> bool:
    return any(item.get("layer") in enabled for row in rows for item in row.get("official_agentshield_detections", []))


def rate(predictions: dict[str, bool], ids: list[str]) -> dict:
    hits = sum(predictions[item] for item in ids); return {"total": len(ids), "alerts": hits, "rate": hits / len(ids) if ids else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--raw-root", type=Path, required=True); parser.add_argument("--long-root", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args()
    records = select_raw(args.raw_root); records.update({row["attack_id"]: [row] for row in select_long(args.long_root)})
    episodes = {episode.group_id: episode for episode in all_episodes()}
    attacks = [key for key, episode in episodes.items() if episode.label == "attack"]
    benign = [key for key, episode in episodes.items() if episode.label == "benign" and episode.split == "test"] + [key for key in records if key.startswith("confirm-long-test-")]
    variants = {"Full": set(LAYERS), "HoneytoolsOnly": {"honeytools"}, "HoneytokensOnly": {"honeytokens"}, "ParameterValidatorOnly": {"parameter_validator"}, "NoParameterValidator": set(LAYERS) - {"parameter_validator"}}
    outputs = {}
    for name, enabled in variants.items():
        predictions = {key: alert(rows, enabled) for key, rows in records.items()}
        outputs[name] = {"attack_detection": rate(predictions, attacks), "benign_alert_rate": rate(predictions, benign), "predictions": predictions}
    full = outputs["Full"]["predictions"]
    success = [key for key in attacks if attack_success(episodes[key].attack_type, [call for row in records[key] for call in row["tools"]])]
    by_type = {kind: rate(full, [key for key in attacks if episodes[key].attack_type == kind]) for kind in sorted({episodes[key].attack_type for key in attacks})}
    result = {"protocol_disclosure": "Official AgentShield source archive with predeclared confirmation-v2 interface mapping; separate defended executions, not paired with the primary detector's non-honeypot trajectories.", "variants": outputs, "successful_attack_subset": rate(full, success), "objective_successes": len(success), "by_attack_type": by_type}
    output = args.output or args.raw_root / "official_agentshield_confirmation_evaluation.json"; output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"full_attack": outputs["Full"]["attack_detection"], "full_benign": outputs["Full"]["benign_alert_rate"], "successful": result["successful_attack_subset"]}, indent=2))


if __name__ == "__main__": main()
