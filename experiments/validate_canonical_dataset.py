"""Audit canonical raw trajectories before any detector evaluation is allowed."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.canonical_protocol import all_episodes


def index_attempt(root: Path) -> dict[str, list[dict[str, Any]]]:
    records = root / "records"
    if not records.is_dir():
        return {}
    indexed: dict[str, list[dict[str, Any]]] = {}
    for path in records.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        indexed.setdefault(record["attack_id"], []).append(record)
    return indexed


def is_complete_ok(records: list[dict[str, Any]], expected_sessions: int) -> bool:
    return (
        len(records) == expected_sessions
        and {item.get("session_index") for item in records} == set(range(expected_sessions))
        and all(item.get("collector_status") == "ok" for item in records)
    )


def audit(base: Path) -> dict[str, Any]:
    attempts = [base] + sorted(path for path in base.glob("recovery_*") if path.is_dir())
    indexed_attempts = [(attempt, index_attempt(attempt)) for attempt in attempts]
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    provenance: dict[str, str] = {}

    for episode in all_episodes():
        chosen: list[dict[str, Any]] | None = None
        chosen_root: Path | None = None
        for attempt, indexed in indexed_attempts:
            candidate = indexed.get(episode.group_id, [])
            if is_complete_ok(candidate, len(episode.sessions)):
                chosen, chosen_root = candidate, attempt
        if chosen is None:
            missing.append(episode.group_id)
            continue
        selected.extend(chosen)
        provenance[episode.group_id] = chosen_root.name if chosen_root else "unknown"

    split_counts = Counter(item["split"] for item in selected if item["label"] == "benign")
    adaptive_counts = Counter(item.get("adaptation_level") for item in selected if item["label"] == "attack")
    attack_type_counts = Counter(item.get("attack_type") for item in selected if item["label"] == "attack")
    expected_sessions = sum(len(item.sessions) for item in all_episodes())
    recovered_groups = sum(origin != base.name for origin in provenance.values())
    valid = (
        not missing
        and len(selected) == expected_sessions
        and split_counts == {"train": 100, "validation": 50, "test": 100}
        and adaptive_counts == {"A0": 80, "A1": 80, "A2": 80, "A3": 80}
        and attack_type_counts == {
            "delayed_trigger": 80, "persistent_memory": 80, "indirect_prompt_injection": 40,
            "tool_chain_exfiltration": 40, "privilege_escalation": 40, "transaction_cover_tracks": 40,
        }
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": str(base), "attempts": [path.name for path in attempts], "valid": valid,
        "expected_session_records": expected_sessions, "selected_session_records": len(selected),
        "selected_groups": len(provenance), "recovered_groups": recovered_groups,
        "benign_split_counts": dict(split_counts), "adaptive_session_counts": dict(adaptive_counts),
        "attack_type_session_counts": dict(attack_type_counts), "missing_groups": missing,
        "group_provenance": provenance,
    }


def main() -> None:
    base = ROOT / "local_results" / "canonical" / "paper_protocol_20260729_1"
    result = audit(base)
    destination = base / "data_audit.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "group_provenance"}, indent=2))
    if not result["valid"]:
        raise SystemExit("Canonical data audit failed; evaluation is blocked.")


if __name__ == "__main__":
    main()
