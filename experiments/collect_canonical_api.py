"""Atomic, resumable raw-trajectory collector for the canonical experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_index(index: Path, record_path: Path) -> None:
    with index.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"record": record_path.name, "written_at_utc": timestamp()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record_id(split: str, group_id: str, session_index: int) -> str:
    return f"{split}__{group_id}__s{session_index:02d}"


def make_record(*, split: str, label: str, attack_type: str | None, group_id: str,
                session_index: int, task: str, calls: Iterable[Any],
                collector_status: str | None = None, error_type: str | None = None,
                adaptation_level: str | None = None, protocol_version: str = "2026-07-29.1") -> dict[str, Any]:
    calls = list(calls)
    return {
        "schema_version": 1,
        "protocol_version": protocol_version,
        "collected_at_utc": timestamp(),
        "split": split,
        "label": label,
        "attack_type": attack_type,
        "adaptation_level": adaptation_level,
        "attack_id": group_id,
        "session_index": session_index,
        "user_id": f"user-{hashlib.sha256(group_id.encode()).hexdigest()[:10]}",
        "tenant_id": "sandbox-01",
        "task": task,
        "tools": [{"name": call.tool_name, "params": call.parameters} for call in calls],
        "n_calls": len(calls),
        "collector_status": collector_status or ("ok" if calls else "empty_tool_trace"),
        "error_type": error_type,
    }


def write_record(root: Path, record: dict[str, Any]) -> bool:
    rid = record_id(record["split"], record["attack_id"], record["session_index"])
    destination = root / "records" / f"{rid}.json"
    if destination.exists():
        return False
    atomic_json_write(destination, record)
    append_index(root / "index.jsonl", destination)
    return True


def collect_smoke(root: Path) -> None:
    """Offline test: validates atomic writes and resume semantics without API use."""
    class Call:
        tool_name = "get_balance"
        parameters = {"account": "checking"}
    record = make_record(split="smoke", label="benign", attack_type=None, group_id="smoke-0",
                         session_index=0, task="smoke", calls=[Call()])
    if not write_record(root, record):
        raise RuntimeError("Smoke record already exists; use a fresh output directory")
    if write_record(root, record):
        raise RuntimeError("Resume protection failed")
    stored = json.loads(next((root / "records").glob("*.json")).read_text(encoding="utf-8"))
    assert stored["collector_status"] == "ok" and stored["n_calls"] == 1


def collect_live(root: Path, selected_group_ids: set[str] | None = None,
                 throttle_seconds: float = 0.0, recovery_of: Path | None = None) -> None:
    """Collect the frozen protocol; existing records are never overwritten."""
    from agent.env import BankingEnvironment
    from agent.function_agent import AgentConfig, FunctionCallingAgent
    from experiments.canonical_protocol import PROTOCOL_VERSION, all_episodes

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    episodes = [item for item in all_episodes()
                if selected_group_ids is None or item.group_id in selected_group_ids]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": timestamp(),
        "model": "deepseek-chat", "temperature": 0.1, "max_turns": 8,
        "episode_groups": len(episodes),
        "session_records_expected": sum(len(item.sessions) for item in episodes),
        "sandbox_only": True,
        "recovery_of": str(recovery_of) if recovery_of else None,
    }
    manifest_path = root / "protocol_manifest.json"
    if not manifest_path.exists():
        atomic_json_write(manifest_path, manifest)

    for episode_index, episode in enumerate(episodes, 1):
        expected = [root / "records" / f"{record_id(episode.split, episode.group_id, idx)}.json"
                    for idx in range(len(episode.sessions))]
        if all(path.exists() for path in expected):
            continue
        env = BankingEnvironment()
        for session_index, task in enumerate(episode.sessions):
            agent = FunctionCallingAgent(AgentConfig(
                api_key=api_key, model="deepseek-chat", temperature=0.1, max_turns=8,
            ), env)
            if episode.injection is not None and session_index == 0:
                agent.inject_content(*episode.injection)
            calls = agent.run(task)
            write_record(root, make_record(
                split=episode.split, label=episode.label, attack_type=episode.attack_type,
                group_id=episode.group_id, session_index=session_index, task=task, calls=calls,
                collector_status=agent.last_run_status, error_type=agent.last_error_type,
                adaptation_level=episode.adaptation_level, protocol_version=PROTOCOL_VERSION,
            ))
        print(f"[{episode_index}/{len(episodes)}] {episode.group_id}", flush=True)
        if throttle_seconds:
            time.sleep(throttle_seconds)


def failed_group_ids(source: Path) -> set[str]:
    """Return groups with a recorded transport failure, never deleting evidence."""
    records = source / "records"
    if not records.is_dir():
        raise ValueError(f"No records directory: {records}")
    failures = set()
    for path in records.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("collector_status") == "api_error":
            failures.add(record["attack_id"])
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "local_results" / "canonical" / "paper_protocol_20260729_1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--retry-failures-from", type=Path,
                        help="Source collection root; retry only groups with recorded api_error")
    parser.add_argument("--recovery-label", default="recovery_01")
    parser.add_argument("--throttle-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.smoke_test:
        collect_smoke(args.output)
    elif args.retry_failures_from:
        groups = failed_group_ids(args.retry_failures_from)
        if not groups:
            raise RuntimeError("No api_error groups found for recovery")
        destination = args.output / args.recovery_label
        print(f"Recovering {len(groups)} failed groups into {destination}", flush=True)
        collect_live(destination, selected_group_ids=groups,
                     throttle_seconds=args.throttle_seconds, recovery_of=args.retry_failures_from)
    else:
        collect_live(args.output, throttle_seconds=args.throttle_seconds)
    print(f"Collector output: {args.output}")
