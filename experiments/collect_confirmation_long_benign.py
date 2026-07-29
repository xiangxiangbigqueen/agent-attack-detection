"""Collect wording-held-out long benign controls for R4/R5 confirmation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.env import BankingEnvironment
from agent.function_agent import AgentConfig, FunctionCallingAgent
from experiments.collect_canonical_api import make_record, record_id, write_record
from experiments.confirmation_protocol import PROTOCOL_VERSION, long_benign_task

COUNTS = {"train": 100, "validation": 50, "test": 100}
WRITE_LOCK = threading.Lock()


def collect_one(output: Path, split: str, index: int, api_key: str) -> str:
    steps = 5 + index % 4
    group = f"confirm-long-{split}-t{steps}-v{index:03d}"
    destination = output / "records" / f"{record_id(split, group, 0)}.json"
    if destination.exists():
        return f"skip_complete {group}"
    task = long_benign_task(index + 2000, steps)
    agent = FunctionCallingAgent(AgentConfig(api_key=api_key, model="deepseek-chat", temperature=0.1, max_turns=8), BankingEnvironment())
    calls = agent.run(task)
    row = make_record(
        split=split, label="benign", attack_type=None, group_id=group, session_index=0,
        task=task, calls=calls, collector_status=agent.last_run_status,
        error_type=agent.last_error_type, protocol_version=PROTOCOL_VERSION,
    )
    with WRITE_LOCK:
        write_record(output, row)
    return f"complete {group}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retry-failures-from", type=Path)
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if args.retry_failures_from:
        failed = []
        for path in (args.retry_failures_from / "records").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("collector_status") == "api_error":
                failed.append(row)
        if not failed:
            raise RuntimeError("no api_error records available for recovery")
        def retry(row: dict) -> str:
            destination = args.output / "records" / f"{record_id(row['split'], row['attack_id'], row['session_index'])}.json"
            if destination.exists():
                return f"skip_complete {row['attack_id']}"
            agent = FunctionCallingAgent(AgentConfig(api_key=key, model="deepseek-chat", temperature=0.1, max_turns=8), BankingEnvironment())
            calls = agent.run(row["task"])
            recovered = make_record(split=row["split"], label="benign", attack_type=None, group_id=row["attack_id"], session_index=row["session_index"], task=row["task"], calls=calls, collector_status=agent.last_run_status, error_type=agent.last_error_type, protocol_version=PROTOCOL_VERSION)
            with WRITE_LOCK:
                write_record(args.output, recovered)
            return f"complete {row['attack_id']}"
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(retry, row) for row in failed]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        return
    work = [(split, index) for split, count in COUNTS.items() for index in range(count)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_one, args.output, split, index, key) for split, index in work]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
