"""Concurrent, resumable collector for a fresh R4/R5 confirmation protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.env import BankingEnvironment
from agent.function_agent import AgentConfig, FunctionCallingAgent
from experiments.collect_canonical_api import make_record, record_id, write_record
from experiments.confirmation_protocol import PROTOCOL_VERSION, all_episodes

WRITE_LOCK = threading.Lock()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_group(output: Path, episode, api_key: str) -> str:
    expected = [output / "records" / f"{record_id(episode.split, episode.group_id, index)}.json" for index in range(len(episode.sessions))]
    if all(path.exists() for path in expected):
        return f"skip_complete {episode.group_id}"
    if any(path.exists() for path in expected):
        return f"skip_partial {episode.group_id}"
    environment = BankingEnvironment()
    for index, task in enumerate(episode.sessions):
        agent = FunctionCallingAgent(AgentConfig(api_key=api_key, model="deepseek-chat", temperature=0.1, max_turns=8), environment)
        if episode.injection and index == 0:
            agent.inject_content(*episode.injection)
        calls = agent.run(task)
        row = make_record(
            split=episode.split, label=episode.label, attack_type=episode.attack_type,
            group_id=episode.group_id, session_index=index, task=task, calls=calls,
            collector_status=agent.last_run_status, error_type=agent.last_error_type,
            adaptation_level=episode.adaptation_level, protocol_version=PROTOCOL_VERSION,
        )
        with WRITE_LOCK:
            write_record(output, row)
    return f"complete {episode.group_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    episodes = all_episodes()
    manifest = args.output / "protocol_manifest.json"
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "created_at_utc": utc(), "model": "deepseek-chat", "temperature": 0.1,
            "max_turns": 8, "workers": args.workers, "episode_groups": len(episodes),
            "session_records_expected": sum(len(item.sessions) for item in episodes),
            "role": "independent_confirmation", "sandbox_only": True,
        }, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_group, args.output, episode, api_key) for episode in episodes]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
