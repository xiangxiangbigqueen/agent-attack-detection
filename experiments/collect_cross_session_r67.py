"""Collect the frozen R6/R7 two-session API traces.

This command requires ``DEEPSEEK_API_KEY`` and never fabricates records.  It
is resumable: an existing record is not overwritten.  The resulting records
are condition-neutral; ``run_cross_session_r67.py`` replays them under the
four pre-registered detector-state policies.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env import BankingEnvironment
from agent.function_agent import AgentConfig, FunctionCallingAgent
from experiments.collect_canonical_api import make_record, record_id, write_record, atomic_json_write, timestamp
from experiments.cross_session_r67_protocol import PROTOCOL_VERSION, all_episodes


def collect(output: Path, selected: set[str] | None = None) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("blocked: DEEPSEEK_API_KEY is required; no R6/R7 records were created")
    episodes = [episode for episode in all_episodes() if selected is None or episode.group_id in selected]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol_manifest.json"
    if not manifest.exists():
        atomic_json_write(manifest, {
            "protocol_version": PROTOCOL_VERSION,
            "created_at_utc": timestamp(),
            "model": "deepseek-chat", "temperature": 0.1, "max_turns": 8,
            "condition_replay": "offline_detector_replay",
            "episode_groups": len(episodes),
            "session_records_expected": sum(len(item.sessions) for item in episodes),
            "environment_policy": "same BankingEnvironment within each two-session attack group",
        })
    for index, episode in enumerate(episodes, 1):
        env = BankingEnvironment()
        for session_index, task in enumerate(episode.sessions):
            destination = output / "records" / f"{record_id(episode.split, episode.group_id, session_index)}.json"
            if destination.exists():
                continue
            agent = FunctionCallingAgent(AgentConfig(
                api_key=key, model="deepseek-chat", temperature=0.1, max_turns=8,
            ), env)
            calls = agent.run(task)
            record = make_record(
                split=episode.split, label=episode.label, attack_type=episode.attack_type,
                group_id=episode.group_id, session_index=session_index, task=task,
                calls=calls, collector_status=agent.last_run_status,
                error_type=agent.last_error_type, protocol_version=PROTOCOL_VERSION,
            )
            record["objective"] = episode.objective
            record["environment_policy"] = "same_environment_within_group"
            record["environment_reset"] = False
            write_record(output, record)
        print(f"[{index}/{len(episodes)}] {episode.group_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local_results" / "canonical" / "cross_session_R6R7")
    parser.add_argument("--group", action="append", dest="groups", help="restrict collection to one or more group IDs")
    args = parser.parse_args()
    collect(args.output, set(args.groups) if args.groups else None)
    print(f"Collector output: {args.output}")
