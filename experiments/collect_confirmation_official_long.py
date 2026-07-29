"""Official AgentShield controls for the R4/R5 wording-held-out long benign set."""
from __future__ import annotations

import argparse
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
from experiments.collect_confirmation_official_agentshield import ConfirmationOfficialMonitor
from experiments.collect_official_agentshield import OFFICIAL_SOURCE, configured_tools, plant_official_honeytokens
from experiments.confirmation_protocol import PROTOCOL_VERSION, long_benign_task

COUNTS = {"train": 100, "validation": 50, "test": 100}
WRITE_LOCK = threading.Lock()


def collect_one(output: Path, split: str, index: int, schemas: list[dict], dispatch: dict, key: str) -> str:
    steps = 5 + index % 4; group = f"confirm-long-{split}-t{steps}-v{index:03d}"
    if (output / "records" / f"{record_id(split, group, 0)}.json").exists(): return f"skip_complete {group}"
    env = BankingEnvironment(); plant_official_honeytokens(env); monitor = ConfirmationOfficialMonitor()
    agent = FunctionCallingAgent(AgentConfig(api_key=key, model="deepseek-chat", temperature=0.1, max_turns=8, tool_schemas=schemas, tool_dispatch=dispatch, tool_observer=monitor.observe), env)
    calls = agent.run(long_benign_task(index + 2000, steps))
    row = make_record(split=split, label="benign", attack_type=None, group_id=group, session_index=0, task=long_benign_task(index + 2000, steps), calls=calls, collector_status=agent.last_run_status, error_type=agent.last_error_type, protocol_version=PROTOCOL_VERSION)
    row["official_agentshield_detections"] = monitor.detections(); row["official_agentshield_alert"] = bool(row["official_agentshield_detections"])
    with WRITE_LOCK: write_record(output, row)
    return f"complete {group}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--workers", type=int, default=6); args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("DEEPSEEK_API_KEY is required")
    sys.path.insert(0, str(OFFICIAL_SOURCE)); schemas, dispatch = configured_tools()
    work = [(split, index) for split, count in COUNTS.items() for index in range(count)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_one, args.output, split, index, schemas, dispatch, key) for split, index in work]
        for future in as_completed(futures): print(future.result(), flush=True)


if __name__ == "__main__": main()
