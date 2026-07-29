"""Official AgentShield collection on the wording-held-out R4/R5 protocol."""
from __future__ import annotations

import argparse
import copy
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
from experiments.collect_official_agentshield import ARCHIVE_SHA256, OFFICIAL_SOURCE, configured_tools, plant_official_honeytokens
from experiments.confirmation_protocol import PROTOCOL_VERSION, all_episodes

WRITE_LOCK = threading.Lock()


class ConfirmationOfficialMonitor:
    """Unmodified official layers with a predeclared v2 banking-schema mapping."""
    def __init__(self) -> None:
        from agentshield.defenses.honeytokens import HoneytokenMonitor
        from agentshield.defenses.honeytools import HoneytoolDetector
        from agentshield.defenses.parameter_validator import ParameterValidator
        rules = {
            "send_email": {"to": {"type": "allowlist", "description": "recipient must be a pre-existing ordinary contact", "allowed_values": ["morgan@partner.example", "riley@vendor.example", "jordan@partner.example", "taylor@vendor.example"]}},
            "transfer_money": {"to_account": {"type": "allowlist", "description": "recipient must be a local account", "allowed_values": ["checking", "savings"]}},
        }
        self.detectors = {"honeytools": HoneytoolDetector(), "honeytokens": HoneytokenMonitor(), "parameter_validator": ParameterValidator(rules=rules)}
        self.counter = 0

    def observe(self, call) -> None:
        from agentdojo.functions_runtime import FunctionCall
        self.counter += 1
        tool_call = FunctionCall(function=call.tool_name, args=call.parameters, id=f"confirmation-{self.counter}")
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        for detector in self.detectors.values():
            detector.query("", None, messages=[message])

    def detections(self) -> list[dict]:
        rows = []
        for detector in self.detectors.values():
            rows.extend(copy.deepcopy(detector.detections))
        return rows


def collect_group(output: Path, episode, schemas: list[dict], dispatch: dict, key: str) -> str:
    expected = [output / "records" / f"{record_id(episode.split, episode.group_id, index)}.json" for index in range(len(episode.sessions))]
    if all(path.exists() for path in expected): return f"skip_complete {episode.group_id}"
    if any(path.exists() for path in expected): return f"skip_partial {episode.group_id}"
    env = BankingEnvironment(); plant_official_honeytokens(env)
    for index, task in enumerate(episode.sessions):
        monitor = ConfirmationOfficialMonitor()
        agent = FunctionCallingAgent(AgentConfig(api_key=key, model="deepseek-chat", temperature=0.1, max_turns=8, tool_schemas=schemas, tool_dispatch=dispatch, tool_observer=monitor.observe), env)
        if episode.injection and index == 0: agent.inject_content(*episode.injection)
        calls = agent.run(task)
        row = make_record(split=episode.split, label=episode.label, attack_type=episode.attack_type, group_id=episode.group_id, session_index=index, task=task, calls=calls, collector_status=agent.last_run_status, error_type=agent.last_error_type, adaptation_level=episode.adaptation_level, protocol_version=PROTOCOL_VERSION)
        row["official_agentshield_detections"] = monitor.detections(); row["official_agentshield_alert"] = bool(row["official_agentshield_detections"])
        with WRITE_LOCK: write_record(output, row)
    return f"complete {episode.group_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--workers", type=int, default=12); args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("DEEPSEEK_API_KEY is required")
    sys.path.insert(0, str(OFFICIAL_SOURCE)); schemas, dispatch = configured_tools(); episodes = all_episodes()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "official_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"official_repository": "https://github.com/Yassin-H-Rassul/AgentShield", "source_archive_sha256": ARCHIVE_SHA256, "protocol_version": PROTOCOL_VERSION, "model": "deepseek-chat", "parameter_adapter": "confirmation-v2 predeclared local banking mapping", "session_records_expected": sum(len(item.sessions) for item in episodes)}, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_group, args.output, episode, schemas, dispatch, key) for episode in episodes]
        for future in as_completed(futures): print(future.result(), flush=True)


if __name__ == "__main__": main()
