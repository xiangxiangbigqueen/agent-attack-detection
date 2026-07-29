"""Collect the canonical protocol with the official AgentShield layers enabled.

The official source is imported unchanged.  This adapter only exposes the
official honeytools to our banking agent, plants the official honeytokens in
the local sandbox, maps equivalent local parameter names to a pre-registered
official ParameterValidator configuration, and records each official alert.
"""
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
from agent.function_agent import AgentConfig, FunctionCallingAgent, TOOL_DISPATCH, TOOL_SCHEMAS
from experiments.canonical_protocol import all_episodes
from experiments.collect_canonical_api import make_record, record_id, write_record

OFFICIAL_SOURCE = ROOT / "external_baselines" / "AgentShield_official_source" / "AgentShield-main"
ARCHIVE_SHA256 = "EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E"
WRITE_LOCK = threading.Lock()


class OfficialAgentShieldMonitor:
    def __init__(self) -> None:
        from agentshield.defenses.honeytokens import HoneytokenMonitor
        from agentshield.defenses.honeytools import HoneytoolDetector
        from agentshield.defenses.parameter_validator import ParameterValidator

        # Fixed before collection.  Names map the official validator's
        # recipient semantics onto this banking sandbox's public tool schema.
        rules = {
            "send_email": {"to": {"type": "allowlist", "description": "recipient must be an existing contact", "allowed_values": ["alice@company.com", "bob@partner.com", "carol@company.com", "david@external.com"]}},
            "transfer_money": {"to_account": {"type": "allowlist", "description": "recipient must be a local account", "allowed_values": ["checking", "savings"]}},
        }
        self.detectors = {
            "honeytools": HoneytoolDetector(),
            "honeytokens": HoneytokenMonitor(),
            "parameter_validator": ParameterValidator(rules=rules),
        }
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0
        for detector in self.detectors.values():
            detector.reset()

    def observe(self, call) -> None:
        from agentdojo.functions_runtime import FunctionCall

        self._counter += 1
        tool_call = FunctionCall(function=call.tool_name, args=call.parameters, id=f"canonical-{self._counter}")
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        for detector in self.detectors.values():
            detector.query("", None, messages=[message])

    def detections(self) -> list[dict]:
        rows = []
        for detector in self.detectors.values():
            rows.extend(copy.deepcopy(detector.detections))
        return rows


def configured_tools() -> tuple[list[dict], dict]:
    from agentshield.defenses.honeytools import HONEYTOOLS

    schemas = copy.deepcopy(TOOL_SCHEMAS)
    dispatch = dict(TOOL_DISPATCH)
    for honeytool in HONEYTOOLS:
        schemas.append({"type": "function", "function": {"name": honeytool.name, "description": honeytool.description, "parameters": honeytool.parameters.model_json_schema()}})
        dispatch[honeytool.name] = lambda _env, **_kwargs: {"status": "error", "reason": "service temporarily unavailable"}
    return schemas, dispatch


def plant_official_honeytokens(env: BankingEnvironment) -> None:
    from agentshield.defenses.honeytokens import HONEYTOKEN_SNIPPET

    env.documents[0].content += HONEYTOKEN_SNIPPET
    env.emails[0].body += HONEYTOKEN_SNIPPET


def collect_episode(episode, output: Path, schemas: list[dict], dispatch: dict, api_key: str) -> str:
    expected = [output / "records" / f"{record_id(episode.split, episode.group_id, index)}.json" for index in range(len(episode.sessions))]
    if all(path.exists() for path in expected):
        return f"skip_complete {episode.group_id}"
    if any(path.exists() for path in expected):
        return f"skip_partial {episode.group_id}"
    env = BankingEnvironment()
    plant_official_honeytokens(env)
    for index, task in enumerate(episode.sessions):
        monitor = OfficialAgentShieldMonitor()
        monitor.reset()
        agent = FunctionCallingAgent(AgentConfig(api_key=api_key, model="deepseek-chat", temperature=0.1, max_turns=8, tool_schemas=schemas, tool_dispatch=dispatch, tool_observer=monitor.observe), env)
        if episode.injection and index == 0:
            agent.inject_content(*episode.injection)
        calls = agent.run(task)
        row = make_record(split=episode.split, label=episode.label, attack_type=episode.attack_type, group_id=episode.group_id, session_index=index, task=task, calls=calls, collector_status=agent.last_run_status, error_type=agent.last_error_type, adaptation_level=episode.adaptation_level)
        row["official_agentshield_detections"] = monitor.detections()
        row["official_agentshield_alert"] = bool(row["official_agentshield_detections"])
        # The JSON files are distinct per session; only index appends need a
        # lock to keep the audit trail line-atomic under group parallelism.
        with WRITE_LOCK:
            write_record(output, row)
    return f"complete {episode.group_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "local_results" / "canonical" / "official_agentshield_R1")
    parser.add_argument("--source", type=Path, default=OFFICIAL_SOURCE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    sys.path.insert(0, str(args.source))
    schemas, dispatch = configured_tools()
    episodes = all_episodes()[:args.limit] if args.limit else all_episodes()
    manifest = {"official_repository": "https://github.com/Yassin-H-Rassul/AgentShield", "source_archive_sha256": ARCHIVE_SHA256, "model": "deepseek-chat", "protocol": "canonical protocol with official AgentShield layers; adapted parameter-name mapping fixed before collection"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "official_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_episode, episode, args.output, schemas, dispatch, os.environ["DEEPSEEK_API_KEY"]) for episode in episodes]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
