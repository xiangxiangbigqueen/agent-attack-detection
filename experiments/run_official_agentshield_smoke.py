"""Run the unmodified official AgentShield layers through DeepSeek's compatible API.

The only adapter maps AgentDojo's OpenAI-specific ``developer`` wire role to
DeepSeek's supported ``system`` role.  It does not alter any AgentShield
detector, honeytool, honeytoken, parameter rule, prompt, or metric.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "external_baselines" / "AgentShield_official_source" / "AgentShield-main"


def install_deepseek_role_adapter() -> None:
    from agentdojo.agent_pipeline.llms import openai_llm

    original = openai_llm._message_to_openai

    def compatible(message, model_name):
        converted = original(message, model_name)
        if converted["role"] == "developer":
            converted = dict(converted)
            converted["role"] = "system"
        return converted

    openai_llm._message_to_openai = compatible


def run_case(suite, tools, pipeline, detectors, user_task, payload: str | None) -> dict:
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentshield.defenses.pipeline import get_all_detections, prepare_environment, reset_all_detectors

    reset_all_detectors(detectors)
    injections = {vector: payload for vector in suite.get_injection_vector_defaults()} if payload else {}
    env = prepare_environment(suite.load_and_inject_default_environment(injections))
    _, _, _, messages, _ = pipeline.query(user_task.PROMPT, FunctionsRuntime(tools), env)
    calls = [
        {"name": tool_call.function, "params": dict(tool_call.args)}
        for message in messages
        if message["role"] == "assistant" and message.get("tool_calls")
        for tool_call in message["tool_calls"]
    ]
    return {"message_count": len(messages), "tool_calls": calls, "detections": get_all_detections(detectors)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "local_results" / "official_agentshield" / "deepseek_smoke.json")
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    sys.path.insert(0, str(args.source))
    install_deepseek_role_adapter()

    from agentdojo.task_suite.load_suites import get_suite
    from agentshield.attacks.attack_prompts import ALL_ATTACKS
    from agentshield.defenses.pipeline import build_agentshield_pipeline, get_augmented_tools

    suite = get_suite("v1.2.2", "banking")
    tools = get_augmented_tools(suite.tools)
    pipeline, detectors = build_agentshield_pipeline(
        llm="deepseek-chat", layers=["honeytools", "honeytokens", "parameter_validator"]
    )
    user_task = suite.get_user_task_by_id("user_task_0")
    attack = ALL_ATTACKS[0]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": str(args.source),
        "source_archive_sha256": "EA08A9424F276E726DE3E96E747F3418C89FD65EDC7F21FC3E1D2BA31F59048E",
        "model": "deepseek-chat",
        "api_adapter": "developer role mapped to system role; official AgentShield logic unchanged",
        "normal": run_case(suite, tools, pipeline, detectors, user_task, None),
        "attack": {"id": attack["id"], "category": attack["category"], **run_case(suite, tools, pipeline, detectors, user_task, attack["payload"])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"normal_detections": len(result["normal"]["detections"]), "attack_detections": len(result["attack"]["detections"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
