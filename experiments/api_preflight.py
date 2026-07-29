"""Verify the configured LLM endpoint against an in-memory sandbox only."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.env import BankingEnvironment
from agent.function_agent import AgentConfig, FunctionCallingAgent


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for the API preflight")

    agent = FunctionCallingAgent(
        AgentConfig(api_key=api_key, model="deepseek-chat", temperature=0.1, max_turns=8),
        BankingEnvironment(),
    )
    calls = agent.run("What is my checking account balance?")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_turns": 8,
        "sandbox_only": True,
        "tool_calls": [call.tool_name for call in calls],
        "status": "ok" if calls else "empty_tool_trace",
    }
    destination = ROOT / "local_results" / "canonical" / "api_preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
