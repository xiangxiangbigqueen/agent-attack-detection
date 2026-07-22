"""Shared type definitions - no torch dependency."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Record of a single tool call made by the agent."""
    session_id: str
    turn_id: int
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: float
    result: Optional[str] = None
    attack_triggered: bool = False
