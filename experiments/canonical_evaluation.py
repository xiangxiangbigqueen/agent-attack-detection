"""Reproducible, session-isolated evaluation for recorded API trajectories.

This script intentionally refuses to claim cross-session performance unless the
input records contain a shared ``attack_id`` and ordered ``session_index``.
It writes a run manifest beside the results so paper tables can be traced to
one immutable dataset hash and one Git revision.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.types import ToolCall
from detection.graph_detector import DetectorConfig, MultiLayerDetector

DATA = ROOT / "data" / "step1_api_trajectories.jsonl"
OUT_DIR = ROOT / "local_results" / "canonical"
SEED = 20260729


@dataclass(frozen=True)
class Session:
    name: str
    label: str
    fold: str
    calls: tuple[ToolCall, ...]
    cross_session: bool


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def load_sessions(path: Path) -> list[Session]:
    sessions: list[Session] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        tools = raw.get("tools", [])
        calls = tuple(
            ToolCall(
                session_id=str(raw.get("attack_id") or raw["task"]),
                turn_id=i,
                tool_name=tool["name"],
                parameters=tool.get("params", {}),
                timestamp=float(i),
            )
            for i, tool in enumerate(tools)
        )
        # Empty trajectories are observed outcomes (for example, an API
        # refusal or a failed tool-selection attempt).  They must remain in
        # denominators; silently dropping them inflates detection estimates.
        sessions.append(Session(
            name=str(raw["task"]), label=str(raw["type"]), fold=str(raw.get("fold", "")),
            calls=calls,
            cross_session="attack_id" in raw and "session_index" in raw,
        ))
    if not sessions:
        raise ValueError(f"No non-empty sessions in {path}")
    return sessions


def baseline_state(train: list[Session]) -> dict[str, Any]:
    detector = MultiLayerDetector(DetectorConfig(min_baseline_samples=3))
    detector.set_training(True)
    for session in train:
        detector.train_on(list(session.calls))
    base = detector.scorer.baseline
    if not base.is_fitted:
        raise ValueError("Training set does not fit a behavioral baseline")
    return {
        "tool_counts": dict(base.tool_counts),
        "transition_counts": {str(key): value for key, value in base.transition_counts.items()},
        "total_calls": base.total_calls,
        "total_transitions": base.total_transitions,
    }


def fresh_detector(state: dict[str, Any], threshold: float | None = None) -> MultiLayerDetector:
    cfg = DetectorConfig(min_baseline_samples=3, alert_threshold=threshold or 1e9)
    detector = MultiLayerDetector(cfg)
    base = detector.scorer.baseline
    base.tool_counts = defaultdict(int, state["tool_counts"])
    base.transition_counts = defaultdict(int, {ast.literal_eval(k): v for k, v in state["transition_counts"].items()})
    base.total_calls, base.total_transitions, base.is_fitted = state["total_calls"], state["total_transitions"], True
    detector.set_training(False)
    return detector


def max_score(session: Session, state: dict[str, Any]) -> float:
    detector = fresh_detector(state)
    return max((detector.analyze_call(call).layer_results["cumulative_score"] for call in session.calls), default=0.0)


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def evaluate(data: Path = DATA) -> dict[str, Any]:
    sessions = load_sessions(data)
    train = [s for s in sessions if s.label == "benign" and s.fold == "TRAIN"]
    benign_test = [s for s in sessions if s.label == "benign" and s.fold == "TEST"]
    attacks = [s for s in sessions if s.label != "benign"]
    if not train or not benign_test or not attacks:
        raise ValueError("Need explicit benign TRAIN, benign TEST, and attack records")
    state = baseline_state(train)
    calibration = [max_score(s, state) for s in train]
    threshold = float(np.quantile(calibration, 0.95, method="higher"))
    benign_scores = [max_score(s, state) for s in benign_test]
    attack_scores = [max_score(s, state) for s in attacks]
    fp = sum(score >= threshold for score in benign_scores)
    tp = sum(score >= threshold for score in attack_scores)
    by_type: dict[str, dict[str, Any]] = {}
    for label in sorted({s.label for s in attacks}):
        scores = [score for session, score in zip(attacks, attack_scores) if session.label == label]
        detected = sum(score >= threshold for score in scores)
        by_type[label] = {"detected": detected, "total": len(scores), "dr": detected / len(scores), "dr_wilson_95": wilson(detected, len(scores))}
    cross_session_ready = all(s.cross_session for s in attacks)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(), "seed": SEED,
        "dataset": {"path": str(data.relative_to(ROOT)), "sha256": sha256(data), "records": len(sessions)},
        "protocol": {"threshold": "P95 of per-session maxima on benign TRAIN", "session_isolation": True,
                     "cross_session_ready": cross_session_ready,
                     "warning": None if cross_session_ready else "Dataset has no ordered multi-session attack groups; do not claim cross-session performance."},
        "counts": {"benign_train": len(train), "benign_test": len(benign_test), "attack_test": len(attacks), "attack_types": dict(Counter(s.label for s in attacks))},
        "threshold": threshold,
        "metrics": {"dr": tp / len(attacks), "dr_wilson_95": wilson(tp, len(attacks)),
                    "fpr": fp / len(benign_test), "fpr_wilson_95": wilson(fp, len(benign_test)),
                    "tp": tp, "fp": fp, "by_attack_type": by_type},
        "scores": {"benign_test": benign_scores, "attack_test": attack_scores},
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = evaluate()
    destination = OUT_DIR / "canonical_evaluation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {destination}")
