"""Create an immutable-by-convention manifest of R1--R3 development evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "local_results" / "canonical"
DESTINATION = CANONICAL / "development_freeze_20260730.json"

ROOTS = {
    "R1": CANONICAL / "paper_protocol_20260729_1",
    "R2": CANONICAL / "replication_R2" / "raw",
    "R3": CANONICAL / "replication_R3" / "raw",
    "official_R1": CANONICAL / "official_agentshield_R1",
    "official_R2": CANONICAL / "official_agentshield_R2",
    "official_R3": CANONICAL / "official_agentshield_R3",
}
SCRIPTS = (
    ROOT / "detection" / "graph_detector.py",
    ROOT / "experiments" / "canonical_protocol.py",
    ROOT / "experiments" / "run_long_calibrated_evaluation.py",
    ROOT / "experiments" / "run_long_calibrated_ablation.py",
    ROOT / "experiments" / "evaluate_attack_outcomes.py",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_summary(root: Path) -> dict:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest(path).encode("ascii"))
    return {"path": str(root), "files": len(files), "sha256_tree": aggregate.hexdigest()}


def main() -> None:
    if DESTINATION.exists():
        raise RuntimeError(f"freeze manifest already exists: {DESTINATION}")
    missing = [str(path) for path in [*ROOTS.values(), *SCRIPTS] if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot freeze missing paths: {missing}")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "R1--R3 are development/diagnostic evidence only; no post-freeze method change may report them as independent confirmation.",
        "development_roots": {name: tree_summary(path) for name, path in ROOTS.items()},
        "code": {str(path.relative_to(ROOT)): digest(path) for path in SCRIPTS},
        "future_confirmation_rule": "Final performance claims require fresh R4/R5 trajectories collected after the revised detector protocol is frozen.",
    }
    DESTINATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(DESTINATION), "roots": list(ROOTS), "scripts": len(SCRIPTS)}, indent=2))


if __name__ == "__main__":
    main()
