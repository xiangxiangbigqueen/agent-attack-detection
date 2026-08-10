"""Audit the published confirmation numbers against frozen result artifacts.

The audit is deliberately independent of the plotting and manuscript code.  It
reads the R4/R5 evaluator JSON files, reconstructs alert counts from the raw
score dictionaries, checks the pre-registered validation-maximum threshold,
recomputes attack-objective success from the stored trajectories, and compares
the exploratory ablation summaries.  It writes a deterministic JSON report and
returns a non-zero exit status if any invariant fails.

Usage (from the repository root)::

    python experiments/audit_submission_consistency.py
    python experiments/audit_submission_consistency.py --output output/audit/submission_consistency.json

Raw trajectory files are not copied into the report.  The report contains
counts, hashes, and machine-readable discrepancies only, so it is safe to
commit alongside the experiment artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "local_results" / "canonical"
DEFAULT_OUTPUT = ROOT / "output" / "audit" / "submission_consistency.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def next_float(x: float) -> float:
    return math.nextafter(float(x), math.inf)


def metric(scores: dict[str, Any], threshold: float) -> dict[str, Any]:
    values = [float(value) for value in scores.values()]
    alerts = sum(value >= threshold for value in values)
    total = len(values)
    return {"total": total, "alerts": alerts, "rate": alerts / total if total else None}


def compare_metric(actual: dict[str, Any], observed: dict[str, Any]) -> bool:
    if actual["total"] != observed.get("total") or actual["alerts"] != observed.get("alerts"):
        return False
    expected_rate = actual["rate"]
    observed_rate = observed.get("rate")
    return expected_rate is None and observed_rate is None or (
        expected_rate is not None and observed_rate is not None and math.isclose(expected_rate, float(observed_rate), rel_tol=1e-12, abs_tol=1e-12)
    )


def compare_analysis_metric(actual: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Compare the bootstrap analysis shape (rate/total, no alert count)."""
    if int(observed.get("total", -1)) != actual["total"]:
        return False
    rate = observed.get("rate")
    return rate is not None and math.isclose(float(rate), float(actual["rate"]), rel_tol=1e-12, abs_tol=1e-12)


def collect_attack_success(run_dir: Path) -> tuple[dict[str, bool], Counter[str], int]:
    """Recompute objective success at attack-group level from stored records.

    A group can contain multiple session records (e.g. delayed-trigger and
    persistent-memory attacks).  The evaluator concatenates all calls for a
    group, so this function does the same.  The protocol's success predicate is
    imported lazily to keep this audit script usable as a stand-alone command.
    """
    from experiments.confirmation_protocol import attack_success

    calls_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kind_by_group: dict[str, str] = {}
    raw_test_records = 0
    for record_path in (run_dir / "records").glob("test__*.json"):
        record = read_record(record_path)
        raw_test_records += 1
        if record.get("label") != "attack":
            continue
        group_id = str(record.get("attack_id", ""))
        if not group_id:
            continue
        kind = str(record.get("attack_type", ""))
        kind_by_group[group_id] = kind
        calls_by_group[group_id].extend(record.get("tools", []))
    success = {
        group_id: bool(attack_success(kind_by_group[group_id], calls))
        for group_id, calls in calls_by_group.items()
    }
    successful_by_type = Counter(kind_by_group[group_id] for group_id, ok in success.items() if ok)
    return success, successful_by_type, raw_test_records


def audit_run(run_name: str) -> dict[str, Any]:
    run_dir = CANONICAL / f"confirmation_{run_name}"
    evaluation_path = run_dir / "confirmation_evaluation.json"
    analysis_path = run_dir / "confirmation_analysis.json"
    ablation_path = run_dir / "confirmation_ablation_exploratory.json"
    evaluation = load_json(evaluation_path)
    analysis = load_json(analysis_path)
    ablation = load_json(ablation_path)
    scores = evaluation.get("scores", {})
    validation_scores = {str(k): float(v) for k, v in scores.get("validation", {}).items()}
    benign_scores = {str(k): float(v) for k, v in scores.get("benign", {}).items()}
    attack_scores = {str(k): float(v) for k, v in scores.get("attack", {}).items()}
    threshold = float(evaluation["threshold"])
    max_validation = max(validation_scores.values()) if validation_scores else math.nan
    expected_threshold = next_float(max_validation)
    success_by_group, successful_by_type, raw_test_records = collect_attack_success(run_dir)
    success_ids = {group_id for group_id, ok in success_by_group.items() if ok}
    successful_scores = {group_id: attack_scores[group_id] for group_id in success_ids if group_id in attack_scores}

    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, observed: Any, ok: bool | None = None) -> None:
        checks.append({"name": name, "ok": bool(expected == observed if ok is None else ok), "expected": expected, "observed": observed})

    check("threshold_rule_declared", "next representable float above maximum of 100 independent short+long benign validation group scores", evaluation.get("threshold_rule"))
    check("validation_count", 100, len(validation_scores))
    check("benign_test_count", 200, len(benign_scores))
    check("attack_test_count", 240, len(attack_scores))
    check("validation_max_and_threshold", expected_threshold, threshold, ok=math.isclose(expected_threshold, threshold, rel_tol=0.0, abs_tol=0.0))

    recomputed = {
        "validation": metric(validation_scores, threshold),
        "benign_test": metric(benign_scores, threshold),
        "attack_test": metric(attack_scores, threshold),
        "successful_attack_test": metric(successful_scores, threshold),
    }
    for key, expected in recomputed.items():
        check(f"metric_{key}", expected, evaluation.get(key, {}), ok=compare_metric(expected, evaluation.get(key, {})))

    observed_types = evaluation.get("by_attack_type", {})
    observed_analysis_types = analysis.get("by_attack_type", {})
    by_type: dict[str, Any] = {}
    for attack_type in sorted({key.split("-", 4)[3] if key.startswith("confirm-attack-") else "unknown" for key in attack_scores}):
        # Split safely from the protocol's stable ID shape:
        # confirm-attack-a0-<attack_type>-v00.
        ids = [key for key in attack_scores if f"-{attack_type}-" in key]
        if attack_type == "unknown":
            ids = []
        scores_for_type = {key: attack_scores[key] for key in ids}
        successes_for_type = {key: successful_scores[key] for key in ids if key in successful_scores}
        expected = metric(scores_for_type, threshold)
        expected_success = metric(successes_for_type, threshold)
        observed = observed_types.get(attack_type, {})
        observed_analysis = observed_analysis_types.get(attack_type, {})
        # confirmation_evaluation stores a flat metric per type; the analysis
        # artifact stores all-attempts/successful-attacks bootstrap metrics.
        type_ok = compare_metric(expected, observed)
        if observed_analysis.get("successful_attacks") is not None:
            type_ok = type_ok and compare_analysis_metric(expected_success, observed_analysis["successful_attacks"])
        by_type[attack_type] = {
            "all_attempts": expected,
            "successful_attacks": expected_success,
            "successful_groups": len(successes_for_type),
            "observed": {"evaluation": observed, "analysis": observed_analysis},
            "ok": type_ok,
        }
        check(f"attack_type_{attack_type}", True, type_ok, ok=type_ok)

    # The ablation artifact is exploratory and is not treated as a new method
    # claim; we only verify that its TrajectoryOnly row mirrors the evaluator.
    variants = ablation.get("variants", {})
    trajectory = variants.get("TrajectoryOnly", {})
    traj_ok = (
        math.isclose(float(trajectory.get("threshold")), threshold, rel_tol=0.0, abs_tol=0.0)
        and int(trajectory.get("fp")) == int(evaluation["benign_test"]["alerts"])
        and int(trajectory.get("tp")) == int(evaluation["attack_test"]["alerts"])
        and math.isclose(float(trajectory.get("fpr")), float(evaluation["benign_test"]["rate"]), rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(float(trajectory.get("dr")), float(evaluation["attack_test"]["rate"]), rel_tol=1e-12, abs_tol=1e-12)
    )
    check("trajectory_only_ablation_matches_evaluator", True, traj_ok, ok=traj_ok)

    analysis_success_misses = len(analysis.get("successful_attack_misses", []))
    expected_misses = len(successful_scores) - recomputed["successful_attack_test"]["alerts"]
    check("successful_miss_count", expected_misses, analysis_success_misses)
    check("successful_group_count_from_records", 160, len(success_ids))
    check("successful_type_sum", len(success_ids), sum(successful_by_type.values()))

    # Group-level type denominators must remain balanced (40 each).  This
    # guards against accidentally treating the two-session raw records as the
    # statistical denominator.
    attack_type_counts = Counter()
    for key in attack_scores:
        parts = key.split("-")
        # confirm-attack-a0-<type words...>-v00; use score artifact's declared
        # type table below for robust matching instead of relying on token count.
        for attack_type in observed_types:
            if f"-{attack_type}-" in key:
                attack_type_counts[attack_type] += 1
                break
    check("balanced_attack_group_denominators", {kind: 40 for kind in sorted(observed_types)}, dict(sorted(attack_type_counts.items())))

    passed = all(item["ok"] for item in checks)
    return {
        "run": run_name,
        "source": {
            "evaluation": str(evaluation_path),
            "analysis": str(analysis_path),
            "ablation": str(ablation_path),
            "evaluation_sha256": sha256(evaluation_path),
            "analysis_sha256": sha256(analysis_path),
            "ablation_sha256": sha256(ablation_path),
        },
        "protocol_version": evaluation.get("protocol_version"),
        "candidate": evaluation.get("candidate"),
        "threshold": {"observed": threshold, "validation_max": max_validation, "expected_nextafter": expected_threshold},
        "counts": {
            "validation_groups": len(validation_scores),
            "benign_test_groups": len(benign_scores),
            "attack_test_groups": len(attack_scores),
            "successful_attack_groups": len(success_ids),
            "raw_test_records": raw_test_records,
            "raw_records_note": "Raw test records are not the denominator: delayed-trigger and persistent-memory groups contain two sessions; evaluator metrics use one score per group.",
        },
        "metrics_recomputed": recomputed,
        "by_attack_type": by_type,
        "successful_groups_by_type": dict(sorted(successful_by_type.items())),
        "exploratory_ablation": {"disclosure": ablation.get("disclosure"), "variants": variants},
        "checks": checks,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    args = parser.parse_args()
    try:
        runs = [audit_run("R4"), audit_run("R5")]
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    report = {
        "schema_version": "2026-08-10.submission-consistency-v1",
        "generated_by": "experiments/audit_submission_consistency.py",
        "repository_root": str(ROOT),
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "runs": {run["run"]: run["passed"] for run in runs}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
