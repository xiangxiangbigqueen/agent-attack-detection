"""Run same-trajectory statistical baselines on the frozen R4/R5 protocol.

The script deliberately reads only the stored confirmation records.  It uses
the same benign training/validation/test groups as ``evaluate_confirmation``
and calibrates each baseline by the next representable float above the largest
validation group score.  No attack/test record is used for fitting or
threshold selection.  Methods are intentionally simple, auditable baselines:
tool-frequency z-score, first-order Markov transition surprise, a regularized
Mahalanobis detector, and (when scikit-learn is available) Isolation Forest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.confirmation_protocol import all_episodes, attack_success

SEED = 20260729
METHOD_ORDER = ("tool_frequency_zscore", "markov_transition", "mahalanobis", "isolation_forest")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z = 1.96
    p = successes / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, c - m), 6), round(min(1.0, c + m), 6)]


def calls(row: dict[str, Any]) -> list[str]:
    return [str(item.get("name", "<missing>")) for item in row.get("tools", [])]


def row_index(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Index a confirmation root by group id, requiring successful records."""
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("collector_status") == "ok":
            result[str(row["attack_id"])].append(row)
    for values in result.values():
        values.sort(key=lambda x: int(x.get("session_index", 0)))
    return dict(result)


def select_raw(root: Path) -> dict[str, list[dict[str, Any]]]:
    indexed = row_index(root)
    selected: dict[str, list[dict[str, Any]]] = {}
    for episode in all_episodes():
        values = indexed.get(episode.group_id, [])
        expected = list(range(len(episode.sessions)))
        if [int(row.get("session_index", -1)) for row in values] != expected:
            raise RuntimeError(f"incomplete R4/R5 group {episode.group_id}: found {len(values)} records")
        selected[episode.group_id] = values
    return selected


def select_long(root: Path, split: str, expected: int) -> list[dict[str, Any]]:
    """Select one successful long-benign row per group across recovery dirs."""
    chosen: dict[str, dict[str, Any]] = {}
    folders = [root] + sorted(path for path in root.glob("recovery_*") if path.is_dir())
    for folder in folders:
        for path in sorted((folder / "records").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("split") == split and row.get("collector_status") == "ok":
                chosen.setdefault(str(row["attack_id"]), row)
    rows = sorted(chosen.values(), key=lambda x: str(x["attack_id"]))
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} long benign {split} groups, found {len(rows)}")
    return rows


@dataclass(frozen=True)
class Dataset:
    run: str
    train: list[list[str]]
    validation: list[list[list[str]]]
    benign_test: dict[str, list[list[str]]]
    attacks: dict[str, list[list[str]]]
    attack_types: dict[str, str]
    successful: set[str]
    benign_lengths: dict[str, int]
    attack_lengths: dict[str, int]
    source_hashes: dict[str, str]


def load_dataset(run: str, root: Path) -> Dataset:
    raw_root = root / f"confirmation_{run}"
    long_root = raw_root / "long_benign_calibration"
    raw = select_raw(raw_root)
    episodes = {episode.group_id: episode for episode in all_episodes()}
    long_train = select_long(long_root, "train", 100)
    long_validation = select_long(long_root, "validation", 50)
    long_test = select_long(long_root, "test", 100)

    train = [calls(raw[key][0]) for key, episode in episodes.items() if episode.split == "train"]
    train.extend(calls(row) for row in long_train)
    validation = [[calls(row) for row in raw[key]] for key, episode in episodes.items() if episode.split == "validation"]
    validation.extend([[calls(row)] for row in long_validation])
    benign_test = {key: [calls(row) for row in raw[key]] for key, episode in episodes.items() if episode.label == "benign" and episode.split == "test"}
    benign_test.update({str(row["attack_id"]): [calls(row)] for row in long_test})
    attacks = {key: [calls(row) for row in raw[key]] for key, episode in episodes.items() if episode.label == "attack"}
    attack_types = {key: str(episodes[key].attack_type) for key in attacks}
    successful = {
        key for key, rows in raw.items()
        if key in attacks and attack_success(
            attack_types[key],
            [item for session in rows for item in session.get("tools", [])],
        )
    }
    benign_lengths = {key: sum(map(len, values)) for key, values in benign_test.items()}
    attack_lengths = {key: sum(map(len, values)) for key, values in attacks.items()}
    source_hashes = {
        "raw_index": sha256(raw_root / "index.jsonl"),
        "long_index": sha256(long_root / "index.jsonl"),
        "protocol": sha256(ROOT / "experiments" / "confirmation_protocol.py"),
    }
    return Dataset(run, train, validation, benign_test, attacks, attack_types, successful, benign_lengths, attack_lengths, source_hashes)


def all_tools(sequences: Iterable[list[str]]) -> list[str]:
    return sorted({tool for sequence in sequences for tool in sequence})


class ToolFrequency:
    name = "tool_frequency_zscore"

    def fit(self, train: list[list[str]]) -> None:
        self.vocab = all_tools(train)
        vectors = np.asarray([self.vector(sequence) for sequence in train], dtype=float)
        self.mean = vectors.mean(axis=0)
        self.std = np.maximum(vectors.std(axis=0, ddof=1), 1e-6)

    def vector(self, sequence: list[str]) -> np.ndarray:
        counts = Counter(sequence)
        n = max(len(sequence), 1)
        return np.asarray([counts.get(tool, 0) / n for tool in self.vocab], dtype=float)

    def score(self, sequence: list[str]) -> float:
        if not sequence:
            return 0.0
        z = np.abs((self.vector(sequence) - self.mean) / self.std)
        return float(np.max(z, initial=0.0))


class MarkovTransition:
    name = "markov_transition"

    def fit(self, train: list[list[str]]) -> None:
        self.vocab = all_tools(train)
        self.vocab_set = set(self.vocab)
        self.alpha = 0.5
        self.counts: Counter[tuple[str, str]] = Counter()
        self.out: Counter[str] = Counter()
        for sequence in train:
            for previous, current in zip(sequence, sequence[1:]):
                self.counts[(previous, current)] += 1
                self.out[previous] += 1

    def score(self, sequence: list[str]) -> float:
        if len(sequence) < 2:
            return 0.0
        size = max(len(self.vocab), 1)
        surprises = []
        for previous, current in zip(sequence, sequence[1:]):
            p = (self.counts[(previous, current)] + self.alpha) / (self.out[previous] + self.alpha * size)
            surprises.append(-math.log2(max(p, 1e-12)))
        return float(max(surprises, default=0.0))


class FeatureVector:
    """Tool and transition proportions, with an explicit unknown bucket."""
    def fit(self, train: list[list[str]]) -> None:
        self.tools = all_tools(train)
        self.transitions = sorted({pair for seq in train for pair in zip(seq, seq[1:])})
        self.tool_index = {key: index for index, key in enumerate(self.tools)}
        self.transition_index = {key: index for index, key in enumerate(self.transitions)}

    def vector(self, sequence: list[str]) -> np.ndarray:
        n = max(len(sequence), 1)
        tool = np.zeros(len(self.tools) + 1, dtype=float)
        for item in sequence:
            index = self.tool_index.get(item)
            tool[index if index is not None else -1] += 1 / n
        transitions = list(zip(sequence, sequence[1:]))
        m = max(len(transitions), 1)
        edge = np.zeros(len(self.transitions) + 1, dtype=float)
        for item in transitions:
            index = self.transition_index.get(item)
            edge[index if index is not None else -1] += 1 / m
        return np.concatenate([tool, edge])


class Mahalanobis:
    name = "mahalanobis"

    def fit(self, train: list[list[str]]) -> None:
        self.features = FeatureVector()
        self.features.fit(train)
        matrix = np.asarray([self.features.vector(sequence) for sequence in train], dtype=float)
        self.mean = matrix.mean(axis=0)
        centered = matrix - self.mean
        covariance = centered.T @ centered / max(len(matrix) - 1, 1)
        # A fixed ridge makes the score defined even when tool/edge columns are
        # collinear (common for short banking traces).
        ridge = max(float(np.trace(covariance)) / max(covariance.shape[0], 1) * 1e-3, 1e-6)
        self.inverse = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))

    def score(self, sequence: list[str]) -> float:
        delta = self.features.vector(sequence) - self.mean
        return float(delta @ self.inverse @ delta)


class IsolationForestBaseline:
    name = "isolation_forest"

    def __init__(self) -> None:
        self.available = False

    def fit(self, train: list[list[str]]) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except Exception as exc:
            self.error = f"scikit-learn unavailable: {exc}"
            return
        self.features = FeatureVector()
        self.features.fit(train)
        matrix = np.asarray([self.features.vector(sequence) for sequence in train], dtype=float)
        self.model = IsolationForest(n_estimators=256, contamination="auto", random_state=SEED, n_jobs=1)
        self.model.fit(matrix)
        self.available = True

    def score(self, sequence: list[str]) -> float:
        if not self.available:
            return float("nan")
        return float(-self.model.decision_function(self.features.vector(sequence).reshape(1, -1))[0])


def score_group(model: Any, group: list[list[str]]) -> float:
    return max((float(model.score(sequence)) for sequence in group), default=0.0)


def rate(ids: list[str], scores: dict[str, float], threshold: float) -> dict[str, Any]:
    alerts = sum(scores[key] >= threshold for key in ids)
    return {"alerts": alerts, "total": len(ids), "rate": alerts / len(ids) if ids else None, "wilson_95": wilson(alerts, len(ids))}


def length_matched(ids_attack: list[str], ids_benign: list[str], lengths_attack: dict[str, int], lengths_benign: dict[str, int], scores: dict[str, float], threshold: float) -> dict[str, Any]:
    attacks_by_length: dict[int, list[str]] = defaultdict(list)
    benign_by_length: dict[int, list[str]] = defaultdict(list)
    for key in ids_attack:
        attacks_by_length[lengths_attack[key]].append(key)
    for key in ids_benign:
        benign_by_length[lengths_benign[key]].append(key)
    chosen_attack: list[str] = []
    chosen_benign: list[str] = []
    strata = {}
    for length in sorted(set(attacks_by_length) & set(benign_by_length)):
        attack = sorted(attacks_by_length[length])
        benign = sorted(benign_by_length[length])
        count = min(len(attack), len(benign))
        chosen_attack.extend(attack[:count])
        chosen_benign.extend(benign[:count])
        strata[str(length)] = {"attack_available": len(attack), "benign_available": len(benign), "matched_per_class": count}
    return {
        "matching": "exact total tool-call count, deterministic lexicographic selection without replacement",
        "attack": rate(chosen_attack, scores, threshold),
        "benign": rate(chosen_benign, scores, threshold),
        "strata": strata,
    }


def evaluate_method(method: Any, dataset: Dataset) -> dict[str, Any]:
    method.fit(dataset.train)
    if getattr(method, "available", True) is False:
        return {"method": method.name, "status": "skipped", "reason": getattr(method, "error", "unavailable")}
    validation_scores = [score_group(method, group) for group in dataset.validation]
    threshold = float(np.nextafter(max(validation_scores), np.inf))
    benign_scores = {key: score_group(method, group) for key, group in dataset.benign_test.items()}
    attack_scores = {key: score_group(method, group) for key, group in dataset.attacks.items()}
    benign_ids = sorted(benign_scores)
    attack_ids = sorted(attack_scores)
    successful_ids = sorted(dataset.successful)
    by_type = {}
    for kind in sorted(set(dataset.attack_types.values())):
        ids = [key for key in attack_ids if dataset.attack_types[key] == kind]
        by_type[kind] = rate(ids, attack_scores, threshold)
    combined_scores = {**benign_scores, **attack_scores}
    return {
        "method": method.name,
        "status": "ok",
        "fit": {"training_sessions": len(dataset.train), "validation_groups": len(dataset.validation)},
        "threshold_rule": "nextafter(max per-group benign validation score, +inf)",
        "threshold": threshold,
        "validation": rate([str(i) for i in range(len(validation_scores))], {str(i): value for i, value in enumerate(validation_scores)}, threshold),
        "benign_test": rate(benign_ids, benign_scores, threshold),
        "attack_test": rate(attack_ids, attack_scores, threshold),
        "successful_attack_test": rate(successful_ids, attack_scores, threshold),
        "by_attack_type": by_type,
        "length_matched": length_matched(attack_ids, benign_ids, dataset.attack_lengths, dataset.benign_lengths, combined_scores, threshold),
        "scores": {"validation": validation_scores, "benign": benign_scores, "attack": attack_scores},
    }


def run(run: str, root: Path, methods: list[str]) -> dict[str, Any]:
    dataset = load_dataset(run, root / "local_results" / "canonical")
    instances = {
        "tool_frequency_zscore": ToolFrequency(),
        "markov_transition": MarkovTransition(),
        "mahalanobis": Mahalanobis(),
        "isolation_forest": IsolationForestBaseline(),
    }
    results = {}
    for name in methods:
        results[name] = evaluate_method(instances[name], dataset)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "seed": SEED,
        "run": run,
        "protocol_version": "2026-07-30.confirmation-v1",
        "dataset": {"source_hashes": dataset.source_hashes, "train_sessions": len(dataset.train), "validation_groups": len(dataset.validation), "benign_test_groups": len(dataset.benign_test), "attack_test_groups": len(dataset.attacks)},
        "methods": results,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", choices=["R4", "R5"], default=["R4", "R5"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "baselines")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_ORDER), default=list(METHOD_ORDER))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for run_name in args.runs:
        result = run(run_name, args.root, args.methods)
        destination = args.output / f"{run_name.lower()}_baseline_suite.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results[run_name] = result
        print(json.dumps({name: value.get("attack_test") for name, value in result["methods"].items()}, indent=2))
    summary = {"schema_version": 1, "runs": list(all_results), "results": {run: {name: value.get("attack_test") for name, value in result["methods"].items()} for run, result in all_results.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
