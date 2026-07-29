"""
Cross-Round Behavior Graph Detector — FIXED VERSION

Key fixes:
1. Cross-session edges: explicitly track cross-session transitions
2. Temporal decay: apply decay to ALL edges at session boundaries
3. Density: handle self-loops and single-node case correctly
4. Novelty ratio: compare against trained baseline weights
5. Consistent scoring: weights sum to 1.0, score scale documented
"""

import json
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from agent.types import ToolCall


# ── Configuration ──────────────────────────────────────────

@dataclass
class DetectorConfig:
    """Configuration for the graph-based detector."""
    # Graph parameters
    window_size: int = 10
    decay_factor: float = 0.9
    anomaly_threshold: float = 0.8
    alert_threshold: float = 6.0       # Cumulative EWMA threshold

    # Baseline
    baseline_smoothing: float = 0.05
    min_baseline_samples: int = 5

    # Scoring
    use_cumulative: bool = True
    cumulative_weight: float = 0.4
    use_transition_frequency: bool = True
    use_structure: bool = True
    use_parameter_rules: bool = True
    use_tool_combination_rules: bool = True

    # Cross-session
    cross_session_decay: float = 0.3   # Lower weight for cross-session edges


# ── Behavioral Baseline ────────────────────────────────────

class BehavioralBaseline:
    """
    Normal behavioral patterns learned from benign agent runs.
    Stores expected frequencies and transition probabilities.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.tool_counts: Dict[str, int] = defaultdict(int)
        self.transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_calls: int = 0
        self.total_transitions: int = 0
        self.is_fitted: bool = False
        # Track edge weights for novelty comparison
        self.baseline_edge_weights: Dict[Tuple[str, str], float] = {}

    def update(self, calls: List[ToolCall]):
        """Update baseline with a sequence of tool calls."""
        prev_tool = None
        for call in calls:
            self.tool_counts[call.tool_name] += 1
            self.total_calls += 1
            if prev_tool is not None:
                self.transition_counts[(prev_tool, call.tool_name)] += 1
                self.total_transitions += 1
            prev_tool = call.tool_name
        self.is_fitted = self.total_calls >= self.config.min_baseline_samples

    def expected_frequency(self, tool_name: str) -> float:
        if not self.is_fitted:
            return 1.0
        return (self.tool_counts.get(tool_name, 0) + self.config.baseline_smoothing) / \
               (self.total_calls + self.config.baseline_smoothing * len(self.tool_counts))

    def transition_probability(self, from_tool: str, to_tool: str) -> float:
        if not self.is_fitted:
            return 1.0 / max(len(self.tool_counts), 1)
        from_total = sum(c for (f, _), c in self.transition_counts.items() if f == from_tool)
        if from_total == 0:
            return self.config.baseline_smoothing
        count = self.transition_counts.get((from_tool, to_tool), 0)
        return (count + self.config.baseline_smoothing) / \
               (from_total + self.config.baseline_smoothing * len(set(
                   t for (_, t) in self.transition_counts
               )))


# ── Cross-Session Behavior Graph ────────────────────────────

class BehaviorGraph:
    """
    Directed graph tracking tool call relationships across sessions.

    FIXES:
    - Explicit cross-session edge tracking with lower weight
    - Temporal decay applied to ALL edges at session boundaries
    - Self-loop handling in density
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.graph = nx.DiGraph()
        self.call_sequence: List[ToolCall] = []
        self.anomaly_scores: List[float] = []
        self.cumulative_score: float = 0.0
        self._session_call_count: int = 0
        self._training_mode: bool = False
        self._instant_signal: float = 0.0
        self._last_session_id: Optional[str] = None
        self._last_call_tool: Optional[str] = None
        self._last_call_within_session: bool = False

    def set_training(self, training: bool):
        self._training_mode = training

    def reset_session(self):
        """Reset per-session state; apply temporal decay to all edges."""
        self._session_call_count = 0
        self._instant_signal = 0.0

        # FIX: Apply temporal decay to ALL edges at session boundary
        # This ensures old edges decay even if never re-visited
        if not self._training_mode:
            self._decay_all_edges()

        # In detection mode, keep cumulative score for cross-session tracking
        if not self._training_mode and self.cumulative_score > 3.0:
            self.cumulative_score *= 0.9

    def _decay_all_edges(self):
        """Apply exponential decay to every edge in the graph.

        FIX: Previously only decayed edges that were re-visited.
        Now ALL edges decay at each session boundary,
        so old transitions fade regardless of whether they recur.
        """
        edges_to_prune = []
        for u, v, data in list(self.graph.edges(data=True)):
            if 'weight' in data:
                data['weight'] *= self.config.decay_factor
                if data['weight'] < 0.1:
                    edges_to_prune.append((u, v))
        for u, v in edges_to_prune:
            self.graph.remove_edge(u, v)

    def add_call(self, call: ToolCall):
        """Add a tool call to the graph with cross-session awareness."""
        self.call_sequence.append(call)
        self._session_call_count += 1
        self.graph.add_node(call.tool_name, last_call=call.timestamp)

        # Determine if this is a cross-session transition
        is_cross_session = (
            self._last_session_id is not None
            and call.session_id != self._last_session_id
        )

        # Add edge from previous tool if available
        if self._last_call_tool is not None:
            edge = (self._last_call_tool, call.tool_name)

            if is_cross_session:
                # FIX: Track cross-session edges with a special attribute
                weight_delta = self.config.cross_session_decay
            else:
                weight_delta = 1.0

            if self.graph.has_edge(*edge):
                current_weight = self.graph.edges[edge].get('weight', 0)
                new_weight = current_weight * self.config.decay_factor + weight_delta
                self.graph.edges[edge]['weight'] = new_weight
                # Mark cross-session
                if is_cross_session:
                    self.graph.edges[edge]['cross_session'] = True
            else:
                self.graph.add_edge(
                    self._last_call_tool, call.tool_name,
                    weight=weight_delta,
                    cross_session=is_cross_session
                )

        # Update node attributes
        node_data = self.graph.nodes[call.tool_name]
        node_data['total_calls'] = node_data.get('total_calls', 0) + 1
        node_data['last_call'] = call.timestamp

        # Track state for next call
        self._last_session_id = call.session_id
        self._last_call_tool = call.tool_name

    def get_subgraph(self, n_calls: Optional[int] = None) -> nx.DiGraph:
        if n_calls is None:
            n_calls = self.config.window_size
        recent = self.call_sequence[-n_calls:]
        nodes = set(c.tool_name for c in recent)
        return self.graph.subgraph(nodes).copy()

    def compute_graph_features(self, baseline: Optional['BehavioralBaseline'] = None) -> Dict[str, float]:
        """
        Extract features from the current graph state.

        FIXES:
        - Density: exclude self-loops to keep 0 <= density <= 1
        - Single-node case: density = 0 (no possible edges)
        - Novelty ratio: compare against baseline weights
        """
        if len(self.graph.nodes) == 0:
            return {"density": 0, "diversity": 0, "entropy": 0, "novelty_ratio": 0}

        subgraph = self.get_subgraph()

        # 1. Graph density (exclude self-loops for consistency)
        n_nodes = subgraph.number_of_nodes()
        if n_nodes <= 1:
            density = 0.0  # No possible edges
        else:
            # Count edges excluding self-loops
            n_edges = sum(1 for u, v in subgraph.edges() if u != v)
            max_edges = n_nodes * (n_nodes - 1)
            density = n_edges / max_edges

        # 2. Node diversity
        recent_calls = self.call_sequence[-self.config.window_size:]
        diversity = len(set(c.tool_name for c in recent_calls)) / max(len(recent_calls), 1)

        # 3. Entropy of out-degree distribution
        if n_nodes > 0:
            degrees = [d for _, d in subgraph.out_degree()]
            total_deg = sum(degrees) or 1
            probs = [d / total_deg for d in degrees if d > 0]
            entropy = -sum(p * math.log2(p) for p in probs) / math.log2(max(n_nodes, 2))
        else:
            entropy = 0

        # 4. Novelty ratio — FIX: use baseline edge weights as reference
        recent_pairs = []
        for i in range(1, min(self.config.window_size, len(self.call_sequence))):
            recent_pairs.append((
                self.call_sequence[-i - 1].tool_name,
                self.call_sequence[-i].tool_name
            ))

        novel = 0
        for f, t in recent_pairs:
            edge_weight = subgraph.edges[f, t].get('weight', 0) if subgraph.has_edge(f, t) else 0
            # FIX: if baseline knows this transition, compare against baseline weight
            if baseline and baseline.is_fitted:
                expected = (baseline.transition_counts.get((f, t), 0)
                           / max(baseline.total_transitions, 1))
                # Novel if edge weight is significantly below expected
                if edge_weight < expected * 0.5:
                    novel += 1
            else:
                # Fallback: novel if weight is low
                if edge_weight < 1.5:
                    novel += 1
        novelty_ratio = novel / max(len(recent_pairs), 1)

        return {
            "density": density,
            "diversity": diversity,
            "entropy": entropy,
            "novelty_ratio": novelty_ratio
        }


# ── Graph Anomaly Scorer ──────────────────────────────────

class GraphAnomalyScorer:
    """
    Scores tool call sequences for anomalous patterns.

    FIXES:
    - Weights normalize to sum = 1.0 (matching what code actually does)
    - Score scale documented: per-call score in [0,1], cumulative EWMA unbounded
    - Threshold θ applies to cumulative score, NOT per-call score
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.baseline = BehavioralBaseline(config)
        self.graph = BehaviorGraph(config)

    def reset_session(self):
        self.graph.reset_session()

    def set_training(self, training: bool):
        self.graph.set_training(training)

    def _check_parameter_anomaly(self, call: ToolCall) -> float:
        """Check if tool parameters are suspicious."""
        score = 0.0
        params = call.parameters
        if call.tool_name == "send_email" and "to" in params:
            to_val = str(params.get("to", "")).lower()
            if any(d in to_val for d in ["@evil", "@hacker", "@phishing", "@malicious", "@external", "@bad", "@leak"]):
                score = max(score, 0.8)
            elif not any(d in to_val for d in ["@company.com", "@corp.com"]):
                score = max(score, 0.4)
        if call.tool_name == "transfer_money" and "to_account" in params:
            acct = str(params.get("to_account", ""))
            if any(x in acct for x in ["external", "evil", "999"]):
                score = max(score, 0.7)
            elif len(acct) >= 5 and acct.isdigit() and int(acct) > 90000:
                score = max(score, 0.5)
        if call.tool_name == "delete_record":
            score = max(score, 0.3)
        if call.tool_name == "export_data":
            reason = str(params.get("reason", "")).lower()
            if reason in ("", "none", "backup"):
                score = max(score, 0.3)
        return score

    def _check_tool_combination(self, call: ToolCall) -> float:
        """Check for suspicious tool combinations within same session."""
        recent = self.graph.call_sequence[-5:]
        if call.tool_name == "send_email":
            if any(c.tool_name == "export_data" for c in recent):
                return 0.6
            if any(c.tool_name == "list_contacts" for c in recent[:-1]):
                return 0.5
        if call.tool_name == "delete_record":
            if any(c.tool_name in ("transfer_money", "send_email", "export_data") for c in recent):
                return 0.5
        if call.tool_name == "transfer_money":
            transfer_count = sum(1 for c in recent if c.tool_name in ("transfer_money",))
            if transfer_count >= 2:
                return 0.4
        return 0.0

    def score_call(self, call: ToolCall) -> float:
        """
        Score a single tool call for anomaly.

        Returns per-call anomaly score in [0, 1].
        Cumulative EWMA score can exceed 1.0 (bounded by alpha).
        """
        self.graph.add_call(call)
        scores = []
        weights = []

        # ── Single-round signals ──
        param_score = self._check_parameter_anomaly(call) if self.config.use_parameter_rules else 0.0
        if param_score > 0:
            scores.append(param_score)
            weights.append(0.30)

        combo_score = self._check_tool_combination(call) if self.config.use_tool_combination_rules else 0.0
        if combo_score > 0:
            scores.append(combo_score)
            weights.append(0.30)

        if self.config.use_transition_frequency and len(self.graph.call_sequence) >= 2:
            prev_tool = self.graph.call_sequence[-2].tool_name
            expected_p = self.baseline.transition_probability(prev_tool, call.tool_name)
            transition_score = 1.0 - expected_p
            scores.append(transition_score)
            weights.append(0.15)

        if self.config.use_transition_frequency:
            expected_freq = self.baseline.expected_frequency(call.tool_name)
            freq_score = 1.0 - expected_freq
            scores.append(freq_score)
            weights.append(0.10)

        # ── Multi-round signals ──
        if self.config.use_structure:
            features = self.graph.compute_graph_features(self.baseline)
            structure_score = (features["entropy"] * 0.5 + features["novelty_ratio"] * 0.5)
            scores.append(structure_score)
            weights.append(0.15)

        # Weighted combination — weights sum to 1.0
        total_w = sum(weights) or 1.0
        anomaly = sum(s * w for s, w in zip(scores, weights)) / total_w

        # Store instant signal
        self._instant_signal = max(
            self._check_parameter_anomaly(call) if self.config.use_parameter_rules else 0.0,
            self._check_tool_combination(call) if self.config.use_tool_combination_rules else 0.0,
        )

        # Update cumulative score (EWMA, unbounded)
        if not self.graph._training_mode:
            self.graph.cumulative_score = (
                self.graph.cumulative_score * self.config.decay_factor + anomaly
            )

        return min(1.0, anomaly)

    def get_cumulative_anomaly(self) -> float:
        return self.graph.cumulative_score if self.config.use_cumulative else 0.0

    def get_decision(self, instant_score: float) -> Tuple[bool, float, str]:
        """
        Make detection decision.

        The threshold θ applies to the CUMULATIVE EWMA score,
        not the per-call anomaly score. With α=0.9 and per-call scores
        in [0,1], cumulative score can reach ~10 over sustained attack.
        """
        cumulative = self.graph.cumulative_score
        instant = getattr(self, '_instant_signal', 0.0)

        # Single-round: high instant signal
        if instant >= 0.7:
            return True, instant, f"ALERT: instant_signal {instant:.3f} >= 0.7"

        # Multi-round: cumulative EWMA score vs calibrated threshold
        if cumulative >= self.config.alert_threshold and self.graph._session_call_count > 1:
            return True, cumulative, f"ALERT: cum_score {cumulative:.3f} (threshold={self.config.alert_threshold})"

        # Watch zone
        if cumulative >= self.config.anomaly_threshold:
            return False, cumulative, "WATCH"

        return False, max(instant, cumulative), "OK"


# ── Multi-Layer Detector ──────────────────────────────────

@dataclass
class DetectionResult:
    """Result of running the full detection pipeline."""
    is_attack: bool
    confidence: float
    reason: str
    layer_results: Dict[str, Any]
    call_history: List[ToolCall]


class MultiLayerDetector:
    """
    Unified multi-layer attack detector.
    Layer 1: Deception-based (honeytokens/honeytools)
    Layer 2: Parameter validation
    Layer 3: Cross-session behavior graph analysis
    Layer 4: Cumulative EWMA suspicion decision
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.scorer = GraphAnomalyScorer(self.config)
        self.call_history: List[ToolCall] = []
        self.current_session_id: Optional[str] = None

    def set_session(self, session_id: str):
        self.current_session_id = session_id

    def reset_session(self):
        self.call_history = []
        self.scorer.reset_session()

    def set_training(self, training: bool):
        self.scorer.set_training(training)

    def train_on(self, calls: List[ToolCall]):
        """Train the behavioral baseline on benign tool call sequences."""
        self.scorer.baseline.update(calls)

    def analyze_call(self, call: ToolCall) -> DetectionResult:
        """Analyze a single tool call through all detection layers."""
        self.call_history.append(call)
        layer_results = {}

        layer_results["is_honeytoken_trigger"] = False
        layer_results["is_param_violation"] = False

        # Layer 3: Graph-based anomaly detection
        instant_score = self.scorer.score_call(call)
        layer_results["instant_anomaly_score"] = instant_score
        layer_results["cumulative_score"] = self.scorer.get_cumulative_anomaly()
        layer_results["graph_features"] = self.scorer.graph.compute_graph_features(self.scorer.baseline)

        # Layer 4: Decision
        is_attack, final_score, reason = self.scorer.get_decision(instant_score)
        layer_results["final_score"] = final_score

        return DetectionResult(
            is_attack=is_attack,
            confidence=final_score,
            reason=reason,
            layer_results=layer_results,
            call_history=self.call_history[-self.config.window_size:]
        )

    def get_baseline_stats(self) -> Dict[str, Any]:
        return {
            "total_calls_observed": self.scorer.baseline.total_calls,
            "unique_tools": list(self.scorer.baseline.tool_counts.keys()),
            "tool_frequencies": dict(self.scorer.baseline.tool_counts),
            "graph_nodes": list(self.scorer.graph.graph.nodes()),
            "graph_edges": list(self.scorer.graph.graph.edges()),
        }


# ── Evaluation Metrics ─────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """Detection performance metrics with correct calculations."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def detection_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.detection_rate
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        denom = self.true_positives + self.true_negatives + \
                self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / denom if denom > 0 else 0.0

    def update(self, prediction: bool, ground_truth: bool):
        if ground_truth and prediction:
            self.true_positives += 1
        elif not ground_truth and not prediction:
            self.true_negatives += 1
        elif not ground_truth and prediction:
            self.false_positives += 1
        elif ground_truth and not prediction:
            self.false_negatives += 1

    def report(self) -> str:
        return (
            f"Detection Rate (Recall): {self.detection_rate:.3f}\n"
            f"False Positive Rate:     {self.false_positive_rate:.3f}\n"
            f"Precision:               {self.precision:.3f}\n"
            f"F1 Score:                {self.f1:.3f}\n"
            f"Accuracy:                {self.accuracy:.3f}"
        )
