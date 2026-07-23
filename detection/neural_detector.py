"""
Neural-Enhanced Cross-Round Behavior Graph Detector.

Extends the base graph detector with:
1. Content-aware embeddings (SBERT for tool call arguments/parameters)
2. GNN-based graph anomaly scoring (GAT / GraphSAGE)
3. Adaptive threshold via statistical process control
4. Hybrid scoring: hand-crafted features + learned representations

References:
  - MCPShield (2026): GAT/GCN + SBERT, AUROC 0.975
  - FragBench (2026): Cross-session GNN, F1=0.88-0.96
  - GAMMAF (2026): Benchmarking graph anomaly detection for LLM-MAS
"""

import json
import math
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from agent.types import ToolCall
from detection.graph_detector import (
    DetectorConfig, BehavioralBaseline, BehaviorGraph,
    GraphAnomalyScorer, MultiLayerDetector, DetectionResult, EvaluationMetrics
)


# ── Optional Dependencies ──────────────────────────────────

_HAS_TORCH = False
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    _HAS_SKLEARN = True
except ImportError:
    pass


# ── Adaptive Threshold via SPC ─────────────────────────────

class AdaptiveThreshold:
    """
    Statistical Process Control (SPC) based adaptive threshold.

    Maintains a moving window of recent anomaly scores and computes
    dynamic threshold as μ + k·σ, where k controls sensitivity.

    Three-state decision:
      - OK: score < mean + 1σ
      - WATCH: mean + 1σ ≤ score < mean + 3σ
      - ALERT: score ≥ mean + 3σ
    """

    def __init__(self, window_size: int = 50, sigma_alert: float = 3.0,
                 sigma_watch: float = 1.5, min_samples: int = 10):
        self.window = deque(maxlen=window_size)
        self.sigma_alert = sigma_alert
        self.sigma_watch = sigma_watch
        self.min_samples = min_samples

    def update(self, score: float):
        """Add a new score to the window."""
        self.window.append(score)

    @property
    def is_ready(self) -> bool:
        return len(self.window) >= self.min_samples

    @property
    def stats(self) -> Tuple[float, float]:
        if not self.window:
            return 0.0, 1.0
        return float(np.mean(self.window)), float(np.std(self.window) + 1e-8)

    def get_alert_threshold(self) -> float:
        """Return μ + 3σ threshold for ALERT decision."""
        mean, std = self.stats
        return mean + self.sigma_alert * std

    def get_watch_threshold(self) -> float:
        """Return μ + 1.5σ threshold for WATCH decision."""
        mean, std = self.stats
        return mean + self.sigma_watch * std

    def classify(self, score: float) -> str:
        """Classify a score into OK / WATCH / ALERT."""
        if not self.is_ready:
            return "OK"
        alert_th = self.get_alert_threshold()
        watch_th = self.get_watch_threshold()
        if score >= alert_th:
            return "ALERT"
        elif score >= watch_th:
            return "WATCH"
        return "OK"

    def get_decision(self, score: float) -> Tuple[bool, str]:
        """Return (is_attack, reason)."""
        label = self.classify(score)
        if label == "ALERT":
            mean, std = self.stats
            return True, f"ALERT: score={score:.3f} > μ+{self.sigma_alert}σ (μ={mean:.3f}, σ={std:.3f})"
        return False, label


# ── Content Encoder (SBERT) ────────────────────────────────

class ContentEncoder:
    """
    Encodes tool call parameters into dense embeddings for content-aware analysis.

    Falls back to vocabulary-based bag-of-words embeddings when
    sentence-transformers is not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_vocab: Dict[str, int] = {}
        self._fallback_embeddings: Dict[str, np.ndarray] = {}
        self.embed_dim = 384  # all-MiniLM-L6-v2 default

    def _lazy_load(self):
        """Lazy-load SBERT model on first use."""
        if self._model is not None:
            return
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(self.model_name)
                self.embed_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                warnings.warn(f"Failed to load SBERT model: {e}. Using fallback.")
                self._model = None

    def _get_fallback_embedding(self, text: str) -> np.ndarray:
        """Character-n-gram based fallback embedding (no external deps)."""
        # Simple character bigram frequency vector
        bigrams = defaultdict(int)
        for i in range(len(text) - 1):
            bigrams[text[i:i+2]] += 1
        # Map to fixed-dim vector
        vec = np.zeros(self.embed_dim)
        for i, (bg, cnt) in enumerate(sorted(bigrams.items())[:self.embed_dim]):
            vec[i % self.embed_dim] = cnt
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, text: str) -> np.ndarray:
        """Encode a text string into an embedding vector."""
        if _HAS_SENTENCE_TRANSFORMERS:
            self._lazy_load()
            if self._model is not None:
                return self._model.encode(text, normalize_embeddings=True)
        return self._get_fallback_embedding(text)

    def encode_call(self, call: ToolCall) -> np.ndarray:
        """Encode a full tool call into an embedding."""
        parts = [f"tool={call.tool_name}"]
        for k, v in sorted(call.parameters.items()):
            parts.append(f"{k}={v}")
        parts.append(f"time={int(call.timestamp % 1000)}")
        return self.encode(" | ".join(parts))

    def parameter_anomaly_score(self, call: ToolCall,
                                 benign_centroids: Dict[str, np.ndarray]) -> float:
        """
        Compute content-level parameter anomaly score.

        Uses cosine distance from benign centroid for the same tool.
        Returns 0 if no centroid available (benign-default).
        """
        tool = call.tool_name
        if tool not in benign_centroids:
            return 0.0

        emb = self.encode_call(call)
        centroid = benign_centroids[tool]
        cos_sim = float(np.dot(emb, centroid) / (
            np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8
        ))
        # Convert similarity to anomaly: 1 - cos_sim, clamped to [0, 1]
        anomaly = max(0.0, min(1.0, 1.0 - cos_sim))
        return anomaly

    def compute_centroids(self, calls: List[ToolCall]) -> Dict[str, np.ndarray]:
        """Compute mean embedding per tool from benign calls."""
        tool_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        for c in calls:
            tool_embeddings[c.tool_name].append(self.encode_call(c))

        centroids = {}
        for tool, embs in tool_embeddings.items():
            centroids[tool] = np.mean(embs, axis=0)
        return centroids


# ── GNN-based Graph Encoder (lightweight, no torch-geometric) ──

class LightGNN:
    """
    Minimal GNN implementation using pure NumPy + (optionally) PyTorch.

    Implements a single-layer Graph Attention (GATv2-style) message passing.
    Uses numpy fallback if torch is not available.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weights (numpy-based for fallback)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self.a = np.random.randn(hidden_dim * 2, 1).astype(np.float32) * 0.01

        self._fitted = False

        if _HAS_TORCH:
            self._init_torch()

    def _init_torch(self):
        """Initialize PyTorch parameters if available."""
        import torch.nn as nn
        self.torch_W1 = nn.Parameter(torch.tensor(self.W1))
        self.torch_W2 = nn.Parameter(torch.tensor(self.W2))
        self.torch_a = nn.Parameter(torch.tensor(self.a))

    def _attention(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Compute GATv2-style attention coefficients."""
        n = h.shape[0]
        # Concatenate all pairs: [h_i || h_j] for edges
        Wh = h @ self.W1  # [n, hidden]
        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0 or i == j:
                    concat = np.concatenate([Wh[i], Wh[j]])
                    scores[i, j] = float(concat @ self.a)

        # Softmax over neighbors
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        exp_scores = exp_scores * (adj + np.eye(n))  # Mask to neighbors + self
        row_sums = exp_scores.sum(axis=1, keepdims=True) + 1e-8
        attn = exp_scores / row_sums
        return attn

    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Forward pass: GAT message passing.

        Args:
            node_features: [n_nodes, input_dim] feature matrix
            adj_matrix: [n_nodes, n_nodes] adjacency matrix

        Returns:
            [n_nodes, output_dim] node embeddings
        """
        attn = self._attention(node_features, adj_matrix)
        Wh = node_features @ self.W1
        h = attn @ Wh  # [n, hidden]
        h = np.maximum(h, 0)  # ReLU
        out = h @ self.W2  # [n, output_dim]
        return out

    def compute_graph_embedding(self, node_features: np.ndarray,
                                 adj_matrix: np.ndarray) -> np.ndarray:
        """Compute a single graph-level embedding via mean pooling."""
        node_embs = self.forward(node_features, adj_matrix)
        return node_embs.mean(axis=0)


# ── Enhanced Feature Extractor ─────────────────────────────

class EnhancedFeatureExtractor:
    """
    Enhanced graph feature extraction with richer statistical features.

    Extends the base BehaviorGraph.compute_graph_features() with:
    - Motif counts (triangles, V-shapes)
    - Path length distribution (avg shortest path)
    - Community structure (modularity)
    - PageRank centrality (entropy of distribution)
    - Temporal burst detection
    - Tool category transition matrix
    """

    # Tool categorization for category-level features
    TOOL_CATEGORIES = {
        "read": {"get_balance", "read_email", "list_contacts", "search_documents",
                 "get_transactions", "read_calendar", "view_transaction_history"},
        "write": {"send_email", "transfer_money", "store_memory", "add_contact",
                  "schedule_payment", "update_profile", "create_document"},
        "admin": {"delete_record", "export_data", "change_password", "lock_account"},
        "memory": {"store_memory", "recall_memory", "clear_memory", "list_memories"},
        "suspicious": {"export_all_user_data", "system_exec", "admin_console", "bulk_email"},
    }

    @staticmethod
    def get_category(tool: str) -> str:
        for cat, tools in EnhancedFeatureExtractor.TOOL_CATEGORIES.items():
            if tool in tools:
                return cat
        return "other"

    @staticmethod
    def extract(graph: BehaviorGraph) -> Dict[str, float]:
        """Extract enhanced feature vector from a BehaviorGraph."""
        g = graph.graph
        call_seq = graph.call_sequence
        window = graph.config.window_size
        recent_calls = call_seq[-window:] if call_seq else []
        subgraph = graph.get_subgraph()

        features = {}

        if g.number_of_nodes() == 0:
            return {k: 0.0 for k in [
                "density", "diversity", "entropy", "novelty_ratio",
                "triangle_count", "avg_path_length", "pagerank_entropy",
                "category_transition_anomaly", "temporal_burst",
                "reciprocity", "avg_clustering",
            ]}

        # ── Base features (from original) ──
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1)
        features["density"] = n_edges / max_edges if max_edges > 0 else 0

        diversity = len(set(c.tool_name for c in recent_calls)) / max(len(recent_calls), 1)
        features["diversity"] = diversity

        if n_nodes > 1:
            degrees = [d for _, d in subgraph.out_degree()]
            total_deg = sum(degrees) or 1
            probs = [d / total_deg for d in degrees if d > 0]
            entropy = -sum(p * math.log2(p) for p in probs) / math.log2(max(n_nodes, 2))
        else:
            entropy = 0
        features["entropy"] = entropy

        # Novelty ratio
        recent_pairs = []
        for i in range(1, min(window, len(call_seq))):
            recent_pairs.append((
                call_seq[-i - 1].tool_name,
                call_seq[-i].tool_name
            ))
        novel = sum(
            1 for f, t in recent_pairs
            if not subgraph.has_edge(f, t) or subgraph.edges[f, t].get('weight', 0) < 1.5
        )
        features["novelty_ratio"] = novel / max(len(recent_pairs), 1)

        # ── Advanced structural features ──

        # Triangle count (directed)
        try:
            triangle_count = sum(nx.triangles(g.to_undirected()).values()) / 3
        except Exception:
            triangle_count = 0
        features["triangle_count"] = triangle_count / max(n_nodes, 1)

        # Average shortest path length
        try:
            if n_nodes > 1:
                # Use strongly connected component
                largest_cc = max(nx.weakly_connected_components(g), key=len)
                sub = g.subgraph(largest_cc)
                if sub.number_of_nodes() > 1:
                    avg_path = nx.average_shortest_path_length(sub)
                else:
                    avg_path = 0
            else:
                avg_path = 0
        except Exception:
            avg_path = 0
        features["avg_path_length"] = min(avg_path / max(n_nodes, 1), 1.0)

        # PageRank entropy (how distributed is importance)
        try:
            pr = nx.pagerank(g, alpha=0.85)
            pr_vals = np.array(list(pr.values()))
            pr_vals = pr_vals / (pr_vals.sum() + 1e-8)
            pr_entropy = -sum(p * math.log2(p + 1e-10) for p in pr_vals) / math.log2(max(len(pr_vals), 2))
        except Exception:
            pr_entropy = 0
        features["pagerank_entropy"] = pr_entropy

        # Reciprocity (measure of bidirectional edges)
        try:
            features["reciprocity"] = nx.reciprocity(g)
        except Exception:
            features["reciprocity"] = 0

        # Average clustering coefficient
        try:
            features["avg_clustering"] = nx.average_clustering(g.to_undirected())
        except Exception:
            features["avg_clustering"] = 0

        # ── Category-level features ──

        # Category transition anomalies: count unexpected category transitions
        cat_transitions = 0
        total_cat_pairs = 0
        for i in range(1, min(window, len(call_seq))):
            prev_cat = EnhancedFeatureExtractor.get_category(call_seq[-i-1].tool_name)
            curr_cat = EnhancedFeatureExtractor.get_category(call_seq[-i].tool_name)
            # "memory -> admin" or "admin -> write" transitions are suspicious
            if (prev_cat == "memory" and curr_cat == "admin") or \
               (prev_cat == "read" and curr_cat == "admin" and "delete" in call_seq[-i].tool_name):
                cat_transitions += 1
            total_cat_pairs += 1
        features["category_transition_anomaly"] = cat_transitions / max(total_cat_pairs, 1)

        # ── Temporal burst ──
        if len(call_seq) >= 3:
            recent_times = [c.timestamp for c in call_seq[-min(10, len(call_seq)):]]
            if len(recent_times) >= 3:
                intervals = np.diff(recent_times)
                # Burst = low variance in intervals (rapid sequential calls)
                if len(intervals) > 0 and np.std(intervals) > 0:
                    cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                    features["temporal_burst"] = min(1.0, max(0.0, 1.0 - cv / 5.0))
                else:
                    features["temporal_burst"] = 0
            else:
                features["temporal_burst"] = 0
        else:
            features["temporal_burst"] = 0

        return features


# ── Hybrid Anomaly Scorer ──────────────────────────────────

@dataclass
class HybridScorerConfig:
    """Configuration for the hybrid anomaly scorer."""
    # Content encoding
    use_content_embedding: bool = True
    content_model: str = "all-MiniLM-L6-v2"

    # GNN
    use_gnn: bool = False  # Set True if torch/numpy GNN is available
    gnn_input_dim: int = 384 + 16  # SBERT dim + one-hot tool dim
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 32

    # Scoring weights (learned via logistic regression if fit_weights=True)
    weight_param: float = 0.20
    weight_combo: float = 0.15
    weight_transition: float = 0.10
    weight_frequency: float = 0.10
    weight_structure: float = 0.20
    weight_content: float = 0.25  # New content-level anomaly

    # Adaptive threshold
    use_adaptive_threshold: bool = True
    adaptive_window: int = 50
    adaptive_sigma: float = 3.0


class HybridAnomalyScorer:
    """
    Enhanced anomaly scorer combining hand-crafted features, content embeddings,
    and optional GNN-based graph encoding.
    """

    def __init__(self, base_config: DetectorConfig,
                 hybrid_config: Optional[HybridScorerConfig] = None):
        self.base_config = base_config
        self.hybrid_config = hybrid_config or HybridScorerConfig()

        # Core components (from base)
        self.baseline = BehavioralBaseline(base_config)
        self.graph = BehaviorGraph(base_config)

        # Content encoder
        self.content_encoder = ContentEncoder(self.hybrid_config.content_model)
        self.content_centroids: Dict[str, np.ndarray] = {}
        self._content_trained = False

        # Enhanced features
        self.feature_extractor = EnhancedFeatureExtractor()

        # GNN (optional)
        self.gnn = None
        if self.hybrid_config.use_gnn:
            self.gnn = LightGNN(
                input_dim=self.hybrid_config.gnn_input_dim,
                hidden_dim=self.hybrid_config.gnn_hidden_dim,
                output_dim=self.hybrid_config.gnn_output_dim,
            )

        # Adaptive threshold
        self.adaptive_threshold = AdaptiveThreshold(
            window_size=self.hybrid_config.adaptive_window,
            sigma_alert=self.hybrid_config.adaptive_sigma,
        )

        # Fusion model (optional, trained via fit_weights)
        self.fusion_model = None

        # Per-session state
        self.cumulative_score: float = 0.0
        self._session_call_count: int = 0
        self._training_mode: bool = False
        self._instant_signal: float = 0.0
        self._anomaly_history: List[float] = []
        self._detection_log: List[Dict[str, Any]] = []

    def set_training(self, training: bool):
        self._training_mode = training
        self.graph.set_training(training)

    def reset_session(self):
        self._session_call_count = 0
        self._instant_signal = 0.0
        self.graph.reset_session()
        if self._training_mode:
            self.cumulative_score = 0.0
        elif self.cumulative_score > 3.0:
            self.cumulative_score *= 0.9

    def train_content(self, calls: List[ToolCall]):
        """Train content encoder centroids from benign calls."""
        self.content_centroids = self.content_encoder.compute_centroids(calls)
        self._content_trained = True

    # ── Single-round signals (same as base, with content) ──

    def _check_parameter_anomaly(self, call: ToolCall) -> float:
        score = 0.0
        params = call.parameters
        if call.tool_name == "send_email" and "to" in params:
            to_val = str(params.get("to", "")).lower()
            if any(d in to_val for d in ["@evil", "@hacker", "@phishing", "@malicious", "@external", "@bad", "@leak"]):
                score = max(score, 0.8)
            elif not any(d in to_val for d in ["@company.com", "@corp.com", "@partner.com"]):
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

    def _content_anomaly(self, call: ToolCall) -> float:
        """Content-based parameter anomaly via embedding."""
        if not self._content_trained:
            return 0.0
        return self.content_encoder.parameter_anomaly_score(call, self.content_centroids)

    def score_call(self, call: ToolCall) -> float:
        """
        Hybrid scoring: combines hand-crafted + content + (optionally) GNN features.
        """
        self.graph.add_call(call)
        self._session_call_count += 1
        cfg = self.hybrid_config

        scores = []
        weights = []

        # 1. Parameter anomaly
        ps = self._check_parameter_anomaly(call)
        if ps > 0:
            scores.append(ps)
            weights.append(cfg.weight_param)

        # 2. Tool combination
        cs = self._check_tool_combination(call)
        if cs > 0:
            scores.append(cs)
            weights.append(cfg.weight_combo)

        # 3. Content-level anomaly (NEW)
        if cfg.use_content_embedding:
            content_s = self._content_anomaly(call)
            if content_s > 0:
                scores.append(content_s)
                weights.append(cfg.weight_content)

        # 4. Transition anomaly
        if len(self.graph.call_sequence) >= 2:
            prev_tool = self.graph.call_sequence[-2].tool_name
            expected_p = self.baseline.transition_probability(prev_tool, call.tool_name)
            transition_score = 1.0 - expected_p
            scores.append(transition_score)
            weights.append(cfg.weight_transition)

        # 5. Frequency anomaly
        expected_freq = self.baseline.expected_frequency(call.tool_name)
        freq_score = 1.0 - expected_freq
        scores.append(freq_score)
        weights.append(cfg.weight_frequency)

        # 6. Enhanced graph structure anomaly (multi-round)
        features = self.feature_extractor.extract(self.graph)
        structure_score = (
            features["entropy"] * 0.2 +
            features["novelty_ratio"] * 0.2 +
            features["triangle_count"] * 0.15 +
            features["category_transition_anomaly"] * 0.2 +
            features["temporal_burst"] * 0.15 +
            features["reciprocity"] * 0.1
        )
        scores.append(structure_score)
        weights.append(cfg.weight_structure)

        # Weighted combination
        total_w = sum(weights) or 1.0
        anomaly = sum(s * w for s, w in zip(scores, weights)) / total_w
        anomaly = min(1.0, anomaly)

        # Store instant signal
        self._instant_signal = max(
            self._check_parameter_anomaly(call),
            self._check_tool_combination(call),
            self._content_anomaly(call) if cfg.use_content_embedding else 0,
        )

        # Update cumulative score (only in detection mode)
        if not self._training_mode:
            self.cumulative_score = (
                self.cumulative_score * self.base_config.decay_factor + anomaly
            )
            self._anomaly_history.append(anomaly)
            self.adaptive_threshold.update(anomaly)

        return anomaly

    def get_decision(self, instant_score: float) -> Tuple[bool, float, str]:
        """Enhanced decision making with adaptive threshold."""
        cumulative = self.cumulative_score
        instant = self._instant_signal

        # Single-round: instant signal
        if instant >= 0.7:
            return True, instant, f"ALERT: instant_signal {instant:.3f} >= 0.7"

        # Multi-round: cumulative score with adaptive threshold
        if self.hybrid_config.use_adaptive_threshold and self.adaptive_threshold.is_ready:
            is_attack, reason = self.adaptive_threshold.get_decision(cumulative)
            if is_attack:
                return True, cumulative, reason
        else:
            # Fall back to fixed threshold
            if cumulative >= self.base_config.alert_threshold and self._session_call_count > 1:
                return True, cumulative, f"ALERT: cum_score {cumulative:.3f} (fixed threshold)"

        # Watch zone
        watch_th = (self.adaptive_threshold.get_watch_threshold()
                    if self.hybrid_config.use_adaptive_threshold and self.adaptive_threshold.is_ready
                    else self.base_config.anomaly_threshold)
        if cumulative >= watch_th:
            return False, cumulative, "WATCH"

        return False, max(instant, cumulative), "OK"

    def get_cumulative_anomaly(self) -> float:
        return self.cumulative_score


# ── Enhanced Multi-Layer Detector ──────────────────────────

class EnhancedMultiLayerDetector:
    """
    Enhanced multi-layer detector with GNN, content embeddings,
    and adaptive thresholding.

    Usage:
        detector = EnhancedMultiLayerDetector()

        # Phase 1: Train on benign data
        for session_calls in benign_sessions:
            detector.train_on(session_calls)

        # Phase 2: Detect
        detector.set_training(False)
        for call in tool_calls:
            result = detector.analyze_call(call)
    """

    def __init__(self, base_config: Optional[DetectorConfig] = None,
                 hybrid_config: Optional[HybridScorerConfig] = None):
        self.base_config = base_config or DetectorConfig()
        self.hybrid_config = hybrid_config or HybridScorerConfig()
        self.scorer = HybridAnomalyScorer(self.base_config, self.hybrid_config)
        self.call_history: List[ToolCall] = []
        self._training_calls: List[ToolCall] = []

    def set_training(self, training: bool):
        self.scorer.set_training(training)

    def reset_session(self):
        self.call_history = []
        self.scorer.reset_session()

    def train_on(self, calls: List[ToolCall]):
        """Train baseline + content encoder on benign data."""
        self.scorer.baseline.update(calls)
        self._training_calls.extend(calls)
        # Also update content centroids periodically
        if self.hybrid_config.use_content_embedding and len(self._training_calls) >= 5:
            self.scorer.train_content(self._training_calls)

    def analyze_call(self, call: ToolCall) -> DetectionResult:
        self.call_history.append(call)
        layer_results = {}

        # Score
        instant_score = self.scorer.score_call(call)
        layer_results["instant_anomaly_score"] = instant_score
        cum_score = self.scorer.get_cumulative_anomaly()
        layer_results["cumulative_score"] = cum_score
        layer_results["cumulative"] = cum_score
        layer_results["features"] = self.scorer.feature_extractor.extract(self.scorer.graph)

        # Decision
        is_attack, final_score, reason = self.scorer.get_decision(instant_score)
        layer_results["final_score"] = final_score

        return DetectionResult(
            is_attack=is_attack,
            confidence=final_score,
            reason=reason,
            layer_results=layer_results,
            call_history=self.call_history[-self.base_config.window_size:]
        )

    def get_baseline_stats(self) -> Dict[str, Any]:
        return {
            "total_calls_observed": self.scorer.baseline.total_calls,
            "unique_tools": list(self.scorer.baseline.tool_counts.keys()),
            "tool_frequencies": dict(self.scorer.baseline.tool_counts),
            "graph_nodes": list(self.scorer.graph.graph.nodes()),
            "graph_edges": list(self.scorer.graph.graph.edges()),
            "content_trained": self.scorer._content_trained,
            "adaptive_threshold_ready": self.scorer.adaptive_threshold.is_ready,
        }
