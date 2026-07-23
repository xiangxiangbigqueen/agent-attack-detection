"""Detection module for agent attack detection."""

from detection.graph_detector import (
    DetectorConfig, BehavioralBaseline, BehaviorGraph,
    GraphAnomalyScorer, MultiLayerDetector, DetectionResult, EvaluationMetrics,
)

try:
    from detection.neural_detector import (
        EnhancedMultiLayerDetector, HybridScorerConfig,
        AdaptiveThreshold, ContentEncoder, EnhancedFeatureExtractor,
    )
except ImportError as e:
    # neural_detector has optional deps; warn but don't break
    import warnings
    warnings.warn(f"neural_detector not fully loaded: {e}")
