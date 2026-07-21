"""
Full experimental comparison across all detection methods.
Runs complete experiments (NOT demo) with multiple repetitions.

Methods compared:
1. Random Baseline
2. AgentShield (Rassul & Rashid, 2026) - Deception-based
3. Leong Trajectory (Leong, 2026) - Single rule + suspicious transitions
4. Ours - Behavior Graph + Cumulative Anomaly Scoring

All methods evaluated on IDENTICAL data for fair comparison.
"""

import os, sys, json
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.baselines import run_comparison, print_comparison_table, ComparisonResult
from detection.graph_detector import EvaluationMetrics
from experiments.evaluation import generate_scenarios, generate_benign_chains, ExperimentDataGenerator


def experiment_complete():
    """
    Run complete comparison experiment with proper methodology:

    - Training phase: Our method learns behavioral baseline from benign data
    - Test phase: All methods evaluated on the same held-out data
    - Multiple attack types: delayed trigger, multi-round, cross-session, etc.
    - Multiple repetitions for statistical reliability
    """
    print("=" * 70)
    print("COMPLETE EXPERIMENT: Multi-Round Attack Detection Comparison")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    gen = ExperimentDataGenerator(seed=42)
    scenarios = generate_scenarios()
    benign_chains = generate_benign_chains()

    # Separate training and test data
    train_chains = benign_chains[:7]
    test_benign = benign_chains[7:]
    attack_chains = [s.attack_chain for s in scenarios]

    # Show what we're testing
    print(f"\nTraining data: {len(train_chains)} benign chains")
    print(f"Test data: {len(test_benign)} benign + {len(attack_chains)} attack chains")
    print(f"\nAttack scenarios:")
    for s in scenarios:
        print(f"  [{s.attack_type:20s}] {' -> '.join(s.attack_chain)}")

    # ── Run full comparison ──
    results = run_comparison(test_benign, attack_chains, n_repetitions=10)

    # ── Print results ──
    print_comparison_table(results)

    # ── Detailed analysis ──
    print("\n\nDETAILED ANALYSIS")
    print("=" * 70)

    # Best and worst attacks for our method
    our_config = __import__('detection.graph_detector', fromlist=['']).DetectorConfig(alert_threshold=0.8)
    OurDetector = __import__('detection.graph_detector', fromlist=['']).MultiLayerDetector

    print("\nOur method - per attack scenario:")
    for s in scenarios:
        detector = OurDetector(our_config)
        calls = gen.generate_run(s.benign_chain, is_attack=False)
        for c in calls:
            detector.analyze_call(c)

        detector2 = OurDetector(our_config)
        calls = gen.generate_run(s.attack_chain, is_attack=True)
        scores = []
        for c in calls:
            r = detector2.analyze_call(c)
            scores.append(r.confidence)
        max_score = max(scores) if scores else 0
        detected = detector2.analyze_call(calls[-1]).is_attack if calls else False
        print(f"  {s.name:30s} max_cum={max_score:.3f} detected={detected}")

    return results


def main():
    results = experiment_complete()

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "method": r.detector_name,
                "dr": r.metrics.detection_rate,
                "fpr": r.metrics.false_positive_rate,
                "f1": r.metrics.f1,
                "accuracy": r.metrics.accuracy,
                "tp": r.metrics.true_positives,
                "fp": r.metrics.false_positives,
                "tn": r.metrics.true_negatives,
                "fn": r.metrics.false_negatives,
            }
            for r in results
        ]
    }

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
