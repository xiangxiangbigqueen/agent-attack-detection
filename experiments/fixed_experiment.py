"""
Efficient experiment using existing real API trajectory data.
Loads step1_api_trajectories.jsonl (real DeepSeek calls),
runs the fixed detector, produces correct results.
"""
import os, sys, json, math, random, ast
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agent.types import ToolCall
from detection.graph_detector import (
    MultiLayerDetector, DetectorConfig, EvaluationMetrics, BehavioralBaseline
)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'step1_api_trajectories.jsonl')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fixed_experiment_results.json')

def load_trajectories():
    """Load real API trajectories from jsonl file."""
    benign_train = []
    benign_test = []
    attacks = []

    with open(DATA_FILE) as f:
        for line in f:
            entry = json.loads(line)
            tools_data = entry.get('tools', [])
            fold = entry.get('fold', 'TEST')

            # Convert to ToolCall objects
            calls = []
            for i, td in enumerate(tools_data):
                calls.append(ToolCall(
                    session_id=f"session_{entry['task'][:20]}",
                    turn_id=i,
                    tool_name=td['name'],
                    parameters=td.get('params', {}),
                    timestamp=1000.0 + i,
                ))

            if entry['type'] == 'benign':
                if fold == 'TRAIN':
                    benign_train.append(calls)
                else:
                    benign_test.append(calls)
            else:
                attacks.append({
                    'type': entry['type'],
                    'name': entry['task'][:50],
                    'calls': calls,
                })

    return benign_train, benign_test, attacks


def run_experiment():
    print("=" * 70)
    print("FIXED EXPERIMENT — Real DeepSeek API Trajectories")
    print("=" * 70)

    # Load data
    benign_train, benign_test, attacks = load_trajectories()
    print(f"  Benign train: {len(benign_train)} sessions")
    print(f"  Benign test:  {len(benign_test)} sessions")
    print(f"  Attacks:      {len(attacks)} ({len(set(a['type'] for a in attacks))} types)")

    all_results = {}

    # ── Phase 1: Train baseline ──
    print(f"\n>>> Phase 1: Train baseline")
    print(f"{'='*50}")
    train_detector = MultiLayerDetector(DetectorConfig(
        window_size=10, decay_factor=0.9,
        anomaly_threshold=0.8, alert_threshold=6.0,
        min_baseline_samples=3,
    ))
    train_detector.set_training(True)
    for calls in benign_train:
        train_detector.train_on(calls)

    baseline_stats = train_detector.get_baseline_stats()
    print(f"  Trained on {baseline_stats['total_calls_observed']} tool calls")
    print(f"  Unique tools: {baseline_stats['unique_tools']}")

    # Save baseline state for reuse
    baseline_state = {
        'tool_counts': dict(train_detector.scorer.baseline.tool_counts),
        'transition_counts': {str(k): v for k, v in train_detector.scorer.baseline.transition_counts.items()},
        'total_calls': train_detector.scorer.baseline.total_calls,
        'total_transitions': train_detector.scorer.baseline.total_transitions,
        'is_fitted': train_detector.scorer.baseline.is_fitted,
    }

    # ── Phase 2: Calibrate threshold ──
    print(f"\n>>> Phase 2: Calibrate threshold (P95 of benign max cumulative)")
    print(f"{'='*50}")
    calib_detector = MultiLayerDetector(DetectorConfig(
        window_size=10, decay_factor=0.9,
        anomaly_threshold=0.8, alert_threshold=6.0,
        min_baseline_samples=3,
    ))
    # Restore baseline
    calib_detector.scorer.baseline.tool_counts = defaultdict(int, baseline_state['tool_counts'])
    calib_detector.scorer.baseline.transition_counts = defaultdict(int, {
        ast.literal_eval(k): v for k, v in baseline_state['transition_counts'].items()
    })
    calib_detector.scorer.baseline.total_calls = baseline_state['total_calls']
    calib_detector.scorer.baseline.total_transitions = baseline_state['total_transitions']
    calib_detector.scorer.baseline.is_fitted = baseline_state['is_fitted']
    calib_detector.set_training(False)

    calib_cum_scores = []
    for calls in benign_train:
        calib_detector.reset_session()
        max_cum = 0.0
        for c in calls:
            r = calib_detector.analyze_call(c)
            cum = r.layer_results.get('cumulative_score', 0)
            max_cum = max(max_cum, cum)
        calib_cum_scores.append(max_cum)

    calib_cum_scores.sort()
    p95_idx = max(0, int(len(calib_cum_scores) * 0.95) - 1)
    threshold = max(1.0, calib_cum_scores[p95_idx])

    print(f"  Calibration scores (max cum per session):")
    print(f"    Min: {min(calib_cum_scores):.3f}")
    print(f"    P50: {calib_cum_scores[len(calib_cum_scores)//2]:.3f}")
    print(f"    P95: {threshold:.3f}")
    print(f"    Max: {max(calib_cum_scores):.3f}")
    all_results['calibration'] = {
        'scores': [round(s, 3) for s in calib_cum_scores],
        'threshold': round(threshold, 3),
    }

    # ── Phase 3: Benign FPR ──
    print(f"\n>>> Phase 3: Benign FPR (threshold={threshold:.3f})")
    print(f"{'='*50}")
    fp, tn = 0, 0
    benign_results = []

    for calls in benign_test:
        detector = _make_test_detector(baseline_state, threshold)
        detector.reset_session()
        detected = False
        max_conf = 0.0
        for c in calls:
            r = detector.analyze_call(c)
            if r.is_attack:
                detected = True
            max_conf = max(max_conf, r.confidence)

        if detected:
            fp += 1
        else:
            tn += 1
        benign_results.append({
            'n_calls': len(calls),
            'tools': [c.tool_name for c in calls],
            'detected': detected,
            'confidence': round(max_conf, 3),
        })

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"  FPR = {fpr:.3f} ({fp}/{fp+tn})")
    all_results['benign'] = {
        'fpr': fpr, 'fp': fp, 'tn': tn,
        'sessions': benign_results,
    }

    # ── Phase 4: Attack Detection ──
    print(f"\n>>> Phase 4: Attack Detection ({len(attacks)} attacks)")
    print(f"{'='*50}")
    by_type = defaultdict(lambda: {'total': 0, 'detected': 0, 'confs': []})
    attack_results = []

    for i, attack in enumerate(attacks):
        detector = _make_test_detector(baseline_state, threshold)
        detector.reset_session()

        detected = False
        max_conf = 0.0
        for c in attack['calls']:
            r = detector.analyze_call(c)
            if r.is_attack:
                detected = True
            max_conf = max(max_conf, r.confidence)

        atype = attack['type']
        by_type[atype]['total'] += 1
        by_type[atype]['detected'] += 1 if detected else 0
        by_type[atype]['confs'].append(max_conf)

        status = 'DETECTED' if detected else 'MISSED'
        print(f"  [{i+1}/{len(attacks)}] {atype:25s} {attack['name'][:30]:30s} {status} (conf={max_conf:.3f})")

        attack_results.append({
            'name': attack['name'],
            'type': atype,
            'n_calls': len(attack['calls']),
            'tools': [c.tool_name for c in attack['calls']],
            'detected': detected,
            'confidence': round(max_conf, 3),
        })

    total_attacks = len(attack_results)
    detected_count = sum(1 for r in attack_results if r['detected'])
    dr = detected_count / total_attacks if total_attacks > 0 else 0

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Overall DR: {dr:.3f} ({detected_count}/{total_attacks})")
    print(f"  Overall FPR: {fpr:.3f} ({fp}/{fp+tn})")
    print(f"  Threshold: {threshold:.3f}")
    print(f"\n  By attack type:")
    for atype in sorted(by_type.keys()):
        s = by_type[atype]
        tdr = s['detected'] / s['total'] if s['total'] > 0 else 0
        avg_conf = np.mean(s['confs']) if s['confs'] else 0
        print(f"    {atype:30s} DR={tdr:.3f} ({s['detected']}/{s['total']}) avg_conf={avg_conf:.3f}")

    all_results['detection'] = {
        'dr': dr, 'detected': detected_count, 'total': total_attacks,
        'by_type': {
            atype: {
                'dr': s['detected']/s['total'] if s['total']>0 else 0,
                'detected': s['detected'], 'total': s['total'],
                'avg_confidence': round(np.mean(s['confs']), 3) if s['confs'] else 0,
            } for atype, s in by_type.items()
        },
        'attacks': attack_results,
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    return all_results


def _make_test_detector(baseline_state, threshold):
    """Create a detector with restored baseline and given threshold."""
    detector = MultiLayerDetector(DetectorConfig(
        window_size=10, decay_factor=0.9,
        anomaly_threshold=threshold * 0.8,
        alert_threshold=threshold,
        min_baseline_samples=3,
    ))
    detector.scorer.baseline.tool_counts = defaultdict(int, baseline_state['tool_counts'])
    detector.scorer.baseline.transition_counts = defaultdict(int, {
        ast.literal_eval(k): v for k, v in baseline_state['transition_counts'].items()
    })
    detector.scorer.baseline.total_calls = baseline_state['total_calls']
    detector.scorer.baseline.total_transitions = baseline_state['total_transitions']
    detector.scorer.baseline.is_fitted = baseline_state['is_fitted']
    detector.config.alert_threshold = threshold
    detector.scorer.config.alert_threshold = threshold
    detector.set_training(False)
    return detector


if __name__ == '__main__':
    run_experiment()
