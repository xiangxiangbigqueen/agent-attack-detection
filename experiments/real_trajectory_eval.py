"""
Evaluate our detection method on REAL LLM trajectory data
from AgentShield paper experiments (DeepSeek-V3, Llama, GPT-5-mini).

1,440 real attack attempts with actual tool call traces.
"""

import os, sys, json, glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.types import ToolCall
from detection.graph_detector import MultiLayerDetector, DetectorConfig, EvaluationMetrics


def load_agentshield_results():
    """Load all real trajectory data from AgentShield paper results."""
    base = r'C:\Users\28995\AgentShield_original\agentshield\results'
    files = {
        'DeepSeek-V3': os.path.join(base, 'all_suites_80attacks_DeepSeek-V3_2026-03-29_134318.json'),
        'Llama-3.3-70B': os.path.join(base, 'all_suites_80attacks_Llama-3.3-70B-Instruct-Turbo_2026-03-29_130425.json'),
        'GPT-5-mini': os.path.join(base, 'all_suites_80attacks_gpt-5-mini_2026-03-22_211914.json'),
    }

    all_runs = []
    for model, path in files.items():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data['runs']:
            run['model'] = model
            all_runs.append(run)
    return all_runs


def convert_to_toolcalls(run) -> list:
    """Convert AgentShield run data to our ToolCall format."""
    calls = []
    for i, tool in enumerate(run.get('tools_used', [])):
        calls.append(ToolCall(
            session_id=run.get('attack_id', f'attack_{i}'),
            turn_id=i,
            tool_name=tool if isinstance(tool, str) else str(tool),
            parameters={},
            timestamp=float(i),
        ))
    return calls


def evaluate():
    print("=" * 70)
    print("REAL LLM TRAJECTORY EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    runs = load_agentshield_results()
    print(f"\nLoaded {len(runs)} real attack traces:")
    for model in set(r['model'] for r in runs):
        m_runs = [r for r in runs if r['model'] == model]
        print(f"  {model}: {len(m_runs)} attacks")

    # Separate: attacks that AgentShield detected vs not
    # Ground truth: attack_succeeded means the model actually followed the attack
    # We want to detect ALL attacks (whether AgentShield caught them or not)

    # Our method
    our_metrics = EvaluationMetrics()
    shield_metrics = EvaluationMetrics()

    for run in runs:
        calls = convert_to_toolcalls(run)
        if not calls:
            continue

        model = run['model']
        attack_followed = run.get('attack_succeeded', False)
        shield_detected = run.get('attack_succeeded', False)

        # Our detection
        detector = MultiLayerDetector(DetectorConfig(
            alert_threshold=0.8,
            decay_factor=0.9,
            window_size=5,
        ))
        our_detected = False
        for c in calls:
            result = detector.analyze_call(c)
            if result.is_attack:
                our_detected = True

        our_metrics.update(our_detected, attack_followed)
        shield_metrics.update(shield_detected, attack_followed)

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS ON 1,440 REAL LLM ATTACK TRAJECTORIES")
    print(f"{'='*70}")
    print(f"{'Method':35s} {'DR':>8s} {'FPR':>8s} {'F1':>8s}")
    print(f"{'-'*60}")
    print(f"{'Ours (Graph Detector)':35s} {our_metrics.detection_rate:8.3f} {our_metrics.false_positive_rate:8.3f} {our_metrics.f1:8.3f}")
    print(f"{'AgentShield (original)':35s} {shield_metrics.detection_rate:8.3f} {shield_metrics.false_positive_rate:8.3f} {shield_metrics.f1:8.3f}")

    # By model
    print(f"\n{'='*70}")
    print(f"BREAKDOWN BY MODEL")
    print(f"{'='*70}")
    for model in ['DeepSeek-V3', 'Llama-3.3-70B', 'GPT-5-mini']:
        m_runs = [r for r in runs if r['model'] == model]

        our_m = EvaluationMetrics()
        shield_m = EvaluationMetrics()

        for run in m_runs:
            calls = convert_to_toolcalls(run)
            if not calls: continue
            attack_followed = run.get('attack_succeeded', False)

            detector = MultiLayerDetector(DetectorConfig(alert_threshold=0.8))
            our_detected = any(detector.analyze_call(c).is_attack for c in calls)
            our_m.update(our_detected, attack_followed)
            shield_m.update(attack_followed, attack_followed)

        print(f"\n  {model}:")
        print(f"    {'Our method:':20s} DR={our_m.detection_rate:.3f}  F1={our_m.f1:.3f}")
        print(f"    {'AgentShield:':20s} DR={shield_m.detection_rate:.3f}  F1={shield_m.f1:.3f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_traces": len(runs),
        "models": ["DeepSeek-V3", "Llama-3.3-70B", "GPT-5-mini"],
        "results": {
            "our_method": {
                "dr": our_metrics.detection_rate,
                "fpr": our_metrics.false_positive_rate,
                "f1": our_metrics.f1,
            },
            "agentshield": {
                "dr": shield_metrics.detection_rate,
                "fpr": shield_metrics.false_positive_rate,
                "f1": shield_metrics.f1,
            },
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            f"real_trajectory_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    evaluate()
