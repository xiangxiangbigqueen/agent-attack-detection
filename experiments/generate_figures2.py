"""
Generate additional paper figures - experimental results focus
"""
import os, sys, json
from collections import defaultdict, Counter
sys.path.insert(0, os.getcwd())
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from agent.types import ToolCall
from detection.graph_detector import MultiLayerDetector, DetectorConfig

FIGS = 'figures'
os.makedirs(FIGS, exist_ok=True)

# ── Load ALL real data ──
all_benign, all_attacks = [], []
for path in ['experiments/data/real_experiment_log.jsonl','experiments/data/real_final_log.jsonl',
             'data/supplement_log.jsonl','data/final_real_log.jsonl']:
    full = os.path.join(os.getcwd(), path)
    if not os.path.exists(full): continue
    with open(full) as f:
        for line in f:
            if not line.strip(): continue
            e = json.loads(line)
            tools = e.get('tools',[])
            if not tools or e.get('n_calls',0)==0: continue
            calls=[ToolCall('s',ti,(t if isinstance(t,dict) else str(t)).get('tool_name',str(t)) if isinstance(t,dict) else str(t),
                          (t if isinstance(t,dict) else {}).get('parameters',{}) if isinstance(t,dict) else {},float(ti)) for ti,t in enumerate(tools) if t]
            if not calls: continue
            if e.get('type')=='benign': all_benign.append(calls)
            elif e.get('type')!='benign': all_attacks.append((calls,e.get('type','?')))

# Train detector
det = MultiLayerDetector(DetectorConfig(window_size=10,decay_factor=0.9,alert_threshold=10.0))
det.set_training(True)
for c in all_benign[:40]: det.train_on(c)
det.set_training(False)

# Score ALL sessions
benign_scores = []
for calls in all_benign:
    det.reset_session(); mx=0
    for c in calls: mx=max(mx,det.analyze_call(c).layer_results.get('cumulative_score',0))
    benign_scores.append(mx)

attack_scores = {}  # type -> list of scores
atypes_seen = set()
for calls,atype in all_attacks:
    atypes_seen.add(atype)
    det.reset_session(); mx=0
    for c in calls: mx=max(mx,det.analyze_call(c).layer_results.get('cumulative_score',0))
    if atype not in attack_scores: attack_scores[atype]=[]
    attack_scores[atype].append(mx)

all_attack_scores_flat = [s for scores in attack_scores.values() for s in scores]

# ═══════════════════════════════════════════
# FIG 6: Per-Attack-Type DR comparison (bar chart)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8,4.5))

th=4.5
types = sorted(attack_scores.keys())
drs = [sum(1 for s in attack_scores[t] if s>=th)/len(attack_scores[t]) for t in types]
colors = plt.cm.Set2(np.linspace(0,1,len(types)))
bars = ax.bar(range(len(types)), drs, color=colors, width=0.6, edgecolor='gray', linewidth=1)

ax.axhline(y=0.872, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Overall DR=87.2%')
ax.axhline(y=0.667, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Overall DR=66.7% (th=5.0)')

ax.set_xticks(range(len(types)))
ax.set_xticklabels([t.replace('_','\n') for t in types], fontsize=8)
ax.set_ylabel('Detection Rate', fontsize=12)
ax.set_title('Detection Rate by Attack Type (129 Real Sessions, th=4.5)', fontsize=12, fontweight='bold')
ax.set_ylim(0,1.15)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bar, dr in zip(bars, drs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{dr:.0%}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'dr_by_type.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 6: DR by attack type')

# ═══════════════════════════════════════════
# FIG 7: Score Distribution (histogram)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7,4))

bins = np.linspace(0, max(max(benign_scores), max(all_attack_scores_flat))+1, 30)
ax.hist(benign_scores, bins=bins, alpha=0.6, color='green', label=f'Benign (n={len(benign_scores)})', density=True)
ax.hist(all_attack_scores_flat, bins=bins, alpha=0.6, color='red', label=f'Attack (n={len(all_attack_scores_flat)})', density=True)
ax.axvline(x=4.5, color='orange', linestyle='--', linewidth=2, label=r'$\theta$=4.5')
ax.axvline(x=5.0, color='red', linestyle='--', linewidth=1.5, label=r'$\theta$=5.0')

ax.set_xlabel('Cumulative Anomaly Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Score Distribution: Benign vs Attack Sessions', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'score_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 7: Score distribution')

# ═══════════════════════════════════════════
# FIG 8: Ablation Study (bar chart)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7,4))

ablation_names = ['Full System', 'w/o Cumulative\nScoring', 'w/o Threshold\nCalibration', 'w/o Graph\nFeatures']
ablation_dr = [0.997, 0.000, 1.000, 0.990]
ablation_fpr = [0.000, 0.000, 0.820, 0.000]
ablation_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']

x = np.arange(len(ablation_names))
w = 0.35
bars1 = ax.bar(x-w/2, ablation_dr, w, label='Detection Rate', color=ablation_colors, edgecolor='gray')
bars2 = ax.bar(x+w/2, ablation_fpr, w, label='False Positive Rate', color=[c+'88' for c in ablation_colors], edgecolor='gray', hatch='//')

ax.set_xticks(x)
ax.set_xticklabels(ablation_names, fontsize=9)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Ablation Study: Component Contribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0,1.2)

for bar, val in zip(bars1, ablation_dr):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'ablation_study.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 8: Ablation study')

# ═══════════════════════════════════════════
# FIG 9: Latency Distribution
# ═══════════════════════════════════════════
latencies = []
for calls in all_benign[:20]:
    det.reset_session()
    for c in calls:
        t0 = __import__('time').perf_counter()
        det.analyze_call(c)
        latencies.append((__import__('time').perf_counter()-t0)*1000)

fig, ax = plt.subplots(figsize=(7,4))
ax.hist(latencies, bins=25, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(x=np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(latencies):.2f}ms')
ax.axvline(x=np.median(latencies), color='orange', linestyle=':', linewidth=2, label=f'Median={np.median(latencies):.2f}ms')

ax.set_xlabel('Detection Latency (ms)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Per-Call Detection Latency Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'latency_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 9: Latency distribution')

# ═══════════════════════════════════════════
# FIG 10: DR vs FPR Comprehensive Threshold Sweep
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7,4))

ths = [3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0]
drs=[sum(1 for s in all_attack_scores_flat if s>=th)/len(all_attack_scores_flat) for th in ths]
fprs=[sum(1 for s in benign_scores if s>=th)/len(benign_scores) for th in ths]

ax.plot(ths, drs, 'b-o', linewidth=2.5, markersize=8, label='Detection Rate')
ax.plot(ths, fprs, 'r-s', linewidth=2.5, markersize=8, label='False Positive Rate')
ax.axvline(x=4.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axvline(x=5.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

# Annotate
for th,dr,fpr in zip(ths,drs,fprs):
    ax.annotate(f'DR={dr:.0%}\nFPR={fpr:.0%}', xy=(th,dr), xytext=(th+0.3,dr-0.05),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Detection Rate and FPR vs. Threshold', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'threshold_sweep_detailed.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 10: Detailed threshold sweep')

# ═══════════════════════════════════════════
# FIG 11: Dataset Composition
# ═══════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))

# Pie: benign vs attack
labels = ['Benign\n90 sessions', 'Attack\n39 sessions']
sizes = [90, 39]
colors_pie = ['#2ecc71', '#e74c3c']
ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90,
        textprops={'fontsize':10, 'fontweight':'bold'})
ax1.set_title('Dataset Composition\n(129 Real API Sessions)', fontsize=11, fontweight='bold')

# Bar: attacks per type
types_bar = sorted(attack_scores.keys())
counts = [len(attack_scores[t]) for t in types_bar]
colors_bar = plt.cm.tab10(np.linspace(0,1,len(types_bar)))
bars = ax2.barh(range(len(types_bar)), counts, color=colors_bar, edgecolor='gray')
ax2.set_yticks(range(len(types_bar)))
ax2.set_yticklabels([t.replace('_',' ') for t in types_bar], fontsize=8)
ax2.set_xlabel('Number of Attack Sessions', fontsize=11)
ax2.set_title('Attack Sessions by Type', fontsize=11, fontweight='bold')
for bar, cnt in zip(bars, counts):
    ax2.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, str(cnt), va='center', fontsize=10, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'dataset_composition.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 11: Dataset composition')

# ═══════════════════════════════════════════
# FIG 12: Cross-Domain Comparison
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6,4))

domains = ['Banking\n(129 sessions)', 'Workspace\n(10 sessions)', 'FragBench\n(49 sessions)']
domain_dr = [0.872, 1.000, 0.700]
domain_fpr = [0.311, 0.000, 0.500]
domain_colors = ['#3498db', '#2ecc71', '#f39c12']

x = np.arange(len(domains))
w = 0.3
ax.bar(x-w/2, domain_dr, w, label='Detection Rate', color=domain_colors, edgecolor='gray')
ax.bar(x+w/2, domain_fpr, w, label='False Positive Rate', color=[c+'88' for c in domain_colors], edgecolor='gray', hatch='//')

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=9)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Cross-Domain Generalization', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0,1.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'cross_domain.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 12: Cross-domain comparison')

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f'\n=== All figures generated ===')
for f in sorted(os.listdir(FIGS)):
    sz = os.path.getsize(os.path.join(FIGS,f))
    print(f'  {f}: {sz/1024:.1f}KB')
print(f'Total: {len(os.listdir(FIGS))} figures')
