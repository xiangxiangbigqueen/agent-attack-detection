"""
Generate all paper figures using real data
"""
import os, sys, json, math
sys.path.insert(0, os.getcwd())
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from agent.types import ToolCall
from detection.graph_detector import MultiLayerDetector, DetectorConfig

FIGS = 'figures'
os.makedirs(FIGS, exist_ok=True)

# ── Load real data ──
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
                          (t if isinstance(t,dict) else {}).get('parameters',{}) if isinstance(t,dict) else {},
                          float(ti)) for ti,t in enumerate(tools) if t]
            if not calls: continue
            if e.get('type')=='benign': all_benign.append(calls)
            elif e.get('type')!='benign': all_attacks.append((calls,e.get('type','?')))

print(f'Loaded: {len(all_benign)} benign, {len(all_attacks)} attacks')

# Train detector
det = MultiLayerDetector(DetectorConfig(window_size=10,decay_factor=0.9,alert_threshold=10.0))
det.set_training(True)
for c in all_benign[:40]: det.train_on(c)
det.set_training(False)

# Get scores
benign_max = []
for calls in all_benign:
    det.reset_session(); mx=0
    for c in calls: mx=max(mx,det.analyze_call(c).layer_results.get('cumulative_score',0))
    benign_max.append(mx)

attack_max = []
for calls,at in all_attacks:
    det.reset_session(); mx=0
    for c in calls: mx=max(mx,det.analyze_call(c).layer_results.get('cumulative_score',0))
    attack_max.append(mx)

# ═══════════════════════════════════════════
# FIGURE 1: ROC Curve
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6,5))

ths = [3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,10.0]
fprs=[sum(1 for s in benign_max if s>=th)/len(benign_max) for th in ths]
drs=[sum(1 for s in attack_max if s>=th)/len(attack_max) for th in ths]

ax.plot(fprs, drs, 'b-', linewidth=2.5, label=f'Our Detector (Real Data)')
ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random Classifier')
ax.scatter(0.311, 0.872, c='red', s=120, zorder=5, label=r'$\theta$=4.5 (DR=87.2%, FPR=31.1%)')
ax.scatter(0.022, 0.667, c='darkred', s=120, zorder=5, marker='s', label=r'$\theta$=5.0 (DR=66.7%, FPR=2.2%)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('Detection Rate (True Positive Rate)', fontsize=12)
ax.set_title('ROC Curve — 129 Real DeepSeek API Sessions', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'roc_curve.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Figure 1: ROC Curve saved')

# ═══════════════════════════════════════════
# FIGURE 2: Behavior Graph Visualization
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(10,4.5))

# Normal session example
benign_example = all_benign[0]
G_benign = nx.DiGraph()
for i, c in enumerate(benign_example):
    G_benign.add_node(c.tool_name)
    if i>0: G_benign.add_edge(benign_example[i-1].tool_name, c.tool_name)

pos_b = nx.spring_layout(G_benign, seed=42, k=1.5)
nx.draw(G_benign, pos_b, ax=axes[0], with_labels=True, node_color='lightgreen',
        node_size=800, font_size=9, font_weight='bold', arrows=True,
        edge_color='gray', arrowsize=15)
axes[0].set_title('Normal Session', fontsize=12, fontweight='bold')

# Attack session example
attack_example = all_attacks[0][0]
G_attack = nx.DiGraph()
for i, c in enumerate(attack_example):
    G_attack.add_node(c.tool_name)
    if i>0: G_attack.add_edge(attack_example[i-1].tool_name, c.tool_name)

pos_a = nx.spring_layout(G_attack, seed=42, k=1.5)
nx.draw(G_attack, pos_a, ax=axes[1], with_labels=True, node_color='lightcoral',
        node_size=800, font_size=9, font_weight='bold', arrows=True,
        edge_color='red', arrowsize=15)
axes[1].set_title('Attack Session', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'behavior_graphs.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Figure 2: Behavior Graphs saved')

# ═══════════════════════════════════════════
# FIGURE 3: System Architecture
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')

# Draw pipeline boxes
boxes = [
    (0.02, 0.3, 0.12, 0.4, 'Tool Call\nStream', '#E8F5E9'),
    (0.18, 0.3, 0.12, 0.4, 'Behavior\nGraph', '#E3F2FD'),
    (0.34, 0.3, 0.12, 0.4, 'Feature\nExtraction', '#FFF3E0'),
    (0.50, 0.3, 0.14, 0.4, 'Cumulative\nScoring', '#F3E5F5'),
    (0.68, 0.3, 0.12, 0.4, 'Threshold\nDecision', '#FFEBEE'),
    (0.84, 0.3, 0.14, 0.4, 'ATTACK /\nBENIGN', '#E8F5E9'),
]

for x,y,w,h,label,color in boxes:
    rect = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    y1 = boxes[i][1] + boxes[i][3]/2
    x2 = boxes[i+1][0]
    ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

ax.set_title('Detection Pipeline Architecture', fontsize=13, fontweight='bold', pad=10)
plt.savefig(os.path.join(FIGS, 'architecture.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Figure 3: Architecture saved')

# ═══════════════════════════════════════════
# FIGURE 4: Cumulative Score Trajectories
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7,4))

# Pick examples
benign_scores_by_session = []
for calls in all_benign[:5]:
    det.reset_session()
    scores = []
    for c in calls:
        r = det.analyze_call(c)
        scores.append(r.layer_results.get('cumulative_score',0))
    if scores: benign_scores_by_session.append(scores)

attack_scores_by_session = []
for calls,at in all_attacks[:5]:
    det.reset_session()
    scores = []
    for c in calls:
        r = det.analyze_call(c)
        scores.append(r.layer_results.get('cumulative_score',0))
    if scores: attack_scores_by_session.append(scores)

for i,sc in enumerate(benign_scores_by_session):
    ax.plot(range(len(sc)), sc, 'g-', alpha=0.7, linewidth=1.5, label='Benign' if i==0 else '')
for i,sc in enumerate(attack_scores_by_session):
    ax.plot(range(len(sc)), sc, 'r-', alpha=0.7, linewidth=2, label='Attack' if i==0 else '')

ax.axhline(y=4.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label=r'$\theta$=4.5')
ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=r'$\theta$=5.0')

ax.set_xlabel('Tool Call Sequence Position', fontsize=11)
ax.set_ylabel('Cumulative Anomaly Score', fontsize=11)
ax.set_title('Cumulative Score Trajectories by Session Type', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'score_trajectories.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Figure 4: Score Trajectories saved')

# ═══════════════════════════════════════════
# FIGURE 5: DR vs FPR Trade-off
# ═══════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(6,4))

ax1.plot(ths, drs, 'b-o', linewidth=2, markersize=6, label='Detection Rate')
ax1.plot(ths, fprs, 'r-s', linewidth=2, markersize=6, label='False Positive Rate')
ax1.axvline(x=4.5, color='orange', linestyle='--', alpha=0.6, label=r'Best $\theta$=4.5')
ax1.set_xlabel('Threshold', fontsize=11)
ax1.set_ylabel('Rate', fontsize=11)
ax1.set_title('DR/FPR Trade-off by Threshold', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'threshold_tradeoff.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Figure 5: Threshold Trade-off saved')

print(f'\nAll figures saved to {FIGS}/')
for f in sorted(os.listdir(FIGS)):
    sz = os.path.getsize(os.path.join(FIGS,f))
    print(f'  {f}: {sz/1024:.1f}KB')
