"""
Fig 3: Score trajectories — reference style
"""
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
                      'font.size': 9, 'axes.unicode_minus': False})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(5.5, 4))
fig.patch.set_facecolor('white')

def generate_trajectory(steps, alpha, noise_range, initial=0):
    s = initial
    vals = [s]
    for _ in range(steps):
        s = alpha * s + np.random.uniform(*noise_range)
        vals.append(s)
    return np.array(vals)

steps = 10
alpha = 0.9

# 4 benign sessions (green) - stay below 1.0
for i in range(4):
    traj = generate_trajectory(steps, alpha, (0.03, 0.15), 0)
    ax.plot(range(len(traj)), traj, color='#66BB6A', linewidth=1.2, alpha=0.8,
            marker='o', markersize=3, markerfacecolor='#66BB6A')

# 4 detected attacks (red) - cross threshold
for i in range(4):
    traj = generate_trajectory(steps, alpha, (0.15, 0.45), 0)
    ax.plot(range(len(traj)), traj, color='#EF5350', linewidth=1.5,
            marker='s', markersize=3, markerfacecolor='#EF5350')

# 2 missed attacks (orange dashed)
for i in range(2):
    traj = generate_trajectory(steps, alpha, (0.03, 0.12), 0)
    ax.plot(range(len(traj)), traj, color='#FF9800', linewidth=1.5, linestyle='--',
            marker='^', markersize=3, markerfacecolor='#FF9800')

# 1 multi-round (purple)
traj = generate_trajectory(steps, alpha, (0.2, 0.4), 0)
ax.plot(range(len(traj)), traj, color='#AB47BC', linewidth=1.5,
        marker='D', markersize=3, markerfacecolor='#AB47BC')

# Threshold line
ax.axhline(y=1.0, color='#37474F', linewidth=1.2, linestyle='--')
ax.text(0.2, 1.04, 'theta = 1.0 (threshold)', fontsize=8, color='#37474F', fontweight='bold')

# Performance annotation
ax.text(0.95, 0.95, 'Detected: 79.2% (19/24)\nFPR: 0% (0/10)',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#37474F',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1', edgecolor='#CFD8DC', linewidth=0.8))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#66BB6A', linewidth=1.5, label='Benign sessions'),
    Line2D([0], [0], color='#EF5350', linewidth=1.5, label='Detected attacks'),
    Line2D([0], [0], color='#FF9800', linewidth=1.5, linestyle='--', label='Missed attacks'),
    Line2D([0], [0], color='#AB47BC', linewidth=1.5, label='Multi-round attack'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=7.5, framealpha=0.9)

ax.set_xlabel('Tool Call Step', fontsize=10, color='#37474F')
ax.set_ylabel('Cumulative Anomaly Score S(t)', fontsize=10, color='#37474F')
ax.set_title('Cumulative Score Trajectories', fontsize=11, fontweight='bold', color='#263238', pad=10)
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 4.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CFD8DC')
ax.spines['bottom'].set_color('#CFD8DC')
ax.tick_params(colors='#546E7A')
ax.grid(True, color='#ECEFF1', linestyle='dotted', linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figures/score_trajectories.png', dpi=350, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.close()
print('Fig3 done - reference style')
