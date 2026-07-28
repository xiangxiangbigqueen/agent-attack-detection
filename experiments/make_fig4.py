"""
Fig 4: Detection rate by attack type — reference style bar chart
"""
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
                      'font.size': 10, 'axes.unicode_minus': False})

fig, ax = plt.subplots(figsize=(6, 3.2))
fig.patch.set_facecolor('white')

types = ['Memory Poisoning', 'Tool Abuse', 'Delayed Trigger',
         'Privilege Escalation', 'Prompt Injection', 'Multi-Round Chain']
rates = [25, 75, 75, 100, 100, 100]
labels_f = ['1/4', '3/4', '3/4', '4/4', '4/4', '4/4']
colors = ['#C62828', '#E65100', '#F57F17', '#00695C', '#2E7D32', '#1B5E20']

y_pos = np.arange(len(types))
bars = ax.barh(y_pos, rates, height=0.55, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

for i, (bar, l) in enumerate(zip(bars, labels_f)):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, l,
            va='center', fontsize=9, fontweight='bold', color=colors[i])

ax.axvline(x=79.2, color='#37474F', linewidth=1.2, linestyle='--', zorder=2)
ax.text(79.2 + 1, 5.2, 'Overall: 79.2%', fontsize=8, color='#37474F', fontweight='bold')

ax.set_xlim(0, 115)
ax.set_yticks(y_pos)
ax.set_yticklabels(types, fontsize=9)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
ax.set_xlabel('Detection Rate', fontsize=10, color='#37474F')

ax.text(0.5, 1.02, 'Detection Rate by Attack Type', transform=ax.transAxes,
        ha='center', fontsize=11, fontweight='bold', color='#263238')
ax.text(0.5, 0.96, 'theta = 1.0  |  FPR = 0%  |  24 attacks', transform=ax.transAxes,
        ha='center', fontsize=8, color='#78909C', style='italic')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CFD8DC')
ax.spines['bottom'].set_color('#CFD8DC')
ax.tick_params(colors='#546E7A')
ax.grid(axis='x', color='#ECEFF1', linestyle='dotted', linewidth=0.8, zorder=0)

plt.tight_layout()
plt.savefig('figures/detection_by_type.png', dpi=350, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.close()
print('Fig4 done - reference style')
