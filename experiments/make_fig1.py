"""
Fig 1: Pipeline architecture — reference style
"""
import os; os.environ['PYTHONIOENCODING'] = 'utf-8'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
                      'font.size': 9, 'axes.unicode_minus': False})

fig, ax = plt.subplots(figsize=(10, 2.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

C = {
    'blue':   ('#1565C0', '#BBDEFB'),
    'green':  ('#2E7D32', '#C8E6C9'),
    'orange': ('#E65100', '#FFE0B2'),
    'purple': ('#6A1B9A', '#E1BEE7'),
    'red':    ('#C62828', '#FFCDD2'),
    'amber':  ('#F57F17', '#FFF9C4'),
    'teal':   ('#00695C', '#B2DFDB'),
    'gray':   '#90A4AE', 'dark': '#37474F',
}

def rect(x, y, w, h, ec, bg, title, sub='', fs=10, fs_sub=7):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                         facecolor=bg, edgecolor=ec, lw=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x+w/2, y+h*0.60, title, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=ec)
    if sub:
        ax.text(x+w/2, y+h*0.22, sub, ha='center', va='center',
                fontsize=fs_sub, color='#546E7A', style='italic')

def ar(x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color=C['gray']))

Y = 0.28; H = 0.55; M = Y+H/2

stages = [
    (0.02, 0.11, 'Tool Call Stream', 'per session input', C['blue']),
    (0.15, 0.11, 'Behavior Graph', 'G=(V,E,W), decay=0.9', C['green']),
    (0.28, 0.11, 'Feature Extraction', '7 structural dims', C['orange']),
    (0.41, 0.12, 'Cumulative Scoring', 'EWMA: S=alphaS+f(x)', C['purple']),
    (0.55, 0.11, 'P95 Threshold', 'theta=P95(S_max train)', C['red']),
    (0.68, 0.11, 'Decision Logic', 'S >= theta ?', C['teal']),
]
for x, w, t, s, (ec, bg) in stages:
    rect(x, Y, w, H, ec, bg, t, s)
for i in range(5):
    ar(stages[i][0]+stages[i][1], stages[i+1][0], M)

ar(-0.025, 0.015, M)
ax.text(-0.028, M+0.14, 'Agent\nCalls', ha='right', va='center',
        fontsize=8, fontweight='bold', color=C['dark'])

fork_x = 0.84
ar(0.79, fork_x, M)
ax.plot([fork_x, fork_x], [0.82, 0.12], lw=1.2, color=C['gray'], zorder=1)

outs = [
    (0.82, 'ATTACK', 'alert + block', C['red']),
    (0.47, 'WATCH', 'manual review', C['amber']),
    (0.12, 'BENIGN', 'continue', C['teal']),
]
for oy, ol, osb, (ec, bg) in outs:
    ax.plot([fork_x, fork_x+0.006], [oy, oy], lw=1.0, color=C['gray'])
    ax.annotate('', xy=(fork_x+0.008, oy), xytext=(fork_x+0.004, oy),
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color=C['gray']))
    rect(fork_x+0.010, oy-0.065, 0.08, 0.13, ec, bg, ol, osb, fs=8.5, fs_sub=6)

cx1 = 0.41+0.06; cx2 = 0.15+0.055
ax.plot([cx1, cx1], [Y, 0.06], lw=0.8, color='#B0BEC5', ls='dashed')
ax.plot([cx2, cx1], [0.06, 0.06], lw=0.8, color='#B0BEC5', ls='dashed')
ax.plot([cx2, cx2], [0.06, Y], lw=0.8, color='#B0BEC5', ls='dashed')
ax.annotate('', xy=(cx2, Y-0.005), xytext=(cx2, 0.06),
            arrowprops=dict(arrowstyle='->', lw=0.8, color='#B0BEC5'))
ax.text((cx1+cx2)/2, 0.05, 'graph update per call',
        ha='center', va='top', fontsize=5.5, color=C['gray'], style='italic')

for lx, lw, lt, lc in [
    (0.02, 0.24, 'Layer 1-2: Input + Graph', C['green'][0]),
    (0.28, 0.11, 'Layer 3: Feature Extraction', C['orange'][0]),
    (0.41, 0.12, 'Layer 4: Cumulative Scoring', C['purple'][0]),
    (0.55, 0.24, 'Layer 5: Calibration + Decision', C['red'][0]),
]:
    ax.plot([lx, lx+lw], [0.87, 0.87], lw=3, color=lc, solid_capstyle='butt')
    ax.text(lx+lw/2, 0.88, lt, ha='center', va='bottom',
            fontsize=6.5, fontweight='bold', color=lc)

ax.set_title('Detection Pipeline Architecture', fontsize=13,
             fontweight='bold', pad=14, color='#263238')

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/architecture.png', dpi=400,
            bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print('Fig1 done - reference style')
