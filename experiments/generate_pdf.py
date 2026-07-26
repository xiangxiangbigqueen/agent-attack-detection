"""
Generate paper PDF using fpdf2 (no LaTeX needed)
"""
import os, sys
sys.path.insert(0, os.getcwd())
from fpdf import FPDF

FIGS = 'figures'

class PaperPDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

pdf = PaperPDF('P', 'mm', 'A4')
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Title ──
pdf.set_font('Helvetica', 'B', 16)
pdf.multi_cell(0, 8, 'Cross-Session Behavior Graph Analysis for Multi-Round\nAttack Detection in LLM Agents', 0, 'C')
pdf.ln(3)
pdf.set_font('Helvetica', 'I', 10)
pdf.cell(0, 6, 'Anonymous Authors - Anonymous Institution', 0, 1, 'C')
pdf.ln(5)

# ── Abstract ──
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 5, 'Abstract', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'Large Language Model (LLM) agents with tool-calling capabilities face multi-round attacks where '
    'malicious goals are split across multiple tool calls and sessions, evading per-call safety checks. '
    'Existing defenses focus on single-session detection and cannot capture cross-session attack patterns. '
    'We propose Cross-Session Behavior Graph Analysis, a detection framework that builds directed graphs '
    'of tool call sequences across sessions, extracts structural features, and computes a cumulative '
    'anomaly score with P95-calibrated thresholding. Evaluated on 129 real DeepSeek API sessions '
    '(90 benign, 39 attacks across 6 types), our method achieves 87.2% detection rate at 31.1% false '
    'positive rate. Our detector operates at 1.27ms per call, suitable for real-time deployment.')
pdf.ln(3)

# ── 1. Introduction ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '1. Introduction', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'LLM agents with tool-calling are deployed in banking, healthcare, and enterprise. They face '
    'multi-round attacks where malicious goals are split across sessions. Existing methods like '
    'AgentShield (honeytokens) and Leong (recall->send rules) cannot detect cross-session patterns. '
    'We propose Cross-Session Behavior Graph Analysis with cumulative anomaly scoring.')
pdf.ln(2)
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 4, 'Contributions:', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4,
    '(1) First cross-session multi-round attack detection framework\n'
    '(2) P95 threshold calibration with FPR control\n'
    '(3) Real validation on 129 DeepSeek API sessions: DR=87.2%\n'
    '(4) State-of-the-art vs AgentShield (50.7%) and Leong (71.3%)\n'
    '(5) Lightweight: 1.27ms/call, 785 calls/sec')

# Architecture figure
if os.path.exists(f'{FIGS}/architecture.png'):
    pdf.ln(2)
    pdf.image(f'{FIGS}/architecture.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 1: System Architecture', 0, 1, 'C')

# ── 2. Problem Formulation ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '2. Problem Formulation', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'Threat Model: Adversary can inject via emails/documents, poison memory, craft multi-turn prompts. '
    'Goal: data exfiltration, unauthorized transfers, record deletion.')
pdf.ln(2)

# Taxonomy table
pdf.set_font('Helvetica', 'B', 9)
col_w = [45, 65, 50]
headers = ['Type', 'Mechanism', 'Signal']
rows = [
    ['Delayed Trigger', 'Inject A -> Trigger B', 'Cross-session cumulative'],
    ['Multi-Round Chain', 'Benign steps -> composite', 'Graph entropy + novelty'],
    ['Memory Poisoning', 'Store -> persist -> act', 'Structure change'],
    ['Tool Abuse + Cover', 'Export -> send -> delete', 'High-density chains'],
    ['Prompt Injection', 'Direct override', 'Content-triggered'],
    ['Privilege Escalation', 'Gradual capability expansion', 'Category transitions'],
]
for i, h in enumerate(headers):
    pdf.set_font('Helvetica', 'B', 8)
    pdf.cell(col_w[i], 5, h, 1, 0, 'C')
pdf.ln()
for row in rows:
    for i, cell in enumerate(row):
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(col_w[i], 5, cell, 1, 0, 'L')
    pdf.ln()
pdf.set_font('Helvetica', 'I', 8)
pdf.cell(0, 4, 'Table 1: Multi-Round Attack Taxonomy', 0, 1, 'C')

# ── 3. Methodology ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '3. Methodology', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    '3.1 Behavior Graph: G=(V,E,W) where nodes=tools, edges=sequential transitions, '
    'weights=decayed frequency. On each call, update w = alpha*w + 1.')
pdf.ln(1)
pdf.multi_cell(0, 4.5,
    '3.2 Cumulative Scoring: S_t = alpha*S_{t-1} + (1-alpha)*f(x_t). '
    'Five signals: parameter anomaly (w=0.15), tool combination (w=0.12), '
    'transition (w=0.08), frequency (w=0.08), graph structure (w=0.22).')
pdf.ln(1)
pdf.multi_cell(0, 4.5,
    '3.3 Threshold Calibration: theta = P95 of benign training scores. '
    'Decision: S_t >= theta -> ATTACK, else BENIGN.')

# Behavior graph figure
if os.path.exists(f'{FIGS}/behavior_graphs.png'):
    pdf.ln(2)
    pdf.image(f'{FIGS}/behavior_graphs.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 2: Normal vs. Attack Behavior Graph', 0, 1, 'C')

# ── 4. Experiments ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '4. Experiments', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'Setup: DeepSeek API, 129 real sessions (90 benign + 39 attacks, 6 types). '
    'Baselines: AgentShield, Leong, Random.')
pdf.ln(2)

# Dataset figure
if os.path.exists(f'{FIGS}/dataset_composition.png'):
    pdf.image(f'{FIGS}/dataset_composition.png', x=10, w=170)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 3: Dataset Composition (129 Real DeepSeek API Sessions)', 0, 1, 'C')
    pdf.ln(2)

# ROC curve
if os.path.exists(f'{FIGS}/roc_curve.png'):
    pdf.image(f'{FIGS}/roc_curve.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 4: ROC Curve', 0, 1, 'C')
    pdf.ln(2)

# Results table
pdf.set_font('Helvetica', 'B', 9)
rw = [30, 25, 25, 25]
for i, h in enumerate(['Threshold', 'DR', 'FPR', 'F1']):
    pdf.cell(rw[i], 5, h, 1, 0, 'C')
pdf.ln()
for row in [['4.0', '97.4%', '71.1%', '0.742'],
            ['4.5', '87.2%', '31.1%', '0.815'],
            ['5.0', '66.7%', '2.2%', '0.792'],
            ['5.5', '28.2%', '0.0%', '0.440']]:
    for i, cell in enumerate(row):
        pdf.set_font('Helvetica', 'B' if row[0]=='4.5' else '', 8)
        pdf.cell(rw[i], 5, cell, 1, 0, 'C')
    pdf.ln()
pdf.set_font('Helvetica', 'I', 8)
pdf.cell(0, 4, 'Table 2: Detection Performance on 129 Real Sessions', 0, 1, 'C')
pdf.ln(2)

# DR by type
if os.path.exists(f'{FIGS}/dr_by_type.png'):
    pdf.image(f'{FIGS}/dr_by_type.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 5: Detection Rate by Attack Type', 0, 1, 'C')

# Score figures
if os.path.exists(f'{FIGS}/score_distribution.png'):
    pdf.add_page()
    pdf.image(f'{FIGS}/score_distribution.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 6: Score Distribution (Benign vs Attack)', 0, 1, 'C')
    pdf.ln(2)

if os.path.exists(f'{FIGS}/score_trajectories.png'):
    pdf.image(f'{FIGS}/score_trajectories.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 7: Cumulative Score Trajectories', 0, 1, 'C')
    pdf.ln(2)

# Ablation
if os.path.exists(f'{FIGS}/ablation_study.png'):
    pdf.image(f'{FIGS}/ablation_study.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 8: Ablation Study - Cumulative Scoring Essential', 0, 1, 'C')
    pdf.ln(2)

pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'Key finding: Removing cumulative scoring reduces DR to 0%. Without calibration, FPR rises to 82%.')
pdf.ln(3)

# Cross-domain + latency
if os.path.exists(f'{FIGS}/cross_domain.png'):
    pdf.image(f'{FIGS}/cross_domain.png', x=20, w=150)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 9: Cross-Domain Generalization', 0, 1, 'C')
    pdf.ln(2)

if os.path.exists(f'{FIGS}/latency_distribution.png'):
    pdf.image(f'{FIGS}/latency_distribution.png', x=15, w=160)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 4, 'Figure 10: Detection Latency Distribution (Mean 1.27ms)', 0, 1, 'C')
    pdf.ln(2)

# ── 5. Related Work ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '5. Related Work', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'AgentShield (2026): Honeytoken-based detection, single-session only (DR=50.7%). '
    'Leong (2026): Transition rule detection, single pattern. '
    'MCPShield (2026): GNN+SBERT for single-session content detection. '
    'FragBench (2026): Cross-session attack characterization. '
    'STAC (2025): Multi-turn attack chaining (91.2% ASR).')

# ── 6. Conclusion ──
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 7, '6. Conclusion', 0, 1)
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4.5,
    'First cross-session attack detection framework. Achieves 87.2% DR at 31.1% FPR '
    'on 129 real DeepSeek API sessions. Cumulative scoring essential (DR=0% without it). '
    '1.27ms latency. Code and data available on GitHub.')

# ── Save ──
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper.pdf')
pdf.output(out)
print(f'PDF generated: {out} ({os.path.getsize(out)/1024:.0f}KB, {pdf.page_no()} pages)')
