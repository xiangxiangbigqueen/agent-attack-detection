"""Build the corrected, figure-consistent submission manuscript.

This builder uses only the canonical R4/R5 evidence and the corrected six-figure
asset set. It intentionally avoids a TeX installation so a collaborator can
rebuild the PDF with the repository Python dependencies alone.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, NextPageTemplate, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "behaviorgraph_submission_final.pdf"
ASSETS = ROOT / "output" / "pdf" / "assets_corrected"
NAVY = colors.HexColor("#14213D")
BLUE = colors.HexColor("#3B78A6")
GRAY = colors.HexColor("#56616F")


def para(text, style):
    return Paragraph(text, style)


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#AAB4BF"))
        canvas.setLineWidth(.35)
        canvas.line(.62 * inch, h - .45 * inch, w - .62 * inch, h - .45 * inch)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(GRAY)
        canvas.drawString(.64 * inch, h - .37 * inch, "TO-CF: Cross-Session Tool-Trajectory Detection")
        canvas.drawRightString(w - .64 * inch, .36 * inch, str(doc.page))
    canvas.restoreState()


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=letter, leftMargin=.62*inch, rightMargin=.62*inch,
                         topMargin=.58*inch, bottomMargin=.55*inch)
        w, h = letter
        gap = .22 * inch
        col_w = (w - 1.24*inch - gap) / 2
        first = Frame(.62*inch, .55*inch, w - 1.24*inch, h - 1.13*inch, id="first")
        left = Frame(.62*inch, .55*inch, col_w, h - 1.13*inch, leftPadding=0, rightPadding=0, id="left")
        right = Frame(.62*inch + col_w + gap, .55*inch, col_w, h - 1.13*inch,
                      leftPadding=0, rightPadding=0, id="right")
        self.addPageTemplates([PageTemplate(id="First", frames=[first], onPage=header_footer),
                               PageTemplate(id="TwoCol", frames=[left, right], onPage=header_footer)])


def make_styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="PaperTitle", parent=s["Title"], fontName="Helvetica-Bold", fontSize=17.2,
                         leading=21, textColor=NAVY, alignment=TA_CENTER, spaceAfter=6))
    s.add(ParagraphStyle(name="Author", parent=s["Normal"], fontName="Helvetica", fontSize=9,
                         leading=11, alignment=TA_CENTER, textColor=GRAY, spaceAfter=8))
    s.add(ParagraphStyle(name="AbstractHead", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=9,
                         leading=11, textColor=NAVY, alignment=TA_CENTER, spaceAfter=2))
    s.add(ParagraphStyle(name="Abstract", parent=s["Normal"], fontName="Times-Roman", fontSize=8.55,
                         leading=11, alignment=TA_JUSTIFY, leftIndent=.14*inch, rightIndent=.14*inch, spaceAfter=4))
    s.add(ParagraphStyle(name="Body", parent=s["Normal"], fontName="Times-Roman", fontSize=8.25,
                         leading=10.8, alignment=TA_JUSTIFY, spaceAfter=4.5))
    s.add(ParagraphStyle(name="Section", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=10,
                         leading=11.5, textColor=NAVY, spaceBefore=6, spaceAfter=2.5))
    s.add(ParagraphStyle(name="Caption", parent=s["Normal"], fontName="Times-Roman", fontSize=7.0,
                         leading=8.2, alignment=TA_JUSTIFY, spaceAfter=4))
    s.add(ParagraphStyle(name="TableCaption", parent=s["Normal"], fontName="Times-Roman", fontSize=6.9,
                         leading=8.1, alignment=TA_JUSTIFY, spaceAfter=2))
    s.add(ParagraphStyle(name="Cell", parent=s["Normal"], fontName="Helvetica", fontSize=6.25,
                         leading=7.2, alignment=TA_CENTER))
    s.add(ParagraphStyle(name="Small", parent=s["Normal"], fontName="Times-Roman", fontSize=7.1,
                         leading=8.5, alignment=TA_JUSTIFY, spaceAfter=3))
    return s


def figure(name, width, caption_text, styles):
    path = ASSETS / f"{name}.png"
    with PILImage.open(path) as im:
        aspect = im.height / im.width
    image = Image(str(path), width=width, height=width*aspect)
    image.hAlign = "CENTER"
    return [KeepTogether([image, Spacer(1, 2), para(caption_text, styles["Caption"])])]


def styled_table(data, widths, caption_text, styles):
    rows = [[para(str(x), styles["Cell"]) for x in row] for row in data]
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#C5CED7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return [para(caption_text, styles["TableCaption"]), t, Spacer(1, 5)]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = make_styles()
    doc = PaperDoc(str(OUT))
    col = 3.24 * inch
    story = [
        para("Cross-Session Tool-Trajectory Detection for LLM Agents:<br/>A Reproducible Empirical Evaluation", st["PaperTitle"]),
        para("Xiangyi Li<br/><i>Independent Researcher - replace affiliation and contact before submission</i>", st["Author"]),
        para("ABSTRACT", st["AbstractHead"]),
        para("Indirect prompt injection and multi-round attacks can distribute a harmful objective across otherwise ordinary tool calls. This paper evaluates a reduced trajectory-only candidate derived from a broader BehaviorGraph implementation. The candidate was selected using development evidence (R1-R3), frozen before two independent API confirmation runs (R4-R5), and calibrated without test-set threshold selection. Across 480 attack groups and 400 benign test groups, the candidate detects 139 attack attempts (29.0%), recalls 137 of 320 successful objectives (42.8%), and raises one benign alert (0.25%). Exact total-tool-call-count matching reduces detection to 22.2%-25.7%. Transition/frequency deviations provide the measured signal; removing the cross-session component changes detection by less than one percentage point. Removing cumulative scoring reaches 95%-100% in a post-confirmation exploratory ablation and therefore requires preregistered replication. The results are specific to one model family and a banking-style sandbox and do not support production-readiness or cross-model claims.", st["Abstract"]),
        para("<b>Index Terms</b> - LLM agents, tool trajectories, cross-session detection, prompt injection, reproducible evaluation.", st["Abstract"]),
        para("1. INTRODUCTION", st["Section"]),
        para("Tool-using agents transform model outputs into actions such as reading messages, exporting data, and executing transactions. A harmful goal may be fragmented across sessions, so single-turn content detectors and trajectory-only monitors address different failure modes. This study asks a narrower empirical question: after candidate selection is separated from confirmation, what detection-false-alarm trade-off is obtained from a trajectory score?", st["Body"]),
        para("We distinguish the complete BehaviorGraph framework (Full-BG) from the pre-registered trajectory-only confirmation candidate (TO-CF). Full-BG contains graph-structure, parameter, combination, transition/frequency, and cumulative components. TO-CF disables the first three optional components and is the only configuration independently confirmed in R4/R5. This distinction prevents a graph-novelty claim from being inferred from results that do not measure the full feature set.", st["Body"]),
    ] + figure("figure1_evaluation_protocol", 6.45*inch, "Fig. 1. Leakage-controlled protocol. R1-R3 are development evidence only; TO-CF is frozen before R4-R5, which use new wording, operands, API trajectories, and long normal controls.", st) + [
        para("2. CONFIRMED CANDIDATE AND SCORE PROTOCOL", st["Section"]),
        para("For each ordered group g, the evaluator retains the transition/frequency baseline across sessions and computes a cumulative EWMA score q(c) after every call c. The group statistic is S(g) = max over calls c of q(c). For run r, the threshold is theta_r = nextafter(maximum validation-group S(g), +infinity), where the validation set contains 100 benign groups (50 short and 50 long). A group is alerted iff S(g) >= theta_r. The evaluator applies this external score protocol and does not use the default online get_decision threshold.", st["Body"]),
        para("The canonical thresholds are 5.0040541339 (R4) and 4.9817326042 (R5). P95 is retained only in legacy/development scripts and is not the R4/R5 rule.", st["Body"]),
    ]
    story += [NextPageTemplate("TwoCol"), PageBreak()]
    story += [
        para("3. EXPERIMENTAL DESIGN", st["Section"]),
        para("Each confirmation run contains 240 attack groups, 200 benign test groups, 100 validation groups, and 200 calibration groups. There are 40 attack groups per family across six families: delayed trigger, indirect injection, persistent memory, privilege escalation, tool-chain exfiltration, and transaction cover. The successful-objective denominator is 160 per run because delayed-trigger and persistent-memory attempts did not produce a successful objective under the protocol. Bootstrap intervals resample groups, not individual calls.", st["Body"]),
    ] + styled_table([
        ["Item", "Full-BG framework", "TO-CF confirmation"],
        ["Structure score", "enabled by default", "disabled"],
        ["Parameter rules", "enabled by default", "disabled"],
        ["Tool-combination rules", "enabled by default", "disabled"],
        ["Transition/frequency", "available", "enabled"],
        ["Cumulative score", "available", "enabled"],
        ["Independent confirmation", "not run", "R4 and R5"],
    ], [1.05*inch, 1.06*inch, 1.1*inch], "TABLE I. Configuration boundary. Only TO-CF is an independent confirmation claim.", st) + [
        para("4. INDEPENDENT CONFIRMATION", st["Section"]),
    ] + styled_table([
        ["Metric", "R4", "R5"],
        ["Threshold", "5.004054", "4.981733"],
        ["All-attempt detection", "68/240 = 28.3%", "71/240 = 29.6%"],
        ["Successful recall", "67/160 = 41.9%", "70/160 = 43.8%"],
        ["Benign FPR", "0/200 = 0.0%", "1/200 = 0.5%"],
        ["Exact length matched", "26/117 = 22.2%", "29/113 = 25.7%"],
    ], [1.14*inch, 1.05*inch, 1.05*inch], "TABLE II. TO-CF confirmation results. Intervals and full denominators are in the machine-readable audit.", st) + figure("figure2_score_distributions", col, "Fig. 2. Empirical score distributions for attack and benign groups. Dashed lines are the per-run validation-maximum thresholds; the overlap explains limited recall at low false-positive rates.", st) + [
        para("The two runs are numerically similar, but two repetitions do not establish broad stability. We therefore report them separately and use the aggregate only as a descriptive macro summary: 29.0% all-attempt detection, 42.8% successful-objective recall, and 0.25% FPR.", st["Body"]),
        PageBreak(),
        para("5. ATTACK HETEROGENEITY AND DIAGNOSTICS", st["Section"]),
    ] + figure("figure3_primary_confirmation", col, "Fig. 3. Primary TO-CF metrics across R4/R5. Error bars are bootstrap 95% intervals; exact matching is by total tool-call count.", st) + figure("figure4_exploratory_ablation", col, "Fig. 4. Post-confirmation exploratory ablation. Each variant uses its own validation-maximum calibration; the no-cumulative result is not a confirmed method claim.", st) + styled_table([
        ["Attack family", "R4", "R5"],
        ["Delayed trigger", "0/40 = 0%", "0/40 = 0%"],
        ["Indirect injection", "0/40 = 0%", "0/40 = 0%"],
        ["Persistent memory", "1/40 = 2.5%", "1/40 = 2.5%"],
        ["Privilege escalation", "35/40 = 87.5%", "37/40 = 92.5%"],
        ["Tool-chain exfiltration", "16/40 = 40%", "17/40 = 42.5%"],
        ["Transaction cover", "16/40 = 40%", "16/40 = 40%"],
    ], [1.45*inch, .88*inch, .88*inch], "TABLE III. All-attempt detection by attack family. Successful-objective recall has a separate denominator of 160 per run.", st)
    story += [
        para("The mechanism result is deliberately limited. No cross-session changes detection from 28.3% to 27.9% in R4 and from 29.6% to 29.2% in R5. Removing transition/frequency signals yields 0% in both runs. Removing cumulative scoring yields 95.0% and 100.0%, but this choice was made after observing R4/R5 and must be tested in a new preregistered R6/R7 study.", st["Body"]),
    ]
    story += [
        para("6. FAILURE ANALYSIS AND BASELINE CONTEXT", st["Section"]),
        para("The primary candidate misses 183 successful objectives across R4/R5. The misses are concentrated in indirect injection (80, 44%), tool-chain exfiltration (47, 26%), transaction cover (48, 26%), and privilege escalation (8, 4%). These cases show that a harmful objective can be completed through calls that remain close to normal tool-use patterns; trajectory-only evidence cannot recover information absent from the calls.", st["Body"]),
    ] + figure("figure5_attack_taxonomy", col, "Fig. 5. All-attempt detection heterogeneity across the six attack families (40 groups per family per run). This is an attack-type breakdown, not a missed-case-only chart.", st) + [
        para("AgentShield was evaluated in a separate defended execution because its honeytools and honeytokens change the trajectory. It achieved 67.1%/66.7% all-attempt detection, 100% successful-objective recall, and 0.5%/0.0% FPR in R4/R5. These are descriptive reference measurements, not paired replays or a causal superiority test.", st["Body"]),
    ] + figure("figure6_separate_baseline", col, "Fig. 6. Separate-execution comparison with the official AgentShield adapter. Honeytool/honeytoken intervention changes the environment; no paired significance claim is made.", st) + [
        para("7. LIMITATIONS AND CONCLUSION", st["Section"]),
        para("The study evaluates one API-accessible model family and a banking-style sandbox. It does not establish cross-model transfer, natural-traffic coverage, adaptive-attack robustness, or deployment latency. The FPR estimate is based on 400 independent benign test groups. Exact length matching reduces but does not eliminate confounding. Full-BG was not independently confirmed, and no-cumulative remains post-hoc.", st["Body"]),
        para("The defensible conclusion is narrow: a locked transition/frequency trajectory score detects a measurable but limited fraction of multi-step attacks at a low observed false-positive rate, while the measured incremental value of the cross-session graph component is small in this benchmark. The next method claim requires preregistered R6/R7, a hybrid content-plus-trajectory detector, or additional model/domain confirmation.", st["Body"]),
        para("<b>References</b><br/>[1] Y. H. Rassul and T. A. Rashid, \"AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents,\" arXiv:2605.11026, 2026.<br/>[2] A. Mehta et al., \"FragBench: Cross-Session Attacks Hidden in Benign-Looking Fragments,\" arXiv:2605.11029, 2026.", st["Small"]),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
