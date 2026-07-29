"""Compose the revised, submission-style manuscript without relying on MiKTeX packages.

The content is deliberately limited to the frozen/independent experimental evidence.
"""
from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, Table,
    TableStyle, PageBreak, NextPageTemplate, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "cross_session_trajectory_detection_revised.pdf"
ASSETS = ROOT / "output" / "pdf" / "assets"
NAVY = colors.HexColor("#14213D")
BLUE = colors.HexColor("#3B78A6")
LIGHT_BLUE = colors.HexColor("#E8F0FA")
GRAY = colors.HexColor("#56616F")


def p(text: str, style):
    return Paragraph(text, style)


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#AAB4BF"))
        canvas.setLineWidth(.35)
        canvas.line(.62 * inch, h - .45 * inch, w - .62 * inch, h - .45 * inch)
        canvas.setFont("Helvetica", 7.3)
        canvas.setFillColor(GRAY)
        canvas.drawString(.64 * inch, h - .37 * inch, "Cross-Session Tool-Trajectory Detection in LLM Agents")
        canvas.drawRightString(w - .64 * inch, .36 * inch, str(doc.page))
    canvas.restoreState()


class SubmissionDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=letter, leftMargin=.62*inch, rightMargin=.62*inch,
                         topMargin=.58*inch, bottomMargin=.55*inch)
        w, h = letter
        gap = .22 * inch
        col_w = (w - 2*.62*inch - gap) / 2
        first = Frame(.62*inch, .55*inch, w - 1.24*inch, h - 1.13*inch, id="first")
        left = Frame(.62*inch, .55*inch, col_w, h - 1.13*inch, leftPadding=0, rightPadding=0, id="left")
        right = Frame(.62*inch + col_w + gap, .55*inch, col_w, h - 1.13*inch,
                      leftPadding=0, rightPadding=0, id="right")
        self.addPageTemplates([
            PageTemplate(id="First", frames=[first], onPage=header_footer),
            PageTemplate(id="TwoCol", frames=[left, right], onPage=header_footer),
        ])


def caption(text, styles):
    return p(f"<b>Fig.</b> {text}", styles["Caption"])


def figure(name: str, width: float, text: str, styles):
    path = ASSETS / f"{name}.png"
    img = Image(str(path), width=width, height=width * (0.40 if name == "figure1_protocol" else 0.58))
    img.hAlign = "CENTER"
    return KeepTogether([img, Spacer(1, 3), caption(text, styles), Spacer(1, 7)])


def table(data, widths, styles, caption_text):
    body = [[p(str(x), styles["Cell"]) for x in row] for row in data]
    t = Table(body, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .28, colors.HexColor("#C5CED7")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([p(caption_text, styles["TableCaption"]), Spacer(1, 3), t, Spacer(1, 7)])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitlePaper", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18,
                              leading=22, textColor=NAVY, alignment=TA_CENTER, spaceAfter=7))
    styles.add(ParagraphStyle(name="Author", parent=styles["Normal"], fontName="Helvetica", fontSize=9.2,
                              leading=12, alignment=TA_CENTER, textColor=GRAY, spaceAfter=10))
    styles.add(ParagraphStyle(name="AbstractHead", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.1,
                              leading=12, textColor=NAVY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2))
    styles.add(ParagraphStyle(name="Abstract", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.8,
                              leading=11.5, alignment=TA_JUSTIFY, leftIndent=.16*inch, rightIndent=.16*inch, spaceAfter=5))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.55,
                              leading=11.4, alignment=TA_JUSTIFY, spaceAfter=5.5))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.2,
                              leading=12, textColor=NAVY, spaceBefore=7, spaceAfter=3))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontName="Times-Roman", fontSize=7.2,
                              leading=8.7, alignment=TA_JUSTIFY, spaceAfter=0))
    styles.add(ParagraphStyle(name="TableCaption", parent=styles["Normal"], fontName="Times-Roman", fontSize=7.1,
                              leading=8.4, alignment=TA_JUSTIFY, spaceAfter=0))
    styles.add(ParagraphStyle(name="Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=6.65,
                              leading=7.7, alignment=TA_CENTER))
    doc = SubmissionDoc(str(OUT))
    story = []
    story += [p("Cross-Session Tool-Trajectory Detection in LLM Agents:<br/>A Replicated Study of Detection-False-Alarm Trade-offs", styles["TitlePaper"]),
              p("Xiangyi Li<br/><i>Independent Researcher - contact information to be supplied for submission</i>", styles["Author"]),
              p("ABSTRACT", styles["AbstractHead"]),
              p("LLM agents turn language-model outputs into sequences of tool calls, so a harmful objective may be expressed across several sessions rather than in one isolated prompt. This paper re-evaluates a cross-session trajectory detector under a frozen, API-executed protocol. A trajectory-only candidate was selected during development (R1-R3), then locked before two independent confirmation runs (R4-R5) with new prompt wording, operands, and long normal controls. On successful attack objectives, the candidate recalled 42.8% +/- 1.3% while producing 0.25% +/- 0.35% false positives across runs; under exact output-length matching, recall fell to 22.2%-25.7%. A diagnostic ablation shows that transition/frequency signals are indispensable, whereas the proposed cross-session graph supplies little incremental benefit in this setting. A post-hoc removal of cumulative scoring performs much better, but is explicitly exploratory. An official AgentShield implementation achieves perfect recall in separate honeytool-augmented executions, so no paired superiority claim is made. The contribution is a reproducible measurement protocol and a calibrated account of what this detector does and does not establish, rather than a production-readiness claim.", styles["Abstract"]),
              p("<b>Index Terms</b> - LLM agents, agent security, tool trajectories, cross-session detection, reproducible evaluation.", styles["Abstract"]),
              Spacer(1, 3),
              p("1. INTRODUCTION", styles["Section"]),
              p("Tool-using LLM agents can browse, retrieve, transact, and communicate. Their security-relevant behavior is consequently a trajectory: an apparently benign call may become harmful only in the context of earlier calls, repeated transitions, or accumulated intent. This motivates cross-session behavioral monitoring, but it also creates a demanding evaluation problem. A detector can appear effective when attack traces are longer, when a threshold is selected on the test set, or when a protected baseline changes the executed environment.", styles["Body"]),
              p("This manuscript replaces a performance-oriented interpretation of an earlier prototype with a frozen, independent evaluation. Its research question is narrow: <i>what detection-false-alarm trade-off does a trajectory score provide after candidate selection is separated from confirmation?</i> The answer is mixed. The locked candidate detects a material fraction of attacks at a low false-positive rate, but it misses most successful objectives and does not justify a deployment or superiority claim.", styles["Body"]),
              figure("figure1_protocol", 6.65*inch, "1. Evaluation architecture. R1-R3 are development-only evidence. The candidate and calibration rule are frozen before R4-R5, which use new wording, operands, API-executed trajectories, and long normal controls.", styles),
              p("2. DETECTOR AND FROZEN PROTOCOL", styles["Section"]),
              p("The candidate maintains decayed transition counts and benign frequency estimates for tool-call trajectories. It scores a session using transition and frequency deviations, then accumulates evidence across the trajectory. Parameter-level rules, tool-combination rules, and a claimed graph-structure component are excluded from the locked candidate because they did not survive development diagnostics. The alarm threshold is fixed before test evaluation as the smallest value above the maximum score among 100 benign validation trajectories.", styles["Body"])]
    story += [NextPageTemplate("TwoCol"), PageBreak()]
    col = 3.27 * inch
    story += [p("3. EXPERIMENTAL DESIGN", styles["Section"]),
              p("Each independent run includes 240 attack groups, 200 normal test groups, 100 validation-normal groups, and 200 calibration-normal groups. Attack groups span indirect injection, tool exfiltration, transaction cover, privilege escalation, and related multi-step patterns. The reported successful-attack recall conditions on objectives that actually succeeded (160 in each run); all-attempt detection includes every attack attempt. We use nonparametric bootstrap intervals over groups for within-run rates. R4/R5 are independent confirmation repetitions, not hyperparameter searches.", styles["Body"]),
              table([["Item", "Development R1-R3", "Confirmation R4-R5"], ["Role", "Candidate selection", "Frozen evaluation"], ["Normal calibration", "Development only", "200 (100 short / 100 long)"], ["Validation normal", "-", "100"], ["Test normal", "-", "200"], ["Attack groups", "Development templates", "240; six held-out families"], ["Execution", "API traces", "New operands and wording"]], [0.68*inch, 1.17*inch, 1.42*inch], styles, "TABLE I. Separation of development and independent confirmation. Counts are per R4/R5 run."),
              p("The main comparison is intentionally limited. The official AgentShield implementation is run unchanged behind a local adapter, but its honeytools and honeytokens alter agent behavior. It therefore receives its own defended executions, rather than paired replays of identical traces. Its values are descriptive reference measurements, not inputs to paired tests or claims of statistical superiority.", styles["Body"]),
              p("4. INDEPENDENT CONFIRMATION RESULTS", styles["Section"]),
              p("The prespecified candidate has a low aggregate false-positive rate but limited successful-attack recall: 41.9% in R4 and 43.8% in R5. Exact length matching further reduces recall, ruling out the interpretation that the main result fully controls for attack-trace length. The high validation-maximum threshold protects normal traffic but leaves many attack groups below the alarm boundary.", styles["Body"]),
              table([["Metric", "R4", "R5"], ["All-attempt detection", "28.3% [22.5, 34.2]", "29.6% [23.8, 35.4]"], ["Successful recall", "41.9% [34.4, 49.4]", "43.8% [36.3, 51.9]"], ["False-positive rate", "0.0% [0.0, 0.0]", "0.5% [0.0, 1.5]"], ["Length-matched detection", "22.2% [14.5, 29.9]", "25.7% [17.7, 33.6]"]], [0.8*inch, 1.24*inch, 1.24*inch], styles, "TABLE II. Frozen candidate on independent confirmation. 95% bootstrap CIs."),
              figure("figure2_score_ecdf", col, "2. Empirical CDFs of maximum trajectory score. Dashed lines are frozen validation-maximum thresholds; attack and benign distributions overlap substantially.", styles),
              p("5. MECHANISM DIAGNOSTICS", styles["Section"]),
              p("The most important mechanistic finding is negative. Removing the cross-session feature changes all-attempt detection by less than half a percentage point. In contrast, removing transition/frequency signals eliminates detection under the same calibration rule. Removing cumulative scoring produces 95% and 100% all-attempt detection in R4/R5. Since that latter ablation was performed after examining confirmation data, it is a hypothesis for R6/R7 replication, not a replacement for the frozen primary result.", styles["Body"]),
              figure("figure4_ablation", col, "3. Exploratory mechanism analysis at the validation-maximum calibration. The no-cumulative result requires fresh preregistered replication before supporting a method claim.", styles),
              table([["Variant", "R4", "R5"], ["Frozen trajectory-only", "28.3%", "29.6%"], ["No cross-session", "27.9%", "29.2%"], ["No transition/frequency", "0.0%", "0.0%"], ["No cumulative (exploratory)", "95.0%", "100.0%"]], [1.35*inch, .88*inch, .88*inch], styles, "TABLE III. Post-confirmation exploratory ablation; values are all-attempt detection."),
              p("6. BASELINE AND FAILURE ANALYSIS", styles["Section"]),
              p("Across R4/R5, 183 successful objectives were missed: 80 indirect-injection cases, 47 tool-exfiltration cases, 48 transaction-cover cases, and 8 privilege-escalation cases. The current score detects unusual transitions but can miss a harmful objective whose individual calls resemble normal trajectories.", styles["Body"]),
              figure("figure3_confirmation", col, "4. Independent confirmation versus official AgentShield. AgentShield uses separate honeytool/honeytoken-protected executions and the comparison is descriptive only.", styles),
              p("7. THREATS TO VALIDITY", styles["Section"]),
              p("This study evaluates one API-accessible model and a banking-style tool sandbox; it does not establish cross-model generalization, real-world coverage, or robustness to adaptive attackers. The aggregate false-positive estimate remains uncertain because only 400 independent test-normal groups are observed. Long-control matching reduces, but does not eliminate, trace-length confounding. The environment-altering AgentShield comparison cannot identify a paired causal advantage.", styles["Body"]),
              p("8. CONCLUSION", styles["Section"]),
              p("Cross-session trajectory monitoring is a worthwhile security direction, but the present candidate is not production-ready. Independent confirmation shows low false positives alongside limited attack recall, negligible measured value from the cross-session component, and substantial failure modes. The revised claim is deliberately narrower: frozen evaluation and transparent diagnostics are necessary to distinguish promising trajectory signals from an overstated safety result.", styles["Body"]),
              p("<b>References</b><br/>[1] Y. H. Rassul and T. A. Rashid, \"AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents,\" arXiv:2605.11026, 2026.<br/>[2] A. Mehta et al., \"FragBench: Cross-Session Attacks Hidden in Benign-Looking Fragments,\" arXiv:2605.11029, 2026.", styles["Caption"])]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
