"""Generate the six data-backed figures used by the revised manuscript.

The script intentionally reads the frozen R4/R5 JSON artifacts instead of
retyping results.  Every exported PDF therefore has a direct provenance path
to the confirmation evaluations.  Raster PNG companions are produced for
visual QA only; the manuscript should include the vector PDFs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local_results" / "canonical"
OUT = ROOT / "output" / "pdf" / "assets_corrected"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata={"Creator": "generate_corrected_figures.py"})
    fig.savefig(OUT / f"{stem}.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


EVAL = {run: read(DATA / f"confirmation_{run}" / "confirmation_evaluation.json") for run in ("R4", "R5")}
SUMMARY = read(DATA / "confirmation_paper_artifacts" / "confirmation_summary.json")
ABL = {run: read(DATA / f"confirmation_{run}" / "confirmation_ablation_exploratory.json") for run in ("R4", "R5")}


def fig1_protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.18), 10.3, 3.85, boxstyle="round,pad=.08,rounding_size=.16",
                                fc="#f7fafc", ec="#718096", lw=1.3, ls=(0, (5, 3))))
    ax.text(.36, 3.72, "Leakage-controlled evaluation protocol", fontsize=13, weight="bold", color="#172b4d")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.09",
                                    fc=color, ec="#30475e", lw=1.1))
        ax.text(x + w / 2, y + h - .27, title, ha="center", va="center", fontsize=9, weight="bold", color="#17324d")
        ax.text(x + w / 2, y + h / 2 - .10, body, ha="center", va="center", fontsize=7.3, color="#2d435a", linespacing=1.3)

    xs = [0.45, 2.32, 4.19, 6.06, 7.93]
    labels = [
        ("API trajectories", "held-out benign +\n6 attack families", "#e6f0fa"),
        ("Cross-session state", "decayed transition\nand frequency baselines", "#dbeafa"),
        ("Frozen candidate", "trajectory-only: transition\n+ frequency + cumulative", "#e6f0fa"),
        ("Per-run calibration", r"$\theta_r=\mathrm{nextafter}(\max V_r,+\infty)$", "#dbeafa"),
        ("Decision", "alert if\nscore $\geq\theta_r$", "#e6f0fa"),
    ]
    for x, (title, body, color) in zip(xs, labels): box(x, 2.05, 1.48, 1.12, title, body, color)
    for x in [1.93, 3.80, 5.67, 7.54]:
        ax.add_patch(FancyArrowPatch((x, 2.61), (x + .35, 2.61), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#314a62"))
    box(.85, .57, 3.7, .72, "Development (R1–R3)", "candidate diagnosis / selection only; never used as confirmation test", "#fff1d8")
    box(5.25, .57, 4.45, .72, "Independent confirmation (R4–R5)", "new wording, operands, API traces, and long benign controls", "#e6f4ea")
    ax.add_patch(FancyArrowPatch((2.7, 1.30), (2.8, 2.0), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#9a6b2f"))
    ax.add_patch(FancyArrowPatch((7.47, 1.30), (6.8, 2.0), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#3b7955"))
    ax.text(5.30, .27, "Test sets are held out from threshold selection; attack labels are used only for final evaluation.", fontsize=7.2, color="#4b5563")
    export(fig, "figure1_evaluation_protocol")


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def fig2_score_ecdf() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    palette = {"R4 attack": "#b55239", "R5 attack": "#d27e31", "R4 benign": "#31698c", "R5 benign": "#6b9ab6"}
    for run in ("R4", "R5"):
        ev = EVAL[run]
        for kind, label in (("attack", f"{run} attack (n={len(ev['scores']['attack'])})"),
                            ("benign", f"{run} benign (n={len(ev['scores']['benign'])})")):
            x, y = ecdf(ev["scores"][kind].values())
            ax.step(x, y, where="post", lw=1.8, color=palette[f"{run} {kind}"], label=label)
        ax.axvline(ev["threshold"], color=palette[f"{run} attack"], lw=1.0, ls=(0, (4, 2)), alpha=.7)
    ax.set_xlabel("Maximum trajectory score per group")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=.22); ax.legend(frameon=True, loc="lower right", ncol=2)
    ax.text(.99, .03, "dashed lines: run-specific threshold", transform=ax.transAxes, ha="right", fontsize=7, color="#59636e")
    export(fig, "figure2_score_distributions")


def ci_err(rate: float, bounds: list[float]) -> np.ndarray:
    return np.array([[max(0.0, rate - bounds[0])], [max(0.0, bounds[1] - rate)]])


def fig3_primary_results() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.35), constrained_layout=True)
    runs = ["R4", "R5"]
    labels = ["All attempts", "Successful attacks", "Exact count matched"]
    metrics = [("all_attempts", None), ("successful_attacks", None), ("length_matched", "attack_detection")]
    for ax, title, (key, sub) in zip(axes, labels, metrics):
        vals = []; lows = []; highs = []
        for run in runs:
            d = SUMMARY["runs"][runs.index(run)]["primary"]
            item = d[key] if sub is None else d[key][sub]
            vals.append(item["rate"]); bounds = item.get("bootstrap_95_ci", [item["rate"], item["rate"]])
            lows.append(item["rate"] - bounds[0]); highs.append(bounds[1] - item["rate"])
        ax.bar(np.arange(2), vals, yerr=np.array([lows, highs]), capsize=3.2, color=["#4c83a8", "#c9793b"], width=.58)
        ax.set_xticks([0, 1], runs); ax.set_title(title); ax.set_ylim(0, 1.05); ax.set_ylabel("Detection rate")
        ax.grid(axis="y", alpha=.23)
        for i, v in enumerate(vals): ax.text(i, min(1.02, v + .06), f"{v:.1%}", ha="center", fontsize=7.5)
    fig.text(.5, -.03, "Bars show point estimates; whiskers are bootstrap 95% intervals. Exact matching uses total tool-call count.", ha="center", fontsize=7.2, color="#59636e")
    export(fig, "figure3_primary_confirmation")


def fig4_ablation() -> None:
    labels = ["Frozen\ntrajectory-only", "No cross-session", "No transition /\nfrequency", "No cumulative\n(exploratory)"]
    x = np.arange(len(labels)); width = .34
    fig, ax = plt.subplots(figsize=(7.35, 4.0))
    for j, run in enumerate(("R4", "R5")):
        vals = [ABL[run]["variants"][name]["dr"] for name in ("TrajectoryOnly", "NoCrossSession", "NoTransitionFrequency", "NoCumulative")]
        bars = ax.bar(x + (j - .5) * width, vals, width, label=run, color=("#4b83a8", "#c4783a")[j])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(1.02, value + .04), f"{value:.1%}", ha="center", fontsize=7)
    ax.set_xticks(x, labels); ax.set_ylabel("All-attempt detection rate"); ax.set_ylim(0, 1.08)
    ax.set_title("Post-confirmation mechanism ablation (exploratory)")
    ax.grid(axis="y", alpha=.22); ax.legend(frameon=True)
    ax.text(.99, .97, "Own validation-maximum threshold per variant; exploratory, not pre-registered.", transform=ax.transAxes, ha="right", va="top", fontsize=6.7, color="#7a4b20")
    export(fig, "figure4_exploratory_ablation")


def fig5_attack_taxonomy() -> None:
    names = ["Delayed\ntrigger", "Indirect\ninjection", "Persistent\nmemory", "Privilege\nescalation", "Tool-chain\nexfiltration", "Transaction\ncover"]
    keys = ["delayed_trigger", "indirect_prompt_injection", "persistent_memory", "privilege_escalation", "tool_chain_exfiltration", "transaction_cover_tracks"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); x = np.arange(len(keys)); width = .34
    for j, run in enumerate(("R4", "R5")):
        vals = [EVAL[run]["by_attack_type"][key]["rate"] for key in keys]
        ax.bar(x + (j - .5) * width, vals, width, label=run, color=("#4b83a8", "#c4783a")[j])
        for i, v in enumerate(vals): ax.text(x[i] + (j - .5) * width, v + .025, f"{v:.0%}", ha="center", fontsize=6.8)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.06); ax.set_ylabel("All-attempt detection rate")
    ax.set_title("Detection heterogeneity across attack families (40 attempts/type/run)")
    ax.grid(axis="y", alpha=.22); ax.legend(frameon=True)
    export(fig, "figure5_attack_taxonomy")


def fig6_baseline() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), constrained_layout=True)
    runs = ["R4", "R5"]
    primary = [SUMMARY["runs"][i]["primary"]["successful_attacks"]["rate"] for i in range(2)]
    official = [SUMMARY["runs"][i]["official_agentshield_separate_execution"]["successful_attacks"]["rate"] for i in range(2)]
    p_fpr = [SUMMARY["runs"][i]["primary"]["benign"]["rate"] for i in range(2)]
    o_fpr = [SUMMARY["runs"][i]["official_agentshield_separate_execution"]["benign"]["rate"] for i in range(2)]
    for ax, a, b, title, ylim in ((axes[0], primary, official, "Successful-attack recall", (0, 1.08)),
                                  (axes[1], p_fpr, o_fpr, "Benign false-positive rate", (0, .035))):
        ax.bar(np.arange(2) - .18, a, .36, color="#4b83a8", label="Trajectory-only candidate")
        ax.bar(np.arange(2) + .18, b, .36, color="#d08a35", label="Official AgentShield")
        ax.set_xticks([0, 1], runs); ax.set_ylim(*ylim); ax.set_title(title); ax.set_ylabel("Rate"); ax.grid(axis="y", alpha=.22)
    axes[0].legend(frameon=True, fontsize=7)
    fig.text(.5, -.04, "Separate defended executions (honeytools/honeytokens); descriptive comparison only, not paired significance testing.", ha="center", fontsize=7.0, color="#7a4b20")
    export(fig, "figure6_separate_baseline")


def main() -> None:
    fig1_protocol(); fig2_score_ecdf(); fig3_primary_results(); fig4_ablation(); fig5_attack_taxonomy(); fig6_baseline()
    manifest = {
        "source": "local_results/canonical/confirmation_R4, confirmation_R5 and confirmation_paper_artifacts/confirmation_summary.json",
        "figures": [
            "figure1_evaluation_protocol", "figure2_score_distributions", "figure3_primary_confirmation",
            "figure4_exploratory_ablation", "figure5_attack_taxonomy", "figure6_separate_baseline",
        ],
        "notes": [
            "R4/R5 are independent confirmation runs; development R1-R3 are not plotted as test results.",
            "AgentShield is a separate execution because its honeytools/honeytokens alter the environment; no paired significance claim is made.",
            "Ablation is exploratory and post-confirmation; it is not presented as a preregistered primary result.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
