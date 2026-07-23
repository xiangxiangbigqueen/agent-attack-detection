"""
Statistical Significance Tests
================================
Computes CIs, McNemar's test, chi-squared, and effect sizes
on experiment results.

Usage: python -m agentshield.compute_statistics
"""

import sys
import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def proportion_ci(successes, total, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return 0, (0, 0)
    p = successes / total
    z = 1.96 if confidence == 0.95 else 2.576
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, (max(0, center - margin), min(1, center + margin))


def mcnemar_test(n01, n10):
    """McNemar's test for paired binary data. Returns chi2 and p-value."""
    if n01 + n10 == 0:
        return 0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)  # with continuity correction
    # Approximate p-value from chi2 with 1 df
    # Using simple approximation
    from math import exp, sqrt
    p = exp(-chi2 / 2) if chi2 < 20 else 0.0
    return chi2, p


def chi_squared_test(observed):
    """Chi-squared test for independence on a 2D contingency table."""
    rows = len(observed)
    cols = len(observed[0])
    total = sum(sum(row) for row in observed)
    if total == 0:
        return 0, 1.0

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(rows)) for c in range(cols)]

    chi2 = 0
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / total
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    # Degrees of freedom
    df = (rows - 1) * (cols - 1)
    # Approximate p-value (chi2 with df degrees of freedom)
    from math import exp
    p = exp(-chi2 / 2) if chi2 < 30 else 0.0  # rough approximation
    return chi2, df, p


def cohens_h(p1, p2):
    """Cohen's h effect size for difference between two proportions."""
    import math
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main():
    results_dir = Path("agentshield/results")

    # Load latest all-suites results for both models
    gpt4o_files = sorted(results_dir.glob("all_suites_80attacks_2026-03-21*.json"))
    gpt5_files = sorted(results_dir.glob("all_suites_80attacks_gpt-5-mini*.json"))

    if not gpt4o_files or not gpt5_files:
        print("Missing result files!")
        return

    with open(gpt4o_files[-1], "r", encoding="utf-8") as f:
        gpt4o = json.load(f)
    with open(gpt5_files[-1], "r", encoding="utf-8") as f:
        gpt5 = json.load(f)

    # Also try to load repeated trial data if available
    trial_files_4o = sorted(results_dir.glob("repeated_trials_gpt-4o-mini*.json"))
    trial_files_5 = sorted(results_dir.glob("repeated_trials_gpt-5-mini*.json"))

    print("=" * 60)
    print("  STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    shared = {"goal_hijack", "data_exfil", "tool_misuse", "adaptive"}

    for model_name, data in [("GPT-4o-mini", gpt4o), ("GPT-5-mini", gpt5)]:
        runs = data["runs"]
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # 1. Detection rate with Wilson CIs per language
        print(f"\n  Detection rates with 95% Wilson CIs (shared categories):")
        print(f"  {'Lang':4s} {'Det':>5s} {'Total':>5s} {'Rate':>8s} {'95% CI':>18s}")
        lang_data = {}
        for lang in ["EN", "KU", "AR", "CS"]:
            l_runs = [r for r in runs if r["language"] == lang and r["category"] in shared]
            l_det = sum(1 for r in l_runs if r["attack_succeeded"])
            p, ci = proportion_ci(l_det, len(l_runs))
            lang_data[lang] = (l_det, len(l_runs), p)
            print(f"  {lang:4s} {l_det:5d} {len(l_runs):5d} {100*p:7.1f}% [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]")

        # 2. McNemar's test: EN vs KU (paired by attack ID)
        print(f"\n  McNemar's test: EN vs KU detection difference:")
        en_results = {}
        ku_results = {}
        for r in runs:
            if r["category"] not in shared:
                continue
            # Match by attack category + number (e.g., HIJACK_01)
            attack_num = r["attack_id"].split("_", 1)[1]  # e.g., "HIJACK_01"
            if r["language"] == "EN":
                en_results[attack_num] = r["attack_succeeded"]
            elif r["language"] == "KU":
                ku_results[attack_num] = r["attack_succeeded"]

        # Count discordant pairs
        n01 = 0  # EN detected, KU not
        n10 = 0  # KU detected, EN not
        for key in en_results:
            if key in ku_results:
                if en_results[key] and not ku_results[key]:
                    n01 += 1
                elif ku_results[key] and not en_results[key]:
                    n10 += 1

        chi2, p = mcnemar_test(n01, n10)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"    EN detected but KU not: {n01}")
        print(f"    KU detected but EN not: {n10}")
        print(f"    McNemar chi2 = {chi2:.3f}, p = {p:.4f} ({sig} at alpha=0.05)")

        # 3. Chi-squared: language × detection
        print(f"\n  Chi-squared test: language x detection independence:")
        observed = []
        for lang in ["EN", "KU", "AR", "CS"]:
            det, total, _ = lang_data[lang]
            observed.append([det, total - det])

        chi2, df, p = chi_squared_test(observed)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"    Chi2 = {chi2:.3f}, df = {df}, p = {p:.4f} ({sig})")

        # 4. Effect sizes (Cohen's h)
        print(f"\n  Cohen's h effect sizes (vs English):")
        en_p = lang_data["EN"][2]
        for lang in ["KU", "AR", "CS"]:
            lang_p = lang_data[lang][2]
            h = cohens_h(en_p, lang_p)
            magnitude = "negligible" if abs(h) < 0.2 else "small" if abs(h) < 0.5 else "medium" if abs(h) < 0.8 else "large"
            print(f"    EN vs {lang}: h = {h:.3f} ({magnitude})")

    # 5. Cross-model comparison
    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL COMPARISON (EN vs KU gap)")
    print(f"{'='*60}")

    for model_name, data in [("GPT-4o-mini", gpt4o), ("GPT-5-mini", gpt5)]:
        runs = data["runs"]
        en_runs = [r for r in runs if r["language"] == "EN" and r["category"] in shared]
        ku_runs = [r for r in runs if r["language"] == "KU" and r["category"] in shared]
        en_rate = sum(1 for r in en_runs if r["attack_succeeded"]) / max(len(en_runs), 1)
        ku_rate = sum(1 for r in ku_runs if r["attack_succeeded"]) / max(len(ku_runs), 1)
        gap = en_rate - ku_rate
        h = cohens_h(en_rate, ku_rate)
        print(f"  {model_name}: EN={100*en_rate:.1f}%, KU={100*ku_rate:.1f}%, gap={100*gap:.1f}%, Cohen's h={h:.3f}")

    # 6. Repeated trials CIs (if available)
    for trial_files, model_name in [(trial_files_4o, "GPT-4o-mini"), (trial_files_5, "GPT-5-mini")]:
        if trial_files:
            print(f"\n{'='*60}")
            print(f"  REPEATED TRIAL CIs — {model_name}")
            print(f"{'='*60}")
            with open(trial_files[-1], "r", encoding="utf-8") as f:
                trial_data = json.load(f)

            for trial_info in trial_data["trials"]:
                runs = trial_info["runs"]
                detected = sum(1 for r in runs if r["attack_succeeded"])
                rate = 100 * detected / len(runs)
                print(f"  Trial {trial_info['trial_num']}: {detected}/{len(runs)} ({rate:.1f}%)")

            # Compute per-trial rates and CI
            trial_rates = []
            for trial_info in trial_data["trials"]:
                runs = trial_info["runs"]
                rate = sum(1 for r in runs if r["attack_succeeded"]) / len(runs)
                trial_rates.append(rate)

            mean = sum(trial_rates) / len(trial_rates)
            if len(trial_rates) > 1:
                std = math.sqrt(sum((r - mean)**2 for r in trial_rates) / (len(trial_rates) - 1))
                t = {2: 4.303, 3: 3.182, 4: 2.776}.get(len(trial_rates) - 1, 1.96)
                margin = t * std / math.sqrt(len(trial_rates))
                print(f"\n  Mean: {100*mean:.1f}% +/- {100*std:.1f}%")
                print(f"  95% CI: [{100*(mean-margin):.1f}%, {100*(mean+margin):.1f}%]")

    print()


if __name__ == "__main__":
    main()
