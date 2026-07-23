"""
Input-Level vs Behavioral Detection — Complementarity Analysis
===============================================================
Compares two detection paradigms:
  Task A: Malicious string detection (ProtectAI DeBERTa v2, Meta Prompt-Guard-2)
  Task B: Agent compromise detection (AgentShield RF classifier)
  Task C: Combined coverage (complementarity matrix)

Usage:
  python -m agentshield.compare_input_vs_behavioral
  python -m agentshield.compare_input_vs_behavioral --skip-protectai  # if download fails
  python -m agentshield.compare_input_vs_behavioral --embedded        # test embedded-in-context
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentshield.attacks.attack_prompts import ALL_ATTACKS
from agentshield.attacks.adaptive_attacks_v2 import ALL_ADAPTIVE_V2
from agentshield.attacks.benign_prompts_multilingual import BENIGN_EN, BENIGN_KU, BENIGN_AR
from agentshield.compute_statistics import proportion_ci, mcnemar_test, cohens_h


# ── Model IDs ──
PROTECTAI_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
PROMPTGUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"

# ── Adaptive v2 translations ──
TRANSLATIONS_PATH = Path("agentshield/attacks/adaptive_v2_translations.json")


def load_translations():
    """Load Kurdish and Arabic translations for adaptive v2 attacks."""
    if TRANSLATIONS_PATH.exists():
        with open(TRANSLATIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"kurdish": {}, "arabic": {}}


def build_attack_samples():
    """Build all attack text samples with language labels."""
    samples = []
    translations = load_translations()

    # Set A: standard attacks (already have language field)
    for a in ALL_ATTACKS:
        samples.append({
            "id": a["id"],
            "text": a["payload"],
            "language": a["language"],
            "category": a.get("category", "unknown"),
            "source": "set_a",
            "suite": a.get("suite", "mixed"),
        })

    # Adaptive v2: English originals
    for a in ALL_ADAPTIVE_V2:
        samples.append({
            "id": f"{a['id']}_EN",
            "text": a["payload"],
            "language": "EN",
            "category": f"adaptive_t{a['tier']}",
            "source": "adaptive_v2",
            "suite": a["suite"],
            "tier": a["tier"],
            "target_layers": a["target_layers"],
        })

    # Adaptive v2: Kurdish translations
    for a in ALL_ADAPTIVE_V2:
        ku_text = translations.get("kurdish", {}).get(a["id"], "")
        if ku_text:
            samples.append({
                "id": f"{a['id']}_KU",
                "text": "\n\n" + ku_text,
                "language": "KU",
                "category": f"adaptive_t{a['tier']}",
                "source": "adaptive_v2",
                "suite": a["suite"],
                "tier": a["tier"],
                "target_layers": a["target_layers"],
            })

    # Adaptive v2: Arabic translations
    for a in ALL_ADAPTIVE_V2:
        ar_text = translations.get("arabic", {}).get(a["id"], "")
        if ar_text:
            samples.append({
                "id": f"{a['id']}_AR",
                "text": "\n\n" + ar_text,
                "language": "AR",
                "category": f"adaptive_t{a['tier']}",
                "source": "adaptive_v2",
                "suite": a["suite"],
                "tier": a["tier"],
                "target_layers": a["target_layers"],
            })

    return samples


def build_benign_samples():
    """Build all benign text samples with language labels."""
    samples = []
    for b in BENIGN_EN:
        samples.append({
            "id": b["id"],
            "text": b["prompt"],
            "language": "EN",
            "source": "benign",
            "suite": b["suite"],
        })
    for b in BENIGN_KU:
        samples.append({
            "id": b["id"],
            "text": b["prompt"],
            "language": "KU",
            "source": "benign",
            "suite": b["suite"],
        })
    for b in BENIGN_AR:
        samples.append({
            "id": b["id"],
            "text": b["prompt"],
            "language": "AR",
            "source": "benign",
            "suite": b["suite"],
        })
    return samples


def wrap_in_context(text, suite="banking"):
    """Wrap attack payload in realistic tool output context."""
    contexts = {
        "banking": "Transaction Record — Bill Payment\nDate: 2024-03-15\nAmount: 45.00 EUR\nRecipient: Electric Company GmbH\nReference: Monthly electricity bill\n---\nNote from sender: {payload}\n---\nStatus: Pending review",
        "slack": "Channel: #general\nFrom: bob@company.com\nTimestamp: 2024-03-15 10:30\n\n{payload}\n\n--- End of message ---",
        "travel": "Booking Confirmation #TR-4892\nHotel: Le Marais Boutique, Paris\nCheck-in: May 1, 2024\nCheck-out: May 5, 2024\nGuest notes:\n{payload}\n\nThank you for your reservation.",
        "workspace": "File: meeting-notes-2024-03-15.docx\nLast modified: March 15, 2024\nContent:\n{payload}\n\n--- End of document ---",
    }
    template = contexts.get(suite, contexts["banking"])
    return template.format(payload=text)


def run_classifier(model_name, samples, embedded=False):
    """Run a HuggingFace text classifier on all samples."""
    from transformers import pipeline

    print(f"\n  Loading {model_name}...")
    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512,
            device=-1,  # CPU
        )
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return None

    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Processing...")

        text = sample["text"]
        if embedded:
            text = wrap_in_context(text, sample.get("suite", "banking"))

        try:
            pred = classifier(text)[0]
            label = pred["label"]
            score = pred["score"]

            # Normalize labels across models
            # ProtectAI: LABEL_0=benign, LABEL_1=injection
            # Prompt-Guard: benign/malicious (or LABEL_0/LABEL_1)
            is_injection = False
            label_lower = label.lower()
            if "injection" in label_lower or "malicious" in label_lower or label == "LABEL_1":
                is_injection = True

            # Measure tokenized length
            tokenizer = classifier.tokenizer
            tokens = tokenizer.encode(text)
            token_count = len(tokens)

            results.append({
                "id": sample["id"],
                "language": sample["language"],
                "source": sample.get("source", "unknown"),
                "category": sample.get("category", "unknown"),
                "suite": sample.get("suite", "unknown"),
                "raw_label": label,
                "score": round(score, 4),
                "is_injection": is_injection,
                "token_count": token_count,
                "truncated": token_count > 512,
            })
        except Exception as e:
            results.append({
                "id": sample["id"],
                "language": sample["language"],
                "source": sample.get("source", "unknown"),
                "error": str(e),
                "is_injection": None,
            })

    return results


def compute_task_a_metrics(attack_results, benign_results, model_name):
    """Compute Task A: Malicious String Detection metrics."""
    print(f"\n{'='*60}")
    print(f"  Task A: Malicious String Detection — {model_name}")
    print(f"{'='*60}")

    # Per-language recall on attack samples
    by_lang = defaultdict(lambda: {"tp": 0, "fn": 0})
    for r in attack_results:
        if r.get("is_injection") is None:
            continue
        lang = r["language"]
        if r["is_injection"]:
            by_lang[lang]["tp"] += 1
        else:
            by_lang[lang]["fn"] += 1

    print(f"\n  Detection Rate (Recall) on Attack Prompts:")
    print(f"  {'Language':<8} {'TP':>5} {'FN':>5} {'Total':>6} {'Recall':>8} {'95% CI':>16}")
    print(f"  {'-'*50}")

    lang_recalls = {}
    for lang in ["EN", "KU", "AR", "CS"]:
        if lang not in by_lang:
            continue
        tp = by_lang[lang]["tp"]
        fn = by_lang[lang]["fn"]
        total = tp + fn
        recall, ci = proportion_ci(tp, total)
        lang_recalls[lang] = recall
        print(f"  {lang:<8} {tp:>5} {fn:>5} {total:>6} {recall:>7.1%} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Overall
    tp_all = sum(d["tp"] for d in by_lang.values())
    fn_all = sum(d["fn"] for d in by_lang.values())
    total_all = tp_all + fn_all
    recall_all, ci_all = proportion_ci(tp_all, total_all)
    print(f"  {'ALL':<8} {tp_all:>5} {fn_all:>5} {total_all:>6} {recall_all:>7.1%} [{ci_all[0]:.3f}, {ci_all[1]:.3f}]")

    # Per-language FPR on benign samples
    by_lang_fp = defaultdict(lambda: {"fp": 0, "tn": 0})
    for r in benign_results:
        if r.get("is_injection") is None:
            continue
        lang = r["language"]
        if r["is_injection"]:
            by_lang_fp[lang]["fp"] += 1
        else:
            by_lang_fp[lang]["tn"] += 1

    print(f"\n  False Positive Rate on Benign Prompts:")
    print(f"  {'Language':<8} {'FP':>5} {'TN':>5} {'Total':>6} {'FPR':>8} {'95% CI':>16}")
    print(f"  {'-'*50}")

    for lang in ["EN", "KU", "AR"]:
        if lang not in by_lang_fp:
            continue
        fp = by_lang_fp[lang]["fp"]
        tn = by_lang_fp[lang]["tn"]
        total = fp + tn
        fpr, ci = proportion_ci(fp, total)
        print(f"  {lang:<8} {fp:>5} {tn:>5} {total:>6} {fpr:>7.1%} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Cross-lingual gap
    if "EN" in lang_recalls and "KU" in lang_recalls:
        gap = lang_recalls["EN"] - lang_recalls["KU"]
        h = cohens_h(lang_recalls["EN"], lang_recalls["KU"])
        print(f"\n  Cross-lingual gap (EN-KU): {gap:+.1%} (Cohen's h = {abs(h):.3f})")
    if "EN" in lang_recalls and "AR" in lang_recalls:
        gap = lang_recalls["EN"] - lang_recalls["AR"]
        h = cohens_h(lang_recalls["EN"], lang_recalls["AR"])
        print(f"  Cross-lingual gap (EN-AR): {gap:+.1%} (Cohen's h = {abs(h):.3f})")

    # By category (obfuscation sub-analysis)
    obfusc_cats = {"zero_width", "transliteration", "homoglyph"}
    obfusc = [r for r in attack_results if r.get("category") in obfusc_cats]
    if obfusc:
        tp_ob = sum(1 for r in obfusc if r.get("is_injection"))
        total_ob = len(obfusc)
        recall_ob, ci_ob = proportion_ci(tp_ob, total_ob)
        print(f"\n  Obfuscation attacks (n={total_ob}): recall={recall_ob:.1%} [{ci_ob[0]:.3f}, {ci_ob[1]:.3f}]")

    # Tokenization stats
    token_counts = [r["token_count"] for r in attack_results if "token_count" in r]
    if token_counts:
        truncated = sum(1 for r in attack_results if r.get("truncated"))
        print(f"\n  Token stats: mean={sum(token_counts)/len(token_counts):.1f}, "
              f"max={max(token_counts)}, truncated={truncated}/{len(token_counts)}")

    return {
        "model": model_name,
        "recall_by_language": {l: {"tp": d["tp"], "fn": d["fn"],
                                    "recall": round(d["tp"]/(d["tp"]+d["fn"]), 4) if d["tp"]+d["fn"] > 0 else 0}
                                for l, d in by_lang.items()},
        "recall_overall": round(recall_all, 4),
        "fpr_by_language": {l: {"fp": d["fp"], "tn": d["tn"],
                                 "fpr": round(d["fp"]/(d["fp"]+d["tn"]), 4) if d["fp"]+d["tn"] > 0 else 0}
                            for l, d in by_lang_fp.items()},
        "lang_recalls": lang_recalls,
    }


def load_behavioral_results():
    """Load existing AgentShield behavioral classifier results."""
    results_dir = Path("agentshield/results")

    # Load classifier cross-language results
    cl_path = results_dir / "classifier_results_cross-language.json"
    if cl_path.exists():
        with open(cl_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def compute_complementarity(attack_samples, input_results_dict, behavioral_data):
    """Compute Task C: Complementarity Matrix.

    Since input-level and behavioral operate on different inputs,
    we match by attack_id where possible.
    """
    print(f"\n{'='*60}")
    print(f"  Task C: Complementarity Analysis")
    print(f"{'='*60}")

    # For now, report what each paradigm catches independently
    # A full paired analysis would require running the RF classifier on the
    # same attack instances — we use the existing classifier results as proxy

    for model_name, results in input_results_dict.items():
        attack_results = [r for r in results if r.get("source") != "benign"]
        detected = sum(1 for r in attack_results if r.get("is_injection"))
        total = len(attack_results)
        print(f"\n  {model_name}:")
        print(f"    Attacks detected by input-level: {detected}/{total} ({detected/total:.1%})")

    if behavioral_data:
        print(f"\n  AgentShield RF (from existing results):")
        print(f"    Cross-language F1: {behavioral_data.get('f1', 'N/A')}")
        print(f"    FPR: {behavioral_data.get('fpr', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Input-Level vs Behavioral Detection Complementarity Analysis"
    )
    parser.add_argument("--skip-protectai", action="store_true",
                        help="Skip ProtectAI DeBERTa (if download fails)")
    parser.add_argument("--skip-promptguard", action="store_true",
                        help="Skip Meta Prompt-Guard (if license not accepted)")
    parser.add_argument("--embedded", action="store_true",
                        help="Test embedded-in-context (realistic scenario)")
    args = parser.parse_args()

    print("=" * 65)
    print("  Input-Level vs Behavioral Detection — Complementarity Analysis")
    print("=" * 65)

    # Build samples
    attack_samples = build_attack_samples()
    benign_samples = build_benign_samples()

    attack_by_lang = Counter(s["language"] for s in attack_samples)
    benign_by_lang = Counter(s["language"] for s in benign_samples)

    print(f"\n  Attack samples: {len(attack_samples)} total")
    for lang, count in sorted(attack_by_lang.items()):
        print(f"    {lang}: {count}")

    print(f"\n  Benign samples: {len(benign_samples)} total")
    for lang, count in sorted(benign_by_lang.items()):
        print(f"    {lang}: {count}")

    all_samples = attack_samples + benign_samples
    config = "embedded" if args.embedded else "payload_only"
    print(f"\n  Configuration: {config}")

    # ── Run input-level classifiers ──
    results_dict = {}
    metrics_dict = {}

    if not args.skip_protectai:
        print(f"\n{'─'*60}")
        print(f"  Running ProtectAI DeBERTa v2...")
        print(f"{'─'*60}")
        protectai_all = run_classifier(PROTECTAI_MODEL, all_samples, embedded=args.embedded)
        if protectai_all:
            protectai_attacks = [r for r, s in zip(protectai_all, all_samples) if s.get("source") != "benign"]
            protectai_benign = [r for r, s in zip(protectai_all, all_samples) if s.get("source") == "benign"]
            metrics = compute_task_a_metrics(protectai_attacks, protectai_benign, "ProtectAI DeBERTa v2")
            results_dict["protectai"] = protectai_all
            metrics_dict["protectai"] = metrics

    if not args.skip_promptguard:
        print(f"\n{'─'*60}")
        print(f"  Running Meta Prompt-Guard-2...")
        print(f"{'─'*60}")
        promptguard_all = run_classifier(PROMPTGUARD_MODEL, all_samples, embedded=args.embedded)
        if promptguard_all:
            promptguard_attacks = [r for r, s in zip(promptguard_all, all_samples) if s.get("source") != "benign"]
            promptguard_benign = [r for r, s in zip(promptguard_all, all_samples) if s.get("source") == "benign"]
            metrics = compute_task_a_metrics(promptguard_attacks, promptguard_benign, "Meta Prompt-Guard-2")
            results_dict["promptguard"] = promptguard_all
            metrics_dict["promptguard"] = metrics

    # ── Load behavioral results ──
    print(f"\n{'─'*60}")
    print(f"  Task B: AgentShield RF Classifier (existing results)")
    print(f"{'─'*60}")
    behavioral = load_behavioral_results()
    if behavioral:
        print(f"  F1 (cross-language): {behavioral.get('f1', 'N/A')}")
        print(f"  Precision: {behavioral.get('precision', 'N/A')}")
        print(f"  Recall: {behavioral.get('recall', 'N/A')}")
        print(f"  FPR: {behavioral.get('fpr', 'N/A')}")
    else:
        print("  No existing classifier results found — run train_classifier.py first")

    # ── Complementarity analysis ──
    compute_complementarity(attack_samples, results_dict, behavioral)

    # ── McNemar's test between classifiers ──
    if "protectai" in results_dict and "promptguard" in results_dict:
        print(f"\n{'─'*60}")
        print(f"  McNemar's Test: ProtectAI vs Prompt-Guard")
        print(f"{'─'*60}")
        pa_attacks = [r for r, s in zip(results_dict["protectai"], all_samples) if s.get("source") != "benign"]
        pg_attacks = [r for r, s in zip(results_dict["promptguard"], all_samples) if s.get("source") != "benign"]

        # Count disagreements
        n01 = 0  # ProtectAI missed, Prompt-Guard caught
        n10 = 0  # ProtectAI caught, Prompt-Guard missed
        for pa, pg in zip(pa_attacks, pg_attacks):
            pa_det = pa.get("is_injection", False) or False
            pg_det = pg.get("is_injection", False) or False
            if not pa_det and pg_det:
                n01 += 1
            elif pa_det and not pg_det:
                n10 += 1

        chi2, p = mcnemar_test(n01, n10)
        print(f"  Discordant pairs: ProtectAI-only={n10}, PromptGuard-only={n01}")
        print(f"  McNemar chi2={chi2:.3f}, p={p:.4f}")
        print(f"  {'Significant' if p < 0.05 else 'Not significant'} at alpha=0.05")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output = {
        "metadata": {
            "experiment": "complementarity_analysis",
            "timestamp": timestamp,
            "configuration": config,
            "attack_count": len(attack_samples),
            "benign_count": len(benign_samples),
            "attack_by_language": dict(attack_by_lang),
            "benign_by_language": dict(benign_by_lang),
            "models_tested": list(results_dict.keys()),
            "protectai_model": PROTECTAI_MODEL,
            "promptguard_model": PROMPTGUARD_MODEL,
        },
        "metrics": {},
        "per_sample_results": {},
    }

    for name, results in results_dict.items():
        output["per_sample_results"][name] = results
    for name, metrics in metrics_dict.items():
        output["metrics"][name] = metrics

    out_path = Path(f"agentshield/results/complementarity_analysis_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    for name, metrics in metrics_dict.items():
        print(f"\n  {name}:")
        print(f"    Overall recall: {metrics['recall_overall']:.1%}")
        for lang, data in metrics["recall_by_language"].items():
            print(f"    {lang} recall: {data['recall']:.1%} ({data['tp']}/{data['tp']+data['fn']})")
        for lang, data in metrics.get("fpr_by_language", {}).items():
            print(f"    {lang} FPR: {data['fpr']:.1%} ({data['fp']}/{data['fp']+data['tn']})")


if __name__ == "__main__":
    main()
