"""Exact paired sign tests on identical primary-test predictions."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def exact_two_sided(b,c):
 n=b+c
 if not n:return 1.0
 tail=sum(math.comb(n,k) for k in range(0,min(b,c)+1))/(2**n)
 return min(1.0,2*tail)
def compare(ours,baseline,ids):
 b=sum(ours[group] and not baseline[group] for group in ids);c=sum(not ours[group] and baseline[group] for group in ids);return {'ours_only':b,'baseline_only':c,'exact_two_sided_p':exact_two_sided(b,c)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1');a=p.parse_args();main=json.loads((a.root/'long_calibrated_evaluation.json').read_text(encoding='utf-8'));baselines=json.loads((a.root/'long_protocol_baselines.json').read_text(encoding='utf-8'));ours={**{group:score>=main['threshold'] for group,score in main['scores']['benign_by_group'].items()},**{group:score>=main['threshold'] for group,score in main['scores']['attack_by_group'].items()}};benign=list(main['scores']['benign_by_group']);attack=list(main['scores']['attack_by_group']);out={'eligibility':'same frozen 200 benign + 240 attack groups; exact McNemar/binomial test, two-sided, unadjusted','comparisons':{}}
 for name,body in baselines['methods'].items():
  baseline=body['predictions'];out['comparisons'][name]={'attack_detection':compare(ours,baseline,attack),'benign_alert_rate':compare(ours,baseline,benign),'disclosure':'For benign alerts, ours_only means only the primary method alerts; lower is preferable, so this is not a superiority direction claim.'}
 destination=a.root/'paired_baseline_statistics.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
