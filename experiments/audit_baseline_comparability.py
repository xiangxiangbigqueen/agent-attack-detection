"""Prevent invalid significance claims against non-comparable baselines."""
from __future__ import annotations
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1');a=p.parse_args();baseline=json.loads((a.root/'baseline_evaluation.json').read_text(encoding='utf-8'));main=json.loads((a.root/'long_calibrated_evaluation.json').read_text(encoding='utf-8'));methods=baseline.get('methods',{});out={'primary_benign_test_groups':main['benign_test'],'primary_attack_groups':main['attack_test'],'baseline_methods':list(methods),'baseline_disclosure':baseline.get('disclosure'),'eligible_for_paired_significance_test':False,'reason':'Existing baseline outputs were independently reimplemented and evaluated on the earlier 100-benign short-length test protocol; the primary result uses 200 benign test groups after long-length calibration. Paired significance would be invalid until every baseline is run on exactly the frozen primary trajectories and protocol.','required_before_claiming_comparison':['verified official baseline source or explicit independent-reimplementation label','same 200 benign and 240 attack groups','pre-specified calibration for each baseline','stored per-group predictions']};destination=a.root/'baseline_comparability_audit.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
