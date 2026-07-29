"""Assess P90/P95/P99 validation thresholds without touching test data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,default=ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1'/'long_calibrated_evaluation.json');p.add_argument('--output',type=Path);a=p.parse_args()
 result=json.loads(a.evaluation.read_text(encoding='utf-8'));validation=result['scores']['validation'];benign=result['scores']['benign'];attack=result['scores']['attack'];out={}
 for quantile in (.90,.95,.99):
  threshold=float(np.quantile(validation,quantile,method='higher'));fp=sum(x>=threshold for x in benign);tp=sum(x>=threshold for x in attack);out[str(quantile)]={'threshold':threshold,'dr':tp/len(attack),'fpr':fp/len(benign),'tp':tp,'fp':fp}
 destination=a.output or a.evaluation.with_name('threshold_sensitivity.json');destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
