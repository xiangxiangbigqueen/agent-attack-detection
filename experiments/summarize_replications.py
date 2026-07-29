"""Aggregate independently collected R1/R2/R3 long-calibrated results."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'local_results'/'canonical'/'replication_summary.json');a=p.parse_args();roots=[ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1',ROOT/'local_results'/'canonical'/'replication_R2'/'raw',ROOT/'local_results'/'canonical'/'replication_R3'/'raw'];runs=[]
 for root in roots:
  result=json.loads((root/'long_calibrated_evaluation.json').read_text(encoding='utf-8'));runs.append({'root':str(root),'dr':result['dr'],'fpr':result['fpr'],'threshold':result['threshold'],'by_attack_type':result['by_attack_type'],'by_adaptation':result['by_adaptation']})
 metric=lambda name:[run[name] for run in runs];out={'protocol':'three independently collected runs; each calibrates threshold on its own independent short+long benign validation data','runs':runs,'aggregate':{'dr_mean':statistics.mean(metric('dr')),'dr_sample_sd':statistics.stdev(metric('dr')),'fpr_mean':statistics.mean(metric('fpr')),'fpr_sample_sd':statistics.stdev(metric('fpr')),'threshold_mean':statistics.mean(metric('threshold')),'threshold_sample_sd':statistics.stdev(metric('threshold'))},'disclosure':'Runs share the same pre-specified prompts, model family, tool environment, and protocol; they measure collection/API stochasticity, not cross-model generalization.'};a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out['aggregate'],indent=2))
if __name__=='__main__':main()
