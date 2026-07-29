"""Validation-calibrated component ablations on frozen trajectories."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detection.graph_detector import DetectorConfig, MultiLayerDetector
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import BASE, calls, selected_records, wilson


def score(records, train, config, preserve_sessions=True):
    detector = MultiLayerDetector(config)
    detector.set_training(True)
    for session in train: detector.train_on(session)
    detector.set_training(False)
    maximum = 0.0
    for index, record in enumerate(records):
        if index and preserve_sessions: detector.reset_session()
        elif index: detector = MultiLayerDetector(config); detector.set_training(True); [detector.train_on(s) for s in train]; detector.set_training(False)
        for call in calls(record):
            result = detector.analyze_call(call)
            maximum = max(maximum, result.layer_results['cumulative_score'] if config.use_cumulative else result.layer_results['final_score'])
    return maximum


def main():
    data, episodes = selected_records(BASE), {e.group_id:e for e in all_episodes()}
    train = [calls(data[k][0]) for k,e in episodes.items() if e.split == 'train']
    validation = [k for k,e in episodes.items() if e.split == 'validation']
    benign = [k for k,e in episodes.items() if e.label == 'benign' and e.split == 'test']
    attacks = [k for k,e in episodes.items() if e.label == 'attack']
    variants = {'Full': ({}, True), 'NoCrossSession': ({}, False), 'NoTransitionFrequency': ({'use_transition_frequency':False}, True), 'NoStructure': ({'use_structure':False}, True), 'NoCumulative': ({'use_cumulative':False}, True)}
    out = {'threshold_protocol':'P95 validation benign per variant','variants':{}}
    for name,(kwargs,preserve) in variants.items():
        cfg=DetectorConfig(min_baseline_samples=3, alert_threshold=1e9, **kwargs)
        threshold=float(np.quantile([score(data[k],train,cfg,preserve) for k in validation],.95,method='higher'))
        bs={k:score(data[k],train,cfg,preserve) for k in benign}; ats={k:score(data[k],train,cfg,preserve) for k in attacks}
        fp=sum(x>=threshold for x in bs.values()); tp=sum(x>=threshold for x in ats.values())
        out['variants'][name]={'threshold':threshold,'dr':tp/len(attacks),'fpr':fp/len(benign),'tp':tp,'fp':fp,'dr_wilson_95':wilson(tp,len(attacks)),'fpr_wilson_95':wilson(fp,len(benign))}
    path=BASE/'ablation_evaluation.json'; path.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
