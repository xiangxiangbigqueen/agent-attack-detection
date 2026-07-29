"""Component ablations under the same independent long-benign protocol as the main result."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from detection.graph_detector import DetectorConfig,MultiLayerDetector
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import calls,selected_records,wilson

def long_rows(root,split):
 rows=[]
 for path in (root/'records').glob('*.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row['split']==split and row['collector_status']=='ok':rows.append(row)
 return rows
def score(records,train,config,preserve_sessions):
 detector=MultiLayerDetector(config);detector.set_training(True)
 for session in train:detector.train_on(session)
 detector.set_training(False);maximum=0.0
 for index,record in enumerate(records):
  if index:
   if preserve_sessions:detector.reset_session()
   else:
    detector=MultiLayerDetector(config);detector.set_training(True)
    for session in train:detector.train_on(session)
    detector.set_training(False)
  for call in calls(record):
   result=detector.analyze_call(call);maximum=max(maximum,float(result.layer_results['cumulative_score'] if config.use_cumulative else result.layer_results['final_score']))
 return maximum
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--long-root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();data=selected_records(a.raw_root);episodes={e.group_id:e for e in all_episodes()}
 train=[calls(data[k][0]) for k,e in episodes.items() if e.split=='train']+[calls(x) for x in long_rows(a.long_root,'train')];validation=[data[k] for k,e in episodes.items() if e.split=='validation']+[[x] for x in long_rows(a.long_root,'validation')];benign=[data[k] for k,e in episodes.items() if e.label=='benign' and e.split=='test']+[[x] for x in long_rows(a.long_root,'test')];attacks=[data[k] for k,e in episodes.items() if e.label=='attack']
 variants={'Full':({},True),'NoCrossSession':({},False),'NoTransitionFrequency':({'use_transition_frequency':False},True),'NoStructure':({'use_structure':False},True),'NoCumulative':({'use_cumulative':False},True)};out={'raw_root':str(a.raw_root),'long_root':str(a.long_root),'threshold_protocol':'P95 independent short+long benign validation per variant','variants':{}}
 for name,(kwargs,preserve) in variants.items():
  config=DetectorConfig(min_baseline_samples=3,alert_threshold=1e9,**kwargs);threshold=float(np.quantile([score(group,train,config,preserve) for group in validation],.95,method='higher'));bs=[score(group,train,config,preserve) for group in benign];ats=[score(group,train,config,preserve) for group in attacks];fp=sum(x>=threshold for x in bs);tp=sum(x>=threshold for x in ats);out['variants'][name]={'threshold':threshold,'benign_test':len(bs),'attack_test':len(ats),'dr':tp/len(ats),'fpr':fp/len(bs),'tp':tp,'fp':fp,'dr_wilson_95':wilson(tp,len(ats)),'fpr_wilson_95':wilson(fp,len(bs))}
 destination=a.output or a.raw_root/'long_calibrated_ablation.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
