"""Run disclosed independent baseline reimplementations on the exact primary test set."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from detection.baselines import AgentShieldDetector,LeongTrajectoryDetector
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import calls,selected_records

def shield():
 detector=AgentShieldDetector()
 for tool in ('export_all_data','admin_console','system_exec'):detector.add_honeytool(tool)
 for address in ('alice@company.com','bob@partner.com','carol@company.com','david@external.com'):detector.allow_param('send_email',address)
 detector.allow_param('transfer_money','savings');return detector
def predict(factory,rows):
 detector=factory()
 for session_index,row in enumerate(rows):
  if session_index:detector.reset()
  if any(detector.detect(call)[0] for call in calls(row)):return True
 return False
def load_long_test(root):
 rows=[]
 for path in (root/'records').glob('*.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row.get('split')=='test' and row.get('collector_status')=='ok':rows.append(row)
 return rows
def metric(predictions,benign,attack):
 fp=sum(predictions[group] for group in benign);tp=sum(predictions[group] for group in attack);precision=tp/(tp+fp) if tp+fp else 0;dr=tp/len(attack);return {'tp':tp,'fp':fp,'dr':dr,'fpr':fp/len(benign),'precision':precision,'f1':2*precision*dr/(precision+dr) if precision+dr else 0}
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--long-root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();data=selected_records(a.raw_root);episodes={episode.group_id:episode for episode in all_episodes()};short_benign=[group for group,episode in episodes.items() if episode.label=='benign' and episode.split=='test'];attack=[group for group,episode in episodes.items() if episode.label=='attack'];long=load_long_test(a.long_root);long_ids=[row['attack_id'] for row in long];groups={**{group:data[group] for group in short_benign+attack},**{row['attack_id']:[row] for row in long}};benign=short_benign+long_ids;out={'created_at_utc':datetime.now(timezone.utc).isoformat(),'raw_root':str(a.raw_root),'long_root':str(a.long_root),'test_counts':{'benign':len(benign),'attack':len(attack)},'disclosure':'Independent reimplementations, not official baseline code. Baselines retain their own documented non-learning rule configuration; no threshold was tuned on test labels.','session_policy':'Each baseline is reset between sessions, so it is intentionally a per-session baseline and does not receive cross-session memory.','methods':{}}
 for name,factory in (('AgentShield_style',shield),('Trajectory_rule',LeongTrajectoryDetector)):
  predictions={group:predict(factory,groups[group]) for group in groups};out['methods'][name]={'metrics':metric(predictions,benign,attack),'predictions':predictions}
 destination=a.output or a.raw_root/'long_protocol_baselines.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({name:body['metrics'] for name,body in out['methods'].items()},indent=2))
if __name__=='__main__':main()
