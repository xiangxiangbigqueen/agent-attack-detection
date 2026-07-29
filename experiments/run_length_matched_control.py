"""Length-matched negative control using the frozen long-calibrated detector."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import calls,group_score,selected_records

def records(root,split=None):
 out=[]
 for path in (root/'records').glob('*.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row.get('collector_status')=='ok' and (split is None or row.get('split')==split):out.append(row)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--long-root',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();evaluation=json.loads((a.evaluation or a.raw_root/'long_calibrated_evaluation.json').read_text(encoding='utf-8'));threshold=evaluation['threshold'];selected=selected_records(a.raw_root);episodes={episode.group_id:episode for episode in all_episodes()};train=[calls(selected[group][0]) for group,episode in episodes.items() if episode.split=='train']+[calls(row) for row in records(a.long_root,'train')]
 attack_by_length=defaultdict(list);benign_by_length=defaultdict(list)
 for group,episode in episodes.items():
  if episode.label=='attack':attack_by_length[sum(row['n_calls'] for row in selected[group])].append(evaluation['scores']['attack_by_group'][group])
  elif episode.split=='test':benign_by_length[sum(row['n_calls'] for row in selected[group])].append(evaluation['scores']['benign_by_group'][group])
 for row in records(a.long_root,'test')+records(a.control_root):benign_by_length[row['n_calls']].append(group_score([row],train))
 common=sorted(set(attack_by_length)&set(benign_by_length));matched={length:min(len(attack_by_length[length]),len(benign_by_length[length])) for length in common};total=sum(matched.values())
 if not total:raise SystemExit('blocked: no matched length support')
 tp=sum(sum(score>=threshold for score in attack_by_length[length][:matched[length]]) for length in common);fp=sum(sum(score>=threshold for score in benign_by_length[length][:matched[length]]) for length in common);out={'raw_root':str(a.raw_root),'long_root':str(a.long_root),'control_root':str(a.control_root),'threshold':threshold,'support':{str(length):{'attack':len(attack_by_length[length]),'benign':len(benign_by_length[length]),'matched':matched[length]} for length in common},'effective_per_class':total,'dr':tp/total,'fpr':fp/total,'tp':tp,'fp':fp,'disclosure':'Matched groups are truncated deterministically by collection order; this is a length-confounding diagnostic, not the primary estimate.'};destination=a.output or a.raw_root/'long_calibrated_length_matched.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
