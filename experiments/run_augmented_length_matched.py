"""Length-matched evaluation with the stratified benign controls."""
from __future__ import annotations
import json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import BASE,calls,group_score,selected_records

def main():
 main=json.loads((BASE/'main_evaluation.json').read_text()); data=selected_records(BASE); eps={e.group_id:e for e in all_episodes()}; train=[calls(data[k][0]) for k,e in eps.items() if e.split=='train']; th=main['threshold']
 b,a=defaultdict(list),defaultdict(list)
 for k,e in eps.items():
  if e.label=='attack': a[sum(x['n_calls'] for x in data[k])].append(main['scores']['attack_test'][k])
  elif e.split=='test': b[sum(x['n_calls'] for x in data[k])].append(main['scores']['benign_test'][k])
 control=BASE/'length_control_stratified'/'records'
 for p in control.glob('*.json'):
  r=json.loads(p.read_text(encoding='utf-8')); b[r['n_calls']].append(group_score([r],train))
 common=sorted(set(a)&set(b)); n=sum(min(len(a[x]),len(b[x])) for x in common); tp=sum(sum(x>=th for x in a[l][:min(len(a[l]),len(b[l]))]) for l in common); fp=sum(sum(x>=th for x in b[l][:min(len(a[l]),len(b[l]))]) for l in common)
 result={'threshold':th,'support':{str(l):{'attack':len(a[l]),'benign':len(b[l]),'matched':min(len(a[l]),len(b[l]))} for l in common},'effective_per_class':n,'dr':tp/n,'fpr':fp/n,'tp':tp,'fp':fp}
 (BASE/'augmented_length_matched_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
