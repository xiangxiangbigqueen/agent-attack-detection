"""Recalibrate using independent long benign data and retest frozen attacks."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import calls,group_score,selected_records,wilson

def load(root: Path, split: str):
 records=[]
 for path in (root/'records').glob('*.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row['split']==split and row['collector_status']=='ok': records.append(row)
 return records
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--raw-root',type=Path,default=ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1')
 parser.add_argument('--long-root',type=Path)
 parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 long_root=args.long_root or args.raw_root/'long_benign_calibration'
 output=args.output or args.raw_root/'long_calibrated_evaluation.json'
 data=selected_records(args.raw_root);eps={e.group_id:e for e in all_episodes()}; shorttrain=[calls(data[k][0]) for k,e in eps.items() if e.split=='train']; longtrain=load(long_root,'train'); train=shorttrain+[calls(x) for x in longtrain]
 val=[data[k] for k,e in eps.items() if e.split=='validation']+[[x] for x in load(long_root,'validation')]; val_scores=[group_score(x,train) for x in val]; th=float(np.quantile(val_scores,.95,method='higher'))
 short_ids=[k for k,e in eps.items() if e.label=='benign' and e.split=='test'];attack_ids=[k for k,e in eps.items() if e.label=='attack'];short=[data[k] for k in short_ids];long=[[x] for x in load(long_root,'test')];attacks=[data[k] for k in attack_ids];benign_scores={**{k:group_score(data[k],train) for k in short_ids},**{x['attack_id']:group_score([x],train) for x in load(long_root,'test')}};attack_scores={k:group_score(data[k],train) for k in attack_ids};bs=list(benign_scores.values());ats=list(attack_scores.values());fp=sum(x>=th for x in bs);tp=sum(x>=th for x in ats)
 by_type={};by_adaptation={}
 for attack_type in sorted({eps[k].attack_type for k in attack_ids}):
  values=[attack_scores[k] for k in attack_ids if eps[k].attack_type==attack_type];detected=sum(x>=th for x in values);by_type[attack_type]={'detected':detected,'total':len(values),'dr':detected/len(values),'wilson_95':wilson(detected,len(values))}
 for adaptation in ('A0','A1','A2','A3'):
  values=[attack_scores[k] for k in attack_ids if eps[k].adaptation_level==adaptation];detected=sum(x>=th for x in values);by_adaptation[adaptation]={'detected':detected,'total':len(values),'dr':detected/len(values),'wilson_95':wilson(detected,len(values))}
 r={'raw_root':str(args.raw_root),'long_root':str(long_root),'threshold_protocol':'P95 of independently collected short+long benign validation group maxima','threshold':th,'benign_test':len(bs),'attack_test':len(ats),'dr':tp/len(ats),'fpr':fp/len(bs),'tp':tp,'fp':fp,'dr_wilson_95':wilson(tp,len(ats)),'fpr_wilson_95':wilson(fp,len(bs)),'by_attack_type':by_type,'by_adaptation':by_adaptation,'scores':{'validation':val_scores,'benign':bs,'attack':ats,'benign_by_group':benign_scores,'attack_by_group':attack_scores}};output.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k!='scores'},indent=2))
if __name__=='__main__':main()
