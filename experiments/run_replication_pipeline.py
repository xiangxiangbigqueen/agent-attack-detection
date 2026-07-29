"""Generic post-collection pipeline for an independently collected replication."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.validate_canonical_dataset import audit

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('long_root',type=Path);a=p.parse_args()
 report=audit(a.root)
 if not report['valid']:raise SystemExit('blocked: raw replication audit failed')
 long=[]
 for f in (a.long_root/'records').glob('*.json'):long.append(json.loads(f.read_text(encoding='utf-8')))
 counts={s:sum(x['split']==s and x['collector_status']=='ok' for x in long) for s in ('train','validation','test')}
 if counts!={'train':100,'validation':50,'test':100}:raise SystemExit(f'blocked: long benign split incomplete: {counts}')
 out={'raw_audit':report,'long_benign_counts':counts,'ready_for_calibrated_evaluation':True}
 (a.root/'pipeline_ready.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 summary={key:value for key,value in report.items() if key not in {'group_provenance','missing_groups'}}
 summary['long_benign_counts']=counts
 summary['ready_for_calibrated_evaluation']=True
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
