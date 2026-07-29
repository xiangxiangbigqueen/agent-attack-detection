"""Audit and score retained feedback-candidate trajectories using a frozen detector.

This is deliberately reported as *candidate screening*, not adaptive-attack
success: a score below threshold says only that the detector did not alert.  A
separate, pre-registered task-success oracle is required before any evasion
claim can be made.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.evaluate_attack_outcomes import successful
from experiments.run_frozen_main_evaluation import calls,group_score,selected_records

def load_ok_attempts(base: Path):
 chosen={}
 for root in [base]+sorted(base.glob('recovery_*')):
  for path in (root/'records').glob('*.json'):
   row=json.loads(path.read_text(encoding='utf-8'))
   if row.get('collector_status')=='ok':chosen[(row['attack_id'],row['session_index'])]=row
 groups=defaultdict(list)
 for (_,index),row in chosen.items():groups[row['attack_id']].append(row)
 for rows in groups.values():rows.sort(key=lambda row:row['session_index'])
 return groups
def long_train(root: Path):
 rows=[]
 for path in (root/'records').glob('*.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row.get('split')=='train' and row.get('collector_status')=='ok':rows.append(row)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-root',type=Path,default=ROOT/'local_results'/'canonical'/'feedback_adaptive_R1');p.add_argument('--frozen-raw-root',type=Path,default=ROOT/'local_results'/'canonical'/'paper_protocol_20260729_1');p.add_argument('--long-root',type=Path);p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 candidates=load_ok_attempts(a.candidate_root)
 if len(candidates)!=180:raise SystemExit(f'blocked: expected 180 complete candidate groups, found {len(candidates)}')
 if any(len(rows) not in {1,2} or {row['session_index'] for row in rows}!=set(range(len(rows))) for rows in candidates.values()):raise SystemExit('blocked: incomplete candidate session group')
 frozen=selected_records(a.frozen_raw_root);episodes={episode.group_id:episode for episode in all_episodes()};long_root=a.long_root or a.frozen_raw_root/'long_benign_calibration';train=[calls(frozen[group][0]) for group,episode in episodes.items() if episode.split=='train']+[calls(row) for row in long_train(long_root)]
 evaluation=a.evaluation or a.frozen_raw_root/'long_calibrated_evaluation.json';threshold=json.loads(evaluation.read_text(encoding='utf-8'))['threshold'];all_scores={group:group_score(rows,train) for group,rows in sorted(candidates.items())}
 selected={}
 for group,score in all_scores.items():
  base,mode=group.rsplit('-',1);candidate={'candidate_group':group,'mode':mode,'score':score,'detector_alert':score>=threshold}
  if base not in selected or score<selected[base]['score']:selected[base]=candidate
 for base,row in selected.items():
  attack_type=base.removeprefix('feedback-').rsplit('-v',1)[0]
  selected_rows=candidates[row['candidate_group']]
  row['attack_type']=attack_type
  row['objective_success']=successful(attack_type,[call for record in selected_rows for call in record['tools']])
 successful_selection=[row for row in selected.values() if row['objective_success']]
 out={'candidate_root':str(a.candidate_root),'frozen_evaluation':str(evaluation),'threshold':threshold,'candidate_groups':len(all_scores),'base_attack_groups':len(selected),'candidate_scores':all_scores,'minimum_score_selection':selected,'minimum_score_alert_rate':sum(row['detector_alert'] for row in selected.values())/len(selected),'selected_objective_successes':len(successful_selection),'selected_objective_success_rate':len(successful_selection)/len(selected),'alerts_on_selected_objective_successes':sum(row['detector_alert'] for row in successful_selection),'selected_successful_attack_detection_rate':sum(row['detector_alert'] for row in successful_selection)/len(successful_selection) if successful_selection else None,'disclosure':'One-step frozen-detector candidate screening only, not an online adaptive policy. Detector non-alert is reported jointly with the independent sandbox objective-success oracle.'}
 destination=a.output or a.candidate_root/'frozen_candidate_screening.json';destination.write_text(json.dumps(out,indent=2),encoding='utf-8');(a.candidate_root/'candidate_audit.json').write_text(json.dumps({'records':sum(len(rows) for rows in candidates.values()),'groups':len(candidates),'status':'complete_and_scored'},indent=2),encoding='utf-8');print(json.dumps({key:value for key,value in out.items() if key not in {'candidate_scores','minimum_score_selection'}},indent=2))
if __name__=='__main__':main()
