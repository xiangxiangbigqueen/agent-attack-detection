"""Group-level bootstrap and paired comparisons for frozen results."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.canonical_protocol import all_episodes
from experiments.run_frozen_main_evaluation import BASE

SEED=20260729; B=10000
def ci(values):
    values=sorted(values); return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]
def main():
    main=json.loads((BASE/'main_evaluation.json').read_text()); base=json.loads((BASE/'baseline_evaluation.json').read_text())
    eps={e.group_id:e for e in all_episodes()}; scores={**main['scores']['benign_test'],**main['scores']['attack_test']}; th=main['threshold']
    ids=list(scores); rng=random.Random(SEED); metrics=[]
    for _ in range(B):
        sample=[rng.choice(ids) for _ in ids]; b=[x for x in sample if eps[x].label=='benign']; a=[x for x in sample if eps[x].label=='attack']
        fp=sum(scores[x]>=th for x in b); tp=sum(scores[x]>=th for x in a); p=tp/(tp+fp) if tp+fp else 0; r=tp/len(a)
        metrics.append((r,fp/len(b),2*p*r/(p+r) if p+r else 0))
    result={'bootstrap_repetitions':B,'seed':SEED,'main_ci95':{'dr':ci([x[0] for x in metrics]),'fpr':ci([x[1] for x in metrics]),'f1':ci([x[2] for x in metrics])},'baseline_disclosure':base['disclosure'],'baseline_metrics':{k:v['metrics'] for k,v in base['methods'].items()}}
    (BASE/'statistical_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
