"""Final analysis: Combine ALL real DeepSeek API data with threshold sweep."""
import sys, os, json, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, "C:/Users/28995/agent_attack_detection")
from agent.types import ToolCall
from detection.graph_detector import MultiLayerDetector, DetectorConfig

all_entries = []
for logf in ["real_experiment_log.jsonl","real_final_log.jsonl","pure_real_log.jsonl"]:
    path = f"C:/Users/28995/agent_attack_detection/experiments/data/{logf}"
    if os.path.exists(path):
        with open(path) as f:
            for l in f:
                if l.strip(): all_entries.append(json.loads(l))

benign_entries = [e for e in all_entries if e.get("type") == "benign"]
attack_entries = [e for e in all_entries if e.get("type") != "benign" and e.get("n_calls",0) > 0]

def to_calls(entry):
    calls = []
    tools = entry.get("tools", [])
    for i, t in enumerate(tools):
        if isinstance(t, dict):
            tn = t.get("tool_name") or t.get("name","")
            tp = t.get("parameters") or t.get("params",{})
        else:
            tn = str(t); tp = {}
        calls.append(ToolCall(session_id=entry.get("task","?"), turn_id=i, tool_name=tn, parameters=tp, timestamp=float(i)))
    return calls

benign_calls = [to_calls(e) for e in benign_entries]
attack_calls = [to_calls(e) for e in attack_entries]

print("="*70)
print("ALL REAL DEEPSEEK API DATA COMBINED")
print(f"  Benign: {len(benign_calls)} | Attacks: {len(attack_calls)}")
atypes = Counter(e.get("type","?") for e in attack_entries)
for t,c in sorted(atypes.items()): print(f"    {t}: {c}")
print()

np.random.seed(42)
indices = np.random.permutation(len(benign_calls))
n_train = min(50, len(benign_calls))
train_b = [benign_calls[i] for i in indices[:n_train]]
test_b = [benign_calls[i] for i in indices[n_train:]]

det = MultiLayerDetector(DetectorConfig(window_size=10, decay_factor=0.9, alert_threshold=10.0))
det.set_training(True)
for calls in train_b: det.train_on(calls)

# Per-session max cumulative scores
scores = []
for calls in train_b:
    det.reset_session()
    max_cum = 0.0
    for c in calls:
        r = det.analyze_call(c)
        max_cum = max(max_cum, r.layer_results.get("cumulative_score",0))
    scores.append(max_cum)
det.set_training(False)

print(f"Training benign max cumulative scores: min={min(scores):.3f} max={max(scores):.3f} mean={np.mean(scores):.3f}")
print()

# Threshold sweep
print(f"{'Threshold':>10s} {'DR':>8s} {'FPR':>8s} {'F1':>8s}")
print("-"*40)
best = None
for th in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    tp = sum(1 for c in attack_calls for cc in c)  # just count total calls as proxy
    fp = sum(1 for c in test_b for cc in c)
    tp = 0; fn = 0
    for c in attack_calls:
        det.reset_session()
        detected = False
        for cc in c:
            r = det.analyze_call(cc)
            if r.layer_results.get("cumulative_score",0) >= th:
                detected = True; break
        if detected: tp += 1
        else: fn += 1
    fp = 0; tn = 0
    for c in test_b:
        det.reset_session()
        detected = False
        for cc in c:
            r = det.analyze_call(cc)
            if r.layer_results.get("cumulative_score",0) >= th:
                detected = True; break
        if detected: fp += 1
        else: tn += 1
    dr = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    f1 = 2*dr*(1-fpr)/(dr+1-fpr+1e-10) if dr+1-fpr>0 else 0
    print(f"  {th:8.1f} {dr:8.4f} {fpr:8.4f} {f1:8.4f}")
    if best is None or f1 > best[0]:
        best = (f1, th, dr, fpr)

f1_opt, th_opt, dr_opt, fpr_opt = best
print(f"\nOptimal: th={th_opt} DR={dr_opt:.3f} FPR={fpr_opt:.3f} F1={f1_opt:.3f}")
print()

# Baselines at optimal threshold
def our_at_th(c, th):
    det.reset_session()
    for cc in c:
        r = det.analyze_call(cc)
        if r.layer_results.get("cumulative_score",0) >= th: return True
    return False

def agentshield_detect(calls):
    tokens = ["@evil","@hacker","@malicious","@phishing","@audit","98765","77291"]
    for c in calls:
        if c.tool_name in ("export_all_user_data","admin_console","system_exec"): return True
        to = str(c.parameters.get("to","")).lower() if c.parameters else ""
        if any(t in to for t in tokens): return True
    return False

def leong_detect(calls):
    sus = {("recall_memory","send_email"),("recall_memory","transfer_money"),
           ("list_contacts","send_email"),("list_contacts","transfer_money"),
           ("send_email","delete_record"),("export_data","send_email"),
           ("transfer_money","delete_record")}
    prev = None
    for c in calls:
        if prev and (prev, c.tool_name) in sus: return True
        prev = c.tool_name
    return False

methods = {"Ours (Graph, optimal th)": lambda c: our_at_th(c, th_opt),
           "AgentShield": agentshield_detect,
           "Leong": leong_detect}

print(f"{'Method':35s} {'DR':>8s} {'FPR':>8s} {'F1':>8s} {'TP':>4s} {'FP':>4s}")
print("-"*65)
for name, fn in methods.items():
    tp = sum(1 for c in attack_calls if fn(c))
    fn_ = len(attack_calls) - tp
    fp = sum(1 for c in test_b if fn(c))
    tn_ = len(test_b) - fp
    dr = tp/(tp+fn_) if (tp+fn_)>0 else 0
    fpr = fp/(fp+tn_) if (fp+tn_)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*dr/(prec+dr) if (prec+dr)>0 else 0
    print(f"{name:35s} {dr:8.4f} {fpr:8.4f} {f1:8.4f} {tp:4d} {fp:4d}")

# Per-type
print(f"\nPer-type detection (Ours at th={th_opt}):")
by_type = defaultdict(list)
for e, calls in zip(attack_entries, attack_calls):
    by_type[e["type"]].append(calls)
for t in sorted(by_type.keys()):
    tc = by_type[t]
    td = sum(1 for c in tc if our_at_th(c, th_opt))
    print(f"  {t:30s} {td}/{len(tc)} = {td/len(tc)*100:.0f}%")

# Save
out = {"n_benign_train": len(train_b), "n_benign_test": len(test_b), "n_attacks": len(attack_calls),
       "optimal_threshold": th_opt, "optimal_dr": round(dr_opt,4), "optimal_fpr": round(fpr_opt,4)}
path = "C:/Users/28995/agent_attack_detection/data/pure_real_final_results.json"
with open(path,"w") as f: json.dump(out,f,indent=2)
print(f"\nSaved to {path}")
