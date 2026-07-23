"""Descriptive standalone CIs for the FUSE-C+ arbiter (and SC-majority) rows/figures.

Reproduces the V2 arbiter's OOF predictions exactly (same features, folds, seed as
fusecplus_validate_v2.py), verifies the point estimates match fusecplus_v2_result.json,
then computes 95% cluster-bootstrap CIs (5,000 resamples, seed 20260101) for macro-F1 and
C-F1 on full gold and core854. Descriptive only — the pre-registered decision test is
untouched (fusecplus_v2_result.json remains the verdict artifact).
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
CLASSES = ["no","A","B","C"]; C2I = {c:i for i,c in enumerate(CLASSES)}
SEED = 20260101; BOOT = 5000

def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]

gold = {r["pair_id"]: r for r in load_jsonl(HERE/"gold_consensus.jsonl")}
strong = {p: C2I[l] for p, l in json.load(open(HERE/"fewshot_gpt54nano_preds.json")).items() if l in C2I}
lp = {r["pair_id"]: np.array([r["llm_p"][c] for c in CLASSES]) for r in load_jsonl(HERE/"llm_probs.jsonl")}
votes = {}
for r in load_jsonl(HERE/"llm_votes.jsonl"):
    votes[r["pair_id"]] = np.array([r["votes"][c] for c in CLASSES], float) / max(r["k"], 1)
ens = []
for sfx in ["", "_seed1", "_seed2"]:
    ens.append({p: np.array(v, float) for p, v in json.load(open(HERE/f"student_probs{sfx}.json")).items()})
sp = {p: np.mean([e[p] for e in ens], 0) for p in ens[0] if all(p in e for e in ens)}
id2row = {json.loads(l)["pair_id"]: i for i, l in enumerate(open(HERE/"llm_labels_v2.jsonl")) if l.strip()}
emb = np.load(HERE/"pair_emb_v2.npz"); eu, zv = emb["eu"], emb["zv"]

ids = [p for p in gold if p in strong and p in lp and p in votes and p in sp and p in id2row]
y = np.array([C2I[gold[p]["label"]] for p in ids])
groups = np.array([gold[p]["event_id"] for p in ids])
rows = np.array([id2row[p] for p in ids])
e = eu[rows]; z = zv[rows]
e = e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-8); z = z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-8)
cos = np.sum(e*z,axis=1)
sim = np.stack([cos, cos**2, np.abs(cos), np.maximum(cos,0), np.minimum(cos,0)],1)
S   = np.vstack([sp[p] for p in ids])
Mlp = np.vstack([lp[p] for p in ids])
V   = np.vstack([votes[p] for p in ids])
m_hard = np.array([strong[p] for p in ids]); Mst = np.eye(4)[m_hard]
v_major = V.argmax(1); s_hard = S.argmax(1)
marg = lambda P: np.sort(P,1)[:,-1] - np.sort(P,1)[:,-2]
s_m, lp_m, v_m = marg(S), marg(Mlp), marg(V)
v_ent = -np.sum(np.where(V>0, V*np.log(V+1e-12), 0), 1)
disagree = (s_hard != m_hard).astype(float)
sBmC = ((s_hard==2) & (m_hard==3)).astype(float)
lpC, vC = Mlp[:,3], V[:,3]
X = np.hstack([S, Mst, Mlp, V, sim, s_m[:,None], lp_m[:,None], v_m[:,None], v_ent[:,None],
               disagree[:,None], sBmC[:,None], lpC[:,None], vC[:,None], (lpC*s_m)[:,None], (vC*s_m)[:,None]])

def oof(factory):
    gkf = GroupKFold(5); pred = np.zeros(len(y), int)
    for tr, te in gkf.split(X, y, groups):
        pred[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return pred
pred_lr  = oof(lambda: LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
pred_mlp = oof(lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=1500, random_state=SEED))
cf = lambda yy, pp: f1_score(yy, pp, labels=[3], average="macro", zero_division=0)
mM = lambda yy, pp: f1_score(yy, pp, average="macro", zero_division=0)
best = pred_lr if cf(y, pred_lr) >= cf(y, pred_mlp) else pred_mlp

# verify exact reproduction of the frozen V2 point estimates
assert abs(cf(y, best) - 0.391) < 5e-4, cf(y, best)
assert abs(mM(y, best) - 0.457) < 5e-4, mM(y, best)
print(f"reproduction verified: full-gold C-F1={cf(y,best):.4f} macro={mM(y,best):.4f}")

key = {str(r["audit_id"]): r["pair_id"] for r in load_jsonl(HERE/"expert_audit/expert_audit_key.jsonl")}
core = {}; core500_pids = set()
for r in csv.DictReader(open(HERE/"expert_audit/expert_audit_sheet.csv")):
    lab = r["your_label (no/A/B/C)"].strip(); pid = key.get(str(r["audit_id"]))
    if pid in gold and lab in C2I:
        core[pid] = C2I[lab]
        if int(r["audit_id"]) < 500: core500_pids.add(pid)  # batches 1-2 = the original 500-pair audit
idx = np.array([k for k, p in enumerate(ids) if p in core])
y_core = np.array([core[ids[k]] for k in idx])
idx500 = np.array([k for k, p in enumerate(ids) if p in core500_pids])
y_500 = np.array([core[ids[k]] for k in idx500])
# verify core500 identification against the manuscript's few-shot numbers
assert len(idx500) == 500 and (y_500 == 3).sum() == 147
assert abs(cf(y_500, m_hard[idx500]) - 0.548) < 5e-4 and abs(mM(y_500, m_hard[idx500]) - 0.525) < 5e-4
assert abs(cf(y_core, best[idx]) - 0.609) < 5e-4, cf(y_core, best[idx])
assert abs(mM(y_core, best[idx]) - 0.524) < 5e-4, mM(y_core, best[idx])
print(f"reproduction verified: core854 C-F1={cf(y_core,best[idx]):.4f} macro={mM(y_core,best[idx]):.4f}")

def boot_ci(pred, yy, gg, metric):
    uniq = np.unique(gg); rng = np.random.default_rng(SEED); vals = []
    for _ in range(BOOT):
        samp = rng.choice(uniq, len(uniq), True)
        mask = np.concatenate([np.where(gg==g)[0] for g in samp])
        vals.append(metric(yy[mask], pred[mask]))
    vals = np.array(vals)
    return [float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))]

def cell(pred, yy, gg):
    return {"C_F1": cf(yy,pred), "C_ci": boot_ci(pred,yy,gg,cf),
            "macro": mM(yy,pred), "m_ci": boot_ci(pred,yy,gg,mM)}

out = {
  "arbiter_full":    cell(best, y, groups),
  "arbiter_core854": cell(best[idx], y_core, groups[idx]),
  "arbiter_core500": cell(best[idx500], y_500, groups[idx500]),
  "scmaj_full":      cell(v_major, y, groups),
  "scmaj_core854":   cell(v_major[idx], y_core, groups[idx]),
  "scmaj_core500":   cell(v_major[idx500], y_500, groups[idx500]),
}
json.dump(out, open(HERE/"arbiter_standalone_cis.json","w"), indent=1)
for k, v in out.items():
    print(f"{k}: C-F1 {v['C_F1']:.3f} [{v['C_ci'][0]:.3f},{v['C_ci'][1]:.3f}]  "
          f"macro {v['macro']:.3f} [{v['m_ci'][0]:.3f},{v['m_ci'][1]:.3f}]")
