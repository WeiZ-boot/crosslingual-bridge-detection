"""FUSE-C+ V2 validation — implements PREREGISTRATION_FUSECPLUS_V2.md exactly. Run ONCE.

Inputs: gold_consensus.jsonl, fewshot_gpt54nano_preds.json, llm_probs.jsonl, llm_votes.jsonl,
student_probs.json + student_probs_seed1.json + student_probs_seed2.json,
pair_emb_v2.npz + llm_labels_v2.jsonl, expert_audit/{expert_audit_key.jsonl,expert_audit_sheet.csv}.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
CLASSES = ["no","A","B","C"]; C2I = {c:i for i,c in enumerate(CLASSES)}
SEED = 20260101; BOOT = 5000

def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]

def main():
    gold = {r["pair_id"]: r for r in load_jsonl(HERE/"gold_consensus.jsonl")}
    strong = {p: C2I[l] for p, l in json.load(open(HERE/"fewshot_gpt54nano_preds.json")).items() if l in C2I}
    lp = {r["pair_id"]: np.array([r["llm_p"][c] for c in CLASSES]) for r in load_jsonl(HERE/"llm_probs.jsonl")}
    votes = {}
    for r in load_jsonl(HERE/"llm_votes.jsonl"):
        k = max(r["k"], 1)
        votes[r["pair_id"]] = np.array([r["votes"][c] for c in CLASSES], float) / k
    ens = []
    for sfx in ["", "_seed1", "_seed2"]:
        ens.append({p: np.array(v, float) for p, v in json.load(open(HERE/f"student_probs{sfx}.json")).items()})
    sp = {p: np.mean([e[p] for e in ens], 0) for p in ens[0] if all(p in e for e in ens)}

    id2row = {json.loads(l)["pair_id"]: i for i, l in enumerate(open(HERE/"llm_labels_v2.jsonl")) if l.strip()}
    emb = np.load(HERE/"pair_emb_v2.npz"); eu, zv = emb["eu"], emb["zv"]

    ids = [p for p in gold if p in strong and p in lp and p in votes and p in sp and p in id2row]
    print(f"[n] usable = {len(ids)} (gold {len(gold)}, votes {len(votes)}, students {len(sp)})")
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
    m_hard = np.array([strong[p] for p in ids])
    Mst = np.eye(4)[m_hard]
    v_major = V.argmax(1)                                     # SC-majority (secondary comparison)
    s_hard = S.argmax(1)
    marg = lambda P: np.sort(P,1)[:,-1] - np.sort(P,1)[:,-2]
    s_m, lp_m, v_m = marg(S), marg(Mlp), marg(V)
    v_ent = -np.sum(np.where(V>0, V*np.log(V+1e-12), 0), 1)
    disagree = (s_hard != m_hard).astype(float)
    sBmC = ((s_hard==C2I["B"]) & (m_hard==C2I["C"])).astype(float)
    lpC, vC = Mlp[:,3], V[:,3]

    X = np.hstack([S, Mst, Mlp, V, sim,
                   s_m[:,None], lp_m[:,None], v_m[:,None], v_ent[:,None],
                   disagree[:,None], sBmC[:,None],
                   lpC[:,None], vC[:,None], (lpC*s_m)[:,None], (vC*s_m)[:,None]])
    print(f"[features] dim = {X.shape[1]}")

    def oof(factory):
        gkf = GroupKFold(5); pred = np.zeros(len(y), int)
        for tr, te in gkf.split(X, y, groups):
            pred[te] = factory().fit(X[tr], y[tr]).predict(X[te])
        return pred
    pred_lr  = oof(lambda: LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
    pred_mlp = oof(lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=1500, random_state=SEED))
    cf = lambda yy, pp: f1_score(yy, pp, labels=[3], average="macro", zero_division=0)
    best = pred_lr if cf(y, pred_lr) >= cf(y, pred_mlp) else pred_mlp
    best_name = "LR" if best is pred_lr else "MLP"

    def rep(pred, name, yy=y):
        return (f"{name:22s} macro={f1_score(yy,pred,average='macro'):.3f} C-F1={cf(yy,pred):.3f} "
                f"C-rec={recall_score(yy,pred,labels=[3],average='macro',zero_division=0):.3f} "
                f"B-prec={precision_score(yy,pred,labels=[2],average='macro',zero_division=0):.3f}")
    for p, n in [(np.vstack([sp[p] for p in ids]).argmax(1),"student-ens"),(m_hard,"strong LLM (1-call)"),
                 (v_major,"SC-majority LLM"),(pred_lr,"arbiter (LR)"),(pred_mlp,"arbiter (MLP)")]:
        print(rep(p, n))

    mM  = lambda yy,pp: f1_score(yy,pp,average="macro",zero_division=0)
    mBp = lambda yy,pp: precision_score(yy,pp,labels=[2],average="macro",zero_division=0)
    def boot(a, b, metric, yy, gg):
        uniq = np.unique(gg); rng = np.random.default_rng(SEED); ds = []
        for _ in range(BOOT):
            samp = rng.choice(uniq, len(uniq), True)
            mask = np.concatenate([np.where(gg==g)[0] for g in samp])
            ds.append(metric(yy[mask], a[mask]) - metric(yy[mask], b[mask]))
        ds = np.array(ds)
        return {"delta": float(ds.mean()),
                "ci": [float(np.percentile(ds,2.5)), float(np.percentile(ds,97.5))],
                "p": float(2*min((ds<=0).mean(),(ds>=0).mean()))}
    def bar(r_c, r_bp):
        return r_c["delta"]>0 and r_c["ci"][0]>0 and r_c["p"]<0.05 and not (r_bp["delta"]<0 and r_bp["ci"][1]<0)

    out = {"n": len(ids), "arbiter": best_name, "prereg": "PREREGISTRATION_FUSECPLUS_V2.md"}
    print(f"\n=== TEST A (full gold): arbiter[{best_name}] vs strong single-call LLM ===")
    A = {nm: boot(best, m_hard, mt, y, groups) for nm, mt in [("C_F1",cf),("macro",mM),("B_prec",mBp)]}
    for nm, r in A.items(): print(f"  Δ{nm:6s}={r['delta']:+.4f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.4f}")
    a_pass = bar(A["C_F1"], A["B_prec"]); out["testA"] = A; out["testA_pass"] = bool(a_pass)

    print("--- honesty secondaries (non-gating) ---")
    out["arb_vs_scmaj"] = boot(best, v_major, cf, y, groups)
    out["scmaj_vs_single"] = boot(v_major, m_hard, cf, y, groups)
    print(f"  arbiter vs SC-majority  ΔC-F1={out['arb_vs_scmaj']['delta']:+.4f} p={out['arb_vs_scmaj']['p']:.4f}")
    print(f"  SC-majority vs 1-call   ΔC-F1={out['scmaj_vs_single']['delta']:+.4f} p={out['scmaj_vs_single']['p']:.4f}")

    # --- Test B: core854, true expert labels ---
    key = {str(r["audit_id"]): r["pair_id"] for r in load_jsonl(HERE/"expert_audit/expert_audit_key.jsonl")}
    core = {}
    for r in csv.DictReader(open(HERE/"expert_audit/expert_audit_sheet.csv")):
        lab = r["your_label (no/A/B/C)"].strip()
        pid = key.get(str(r["audit_id"]))
        if pid in gold and lab in C2I: core[pid] = C2I[lab]
    idx = np.array([k for k, p in enumerate(ids) if p in core])
    y_core = np.array([core[ids[k]] for k in idx])
    print(f"\n=== TEST B (core854, expert labels): n={len(idx)}, events={len(set(groups[idx]))} ===")
    for p, n in [(m_hard[idx],"strong LLM (1-call)"),(best[idx],f"arbiter ({best_name})")]:
        print(rep(p, n, yy=y_core))
    B = {nm: boot(best[idx], m_hard[idx], mt, y_core, groups[idx])
         for nm, mt in [("C_F1",cf),("macro",mM),("B_prec",mBp)]}
    for nm, r in B.items(): print(f"  Δ{nm:6s}={r['delta']:+.4f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.4f}")
    b_pass = bar(B["C_F1"], B["B_prec"]); out["testB"] = B; out["testB_pass"] = bool(b_pass)
    out["core_n"] = int(len(idx))

    verdict = ("SHIP (full-gold claim)" if a_pass else
               "SHIP (core854-scoped claim only)" if b_pass else
               "DO NOT SHIP — keep near-parity framing; remaining route is gold expansion")
    out["DECISION"] = verdict
    print(f"\n>>> DECISION: {verdict}")
    json.dump(out, open(HERE/"fusecplus_v2_result.json","w"), indent=1)
    print("wrote fusecplus_v2_result.json")

if __name__ == "__main__":
    main()
