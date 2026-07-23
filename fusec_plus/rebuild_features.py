"""Stage-0 reconstruction: regenerate features.npz + pair_texts.json (lost with a prior
workspace) for train_fusec.py.

Sources, all verified deterministic (rebuild_pools_check.py: 13,992/13,992 exact text match):
 - pair pools: replayed unified_candidates.py load_en/load_zh over out/<event>/*.jsonl and the
   cphot archive; pair_id "{eid}_{i}_{j}" indexes the pools, giving per-pair (ts_en, ts_zh).
 - temporal(7) per METHOD_SPEC.md §4(b):
   [1{EN→ZH}, 1{ZH→EN}, 1{concurrent}, 1{unknown}, sign(Δt), log(1+|Δt_h|), 1{missing}]
   with Δt = t_zh − t_en, concurrent ⇔ |Δt| < 24h, unknown/missing ⇔ either ts absent.
 - sim(5): [cos, cos², |cos|, max(cos,0), min(cos,0)] from pair_emb_v2.npz (BGE-M3, rows
   aligned to llm_labels_v2.jsonl) — same 5-d construction as arbiter_nli_offline.py.
 - split per METHOD_SPEC.md §5: train = GPT-5.5 silver on the 37 non-gold events (7,392 pairs,
   is_bridge/bridge_type → {no,A,B,C}); test = gold_consensus.jsonl v2.3 (2,798 pairs, 33 events).
"""
import json, math, re
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
import os
CPHOT = Path(os.environ.get("CPHOT_RAW_DIR", "cphot/raw"))  # cphot archive (Tier-3 re-collection or data release)
EN_PLATS = ["reddit", "x", "youtube", "hackernews", "bluesky", "telegram"]
ZH_PLATS = ["weibo", "bilibili", "zhihu", "douyin"]
EN_CAP, ZH_CAP, MINLEN = 1000, 1000, 8
CLASSES = ["no", "A", "B", "C"]; C2I = {c: i for i, c in enumerate(CLASSES)}

def _clean(t): return re.sub(r"\s+", " ", (t or "").strip())

def load_en(eid):
    seen, rows = set(), []
    for plat in EN_PLATS:
        f = OUT / eid / f"{plat}.jsonl"
        if not f.exists(): continue
        for l in f.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l); t = _clean(r.get("text"))
            if len(t) < MINLEN or t[:80] in seen: continue
            seen.add(t[:80]); rows.append({"text": t, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:EN_CAP]

def load_zh(eid):
    f = CPHOT / f"{eid}_001.json"
    seen, rows = set(), []
    if f.is_file():
        for p in json.loads(f.read_text()).get("posts", []):
            t = _clean(p.get("text"))
            if len(t) < MINLEN or t[:60] in seen: continue
            seen.add(t[:60]); rows.append({"text": t, "ts": p.get("timestamp")})
    else:
        for plat in ZH_PLATS:
            pf = OUT / eid / f"{plat}.jsonl"
            if not pf.exists(): continue
            for l in pf.read_text().splitlines():
                if not l.strip(): continue
                r = json.loads(l); t = _clean(r.get("text"))
                if len(t) < MINLEN or t[:60] in seen: continue
                seen.add(t[:60]); rows.append({"text": t, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:ZH_CAP]

def parse_ts(s):
    if not s: return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        pass
    try:  # twitter ctime: 'Sun Sep 01 23:57:07 +0000 2024'
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except (ValueError, TypeError):
        return None

def temporal7(te, tz):
    if te is None or tz is None:
        return [0., 0., 0., 1., 0., 0., 1.]
    dt_h = (tz - te) / 3600.0
    conc = abs(dt_h) < 24.0
    en2zh = (not conc) and dt_h > 0
    zh2en = (not conc) and dt_h < 0
    return [float(en2zh), float(zh2en), float(conc), 0.,
            float(np.sign(dt_h)), math.log1p(abs(dt_h)), 0.]

def sim5(c):
    return [c, c * c, abs(c), max(c, 0.0), min(c, 0.0)]

# --- per-pair timestamp + text lookup via replayed pools ---
pairs = {}
for l in open(HERE / "annotation_set_v2.jsonl"):
    if l.strip(): r = json.loads(l); pairs[r["pair_id"]] = r
by_event = {}
for r in pairs.values(): by_event.setdefault(r["event_id"], []).append(r)

ts_of = {}
for eid, rows in sorted(by_event.items()):
    en, zh = load_en(eid), load_zh(eid)
    for r in rows:
        i, j = (int(x) for x in r["pair_id"][len(eid) + 1:].split("_"))
        assert en[i]["text"][:400] == r["en_text"] and zh[j]["text"][:400] == r["zh_text"], r["pair_id"]
        ts_of[r["pair_id"]] = (parse_ts(en[i]["ts"]), parse_ts(zh[j]["ts"]))

# --- cosine per pair from pair_emb_v2 (rows aligned to llm_labels_v2.jsonl) ---
id2row = {json.loads(l)["pair_id"]: k for k, l in enumerate(open(HERE / "llm_labels_v2.jsonl")) if l.strip()}
emb = np.load(HERE / "pair_emb_v2.npz"); eu, zv = emb["eu"], emb["zv"]
eu = eu / (np.linalg.norm(eu, axis=1, keepdims=True) + 1e-8)
zv = zv / (np.linalg.norm(zv, axis=1, keepdims=True) + 1e-8)
cos_all = np.sum(eu * zv, axis=1)

# --- split ---
gold = [json.loads(l) for l in open(HERE / "gold_consensus.jsonl") if l.strip()]
gold_events = {g["event_id"] for g in gold}
silver = []
for l in open(HERE / "llm_labels_v2.jsonl"):
    r = json.loads(l)
    if r["event_id"] in gold_events: continue
    lab = "no" if r["is_bridge"] == "no" else r["bridge_type"]
    assert lab in CLASSES, (r["pair_id"], lab)
    silver.append((r["pair_id"], r["event_id"], lab))

def build(items):  # items: (pair_id, event_id, label)
    y = np.array([C2I[lab] for _, _, lab in items], dtype=np.int64)
    ev = np.array([e for _, e, _ in items])
    Xt = np.array([temporal7(*ts_of[p]) for p, _, _ in items], dtype=np.float32)
    Xs = np.array([sim5(float(cos_all[id2row[p]])) for p, _, _ in items], dtype=np.float32)
    return y, ev, Xt, Xs

tr_items = silver
te_items = [(g["pair_id"], g["event_id"], g["label"]) for g in gold]
ytr, evtr, Xtmp_tr, Xsim_tr = build(tr_items)
yte, evte, Xtmp_te, Xsim_te = build(te_items)

np.savez(HERE / "features.npz", ytr=ytr, yte=yte, evtr=evtr, evte=evte,
         Xtmp_tr=Xtmp_tr, Xsim_tr=Xsim_tr, Xtmp_te=Xtmp_te, Xsim_te=Xsim_te)
json.dump({
    "train": [{"pair_id": p, "en": pairs[p]["en_text"], "zh": pairs[p]["zh_text"]} for p, _, _ in tr_items],
    "test": [{"pair_id": p, "en": pairs[p]["en_text"], "zh": pairs[p]["zh_text"]} for p, _, _ in te_items],
}, open(HERE / "pair_texts.json", "w"), ensure_ascii=False)

print(f"train={len(tr_items)} (events={len(set(evtr))})  test={len(te_items)} (events={len(set(evte))})")
print("train label dist:", dict(zip(CLASSES, np.bincount(ytr, minlength=4))))
print("test  label dist:", dict(zip(CLASSES, np.bincount(yte, minlength=4))))
for nm, X in [("Xtmp_tr", Xtmp_tr), ("Xtmp_te", Xtmp_te)]:
    print(f"{nm}: unknown-ts rate={(X[:,3]==1).mean():.1%}  concurrent rate={(X[:,2]==1).mean():.1%}")
print("wrote features.npz + pair_texts.json")
