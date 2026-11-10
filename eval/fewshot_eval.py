"""(a) Few-shot LLM system: how high does C-F1 go on the consensus gold?

Non-circular: GPT-5.5 and Claude are the gold annotators, so they are EXCLUDED. Valid systems here
are gpt-5.4-nano (== our zero-shot baseline's model, for a clean 'does richer few-shot help' test)
and gpt-5.4 (stronger non-teacher model, for the ceiling; mild same-family caveat vs the 5.5 gold).

Richer few-shot than classify_pairs.py: more diverse C exemplars + explicit B-vs-C contrast, all
hand-crafted/paraphrased (NOT drawn from the gold) to avoid test leakage. Concurrent API calls.
Scores vs gold_consensus.jsonl (enriched, 1771 pairs), saves preds, prints per-class F1 + confusion.

Run:  ~/opt/anaconda3/envs/dify/bin/python fewshot_eval.py --model gpt-5.4-nano
      ~/opt/anaconda3/envs/dify/bin/python fewshot_eval.py --model gpt-5.4
"""
from __future__ import annotations
import argparse, json, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
CLASSES = ["no", "A", "B", "C"]

SYSTEM = """You are an expert annotator of CROSS-LINGUAL information bridges between English social
media and Chinese social media about the same hot-topic event. Output ONLY a JSON object.

is_bridge: "yes" if the Chinese post carries the SAME specific event/sub-event/claim across the
language boundary (translating, reframing, or responding to it); "no" if merely the same broad
topic, a DIFFERENT sub-event, or coincidental keyword overlap.

bridge_type (only if is_bridge=="yes") — judge what the CHINESE text does RELATIVE TO the English:
  A = faithful restatement: SAME facts AND framing in Chinese, no added opinion/angle/rebuttal
      (translation, official statement/obituary restated, factual photo/figure/list).
  B = reframing: same event but ADDS a non-oppositional angle/emphasis/opinion or recentres on China,
      WITHOUT directly rebutting the English framing (commentary, analysis, celebration, China-angle).
  C = counter-narrative: explicitly rebuts / mocks / inverts the Western framing, or asserts the
      opposite stance.

DECISIVE B-vs-C RULE: C requires the Chinese post to OPPOSE the English post's stance. If the English
post is already anti-US/critical and the Chinese post merely agrees or reinforces it -> B (not C). If
the English post is neutral/pro-Western and the Chinese post pushes back -> C.

EXEMPLARS (illustrative, paraphrased):
- EN: "Pentagon releases U-2 photo of the Chinese spy balloon." / ZH: "五角大楼公布U-2拍摄的中国气球照片。"
  -> {"is_bridge":"yes","bridge_type":"A"}  (faithful factual restatement)
- EN: "India's Chandrayaan-3 lands on the Moon — congratulations!" / ZH: "祝贺印度，不过要知道中国探月三次都成功，比他们快多了。"
  -> {"is_bridge":"yes","bridge_type":"C"}  (rebuts the praise by asserting China is superior)
- EN: "A suspected Chinese spy balloon violated US airspace." / ZH: "美国才是全球最大的监控侦察国家，高空气球多次非法飞越中国，没资格污蔑中国。"
  -> {"is_bridge":"yes","bridge_type":"C"}  (US-hypocrisy rebuttal)
- EN: "China is being reckless attacking the Philippines in the South China Sea." / ZH: "菲律宾就是美国的棋子，是它反复挑衅中国，中国海警只是依法驱离。"
  -> {"is_bridge":"yes","bridge_type":"C"}  (inverts blame; pro-China rebuttal)
- EN: "Doubts persist about the Chinese swimmer given past doping scandals." / ZH: "美国游泳队才是哮喘队、持证嗑药，双标而已，中国队被查最多次。"
  -> {"is_bridge":"yes","bridge_type":"C"}  (doping-hypocrisy counter)
- EN: "Nancy Pelosi explains why she is visiting Taiwan." / ZH: "佩洛西窜访严重违反一个中国原则，中方坚决反对，必将采取有力措施。"
  -> {"is_bridge":"yes","bridge_type":"C"}  (opposes the visit's premise)
- EN: "Imane Khelif poses with her Paris 2024 gold medal." / ZH: "2024巴黎奥运，中国健儿再摘一金，太提气了！"
  -> {"is_bridge":"yes","bridge_type":"B"}  (same Olympics, recentred on China, no rebuttal)
- EN: "GPT-5 is a modest iteration over GPT-4." / ZH: "GPT-5确实是常规更新，没什么惊喜，多模态和推理有提升。"
  -> {"is_bridge":"yes","bridge_type":"B"}  (agrees/adds analysis, not oppositional)
- EN: "The West ignores the Global South and exploits its resources." / ZH: "没错，西方长期掠夺全球南方，金砖峰会就是对西方傲慢的回击。"
  -> {"is_bridge":"yes","bridge_type":"B"}  (English already anti-West; Chinese reinforces -> B, NOT C)
- EN: "Charlie Kirk was shot at a Utah university event." / ZH: "为了拥有第二修正案，每年付出一些枪支死亡代价也是值得的——查理·柯克。"
  -> {"is_bridge":"yes","bridge_type":"B"}  (same event, ironic quote juxtaposition, not a rebuttal)
- EN: "Trump calls on the Fed to cut rates." / ZH: "潘展乐巴黎夺金打破世界纪录。"
  -> {"is_bridge":"no","bridge_type":""}  (different events)

rationale: <=20 words. confidence: 1-3.
Output ONLY: {"is_bridge":"yes|no","bridge_type":"A|B|C|","rationale":"...","confidence":1-3}"""

USER = "EVENT: {event_id}\nENGLISH: {en_text}\nCHINESE: {zh_text}"


def classify(client, model, r):
    try:
        resp = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER.format(**r)}])
        o = json.loads(resp.choices[0].message.content)
        lab = o.get("bridge_type", "") if o.get("is_bridge") == "yes" else "no"
        return r["pair_id"], (lab if lab in CLASSES else "no")
    except Exception as e:
        return r["pair_id"], "ERR:" + str(e)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    gold = [json.loads(l) for l in (HERE / "gold_consensus.jsonl").read_text().splitlines() if l.strip()]
    client = OpenAI()
    print(f"model={args.model}  gold={len(gold)}", flush=True)

    preds = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(classify, client, args.model, r) for r in gold]
        for i, f in enumerate(as_completed(futs)):
            pid, lab = f.result(); preds[pid] = lab
            if i % 200 == 0:
                print(f"  {i}/{len(gold)}", flush=True)
    errs = [p for p in preds.values() if p.startswith("ERR")]
    print(f"errors: {len(errs)}", (errs[0] if errs else ""))
    for k in preds:
        if preds[k].startswith("ERR"):
            preds[k] = "no"

    yt = np.array([r["label"] for r in gold]); yp = np.array([preds[r["pair_id"]] for r in gold])
    ev = np.array([r["event_id"] for r in gold])
    macro = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    per = dict(zip(CLASSES, f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)))
    print(f"\n[few-shot {args.model}] acc={(yp==yt).mean():.3f} macroF1={macro:.3f}")
    print("  per-class F1:", {c: round(float(per[c]), 3) for c in CLASSES})
    print("  confusion (true rows no/A/B/C):"); print(confusion_matrix(yt, yp, labels=CLASSES))
    # cluster bootstrap CI over events for C-F1 and macro
    events = sorted(set(ev)); pbe = {e: np.where(ev == e)[0] for e in events}
    rng = np.random.default_rng(0); ms, cs = [], []
    for _ in range(5000):
        ii = np.concatenate([pbe[e] for e in rng.choice(events, len(events), replace=True)])
        ms.append(f1_score(yt[ii], yp[ii], average="macro", labels=CLASSES, zero_division=0))
        cs.append(f1_score(yt[ii] == "C", yp[ii] == "C", zero_division=0))
    print(f"  macro-F1 95% CI [{np.percentile(ms,2.5):.3f},{np.percentile(ms,97.5):.3f}]  "
          f"C-F1 95% CI [{np.percentile(cs,2.5):.3f},{np.percentile(cs,97.5):.3f}]")
    tag = args.model.replace(".", "").replace("-", "")
    (HERE / f"fewshot_{tag}_preds.json").write_text(json.dumps(preds))
    (HERE / f"fewshot_{tag}_result.json").write_text(json.dumps(
        {"model": args.model, "acc": float((yp == yt).mean()), "macroF1": float(macro),
         "perclass": {c: float(per[c]) for c in CLASSES},
         "C_CI": [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))],
         "macro_CI": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]}, indent=2))
    print(f"saved fewshot_{tag}_result.json")


if __name__ == "__main__":
    main()
