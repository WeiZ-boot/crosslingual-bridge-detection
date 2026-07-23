"""Few-shot nano on gold v3 extended pairs; checkpointed + resumable. Merges cached v2 preds, scores v3."""
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from fewshot_eval import classify, CLASSES
from openai import OpenAI
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
CKPT = HERE / 'fewshot_v3_ckpt.jsonl'
v3 = [json.loads(l) for l in open(HERE/'gold_v3_extended.jsonl') if l.strip()]
preds = dict(json.load(open(HERE/'fewshot_gpt54nano_preds.json')))
if CKPT.exists():
    for l in open(CKPT):
        if l.strip():
            r = json.loads(l); preds[r['pair_id']] = r['label']
todo = [r for r in v3 if r['pair_id'] not in preds]
print(f'v3={len(v3)}, have={len(v3)-len(todo)}, to classify={len(todo)}', flush=True)

if todo:
    client = OpenAI(timeout=60.0, max_retries=1)
    rows = [{'pair_id': r['pair_id'], 'event_id': r['event_id'], 'en_text': r['en_text'], 'zh_text': r['zh_text']} for r in todo]
    lock = Lock(); done = 0
    with open(CKPT, 'a') as ck, ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(classify, client, 'gpt-5.4-nano', r) for r in rows]
        for f in as_completed(futs):
            pid, lab = f.result()
            lab = lab if not lab.startswith('ERR') else 'no'
            with lock:
                preds[pid] = lab
                ck.write(json.dumps({'pair_id': pid, 'label': lab}) + '\n'); ck.flush()
                done += 1
                if done % 250 == 0: print(f'  {done}/{len(rows)}', flush=True)

json.dump(preds, open(HERE/'fewshot_gpt54nano_preds_v3.json','w'))
yt = np.array([r['label'] for r in v3]); yp = np.array([preds[r['pair_id']] for r in v3])
ev = np.array([r['event_id'] for r in v3]); events = sorted(set(ev))
pbe = {e: np.where(ev==e)[0] for e in events}
per = dict(zip(CLASSES, f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)))
rng = np.random.default_rng(0); cs, ms = [], []
for _ in range(5000):
    ii = np.concatenate([pbe[e] for e in rng.choice(events, len(events), replace=True)])
    ms.append(f1_score(yt[ii], yp[ii], average='macro', labels=CLASSES, zero_division=0))
    cs.append(f1_score(yt[ii]=='C', yp[ii]=='C', zero_division=0))
print(f'[few-shot nano vs GOLD v3 n={len(v3)}] acc={(yp==yt).mean():.3f} '
      f'macro={f1_score(yt,yp,average="macro",labels=CLASSES,zero_division=0):.3f} '
      f'per-class ' + ' '.join(f'{c}={per[c]:.3f}' for c in CLASSES))
print(f'  macro CI [{np.percentile(ms,2.5):.3f},{np.percentile(ms,97.5):.3f}]  '
      f'C-F1 CI [{np.percentile(cs,2.5):.3f},{np.percentile(cs,97.5):.3f}]')
