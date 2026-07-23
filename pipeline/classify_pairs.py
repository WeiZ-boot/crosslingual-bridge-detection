"""Phase 3 — LLM as interpretable cross-lingual bridge classifier (the core contribution).

Given an (English anchor, Chinese post) pair, the LLM decides is_bridge + type (A/B/C) AND
emits a short rationale (interpretability). Same labels schema as the human annotation set, so
LLM output can be (a) evaluated against human gold and (b) used as scalable SILVER labels to
fine-tune a smaller model (Phase 4).

Requires ANTHROPIC_API_KEY + `pip install anthropic`. Default model is a current Claude.
Run: python classify_pairs.py --in annotation_set.jsonl --out llm_labels.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

PROVIDER = os.environ.get("BRIDGE_LLM_PROVIDER", "openai")
MODEL = os.environ.get("BRIDGE_LLM_MODEL", "gpt-5.4-nano")

SYSTEM = """You are an expert annotator of CROSS-LINGUAL information bridges between English
social media (X/Reddit) and Chinese social media (Weibo/Zhihu/Bilibili/Douyin) about the same
hot-topic event. For each English+Chinese post pair you output a JSON object only.

is_bridge: "yes" if the Chinese post carries the SAME specific event/sub-event/claim across the
language boundary (by translating, reframing, or responding to it); "no" if it is merely the same
broad topic, a different sub-event, or coincidental keyword overlap.

bridge_type (only if is_bridge == "yes"):
  A = Direct translation: essentially the SAME facts AND framing rendered in Chinese — a translation
      or faithful repost with no added opinion, angle, or rebuttal. (Translations are common; use A
      whenever the Chinese post is a faithful restatement, even if wording differs.)
  B = Reframing: same event, but a different angle/emphasis/selection/affect is ADDED beyond
      translation (recentres on China, adds opinion/commentary, picks a different sub-detail)
      WITHOUT directly rebutting the English framing.
  C = Counter-narrative: the Chinese post explicitly pushes back on / rebuts / mocks / is skeptical
      of the English/Western framing — "the real reason is...", "absurd/suspicious", hypocrisy
      accusations, sarcasm toward the Western account, or asserting the opposite claim.
Distinguish carefully: A = faithful restatement; B = adds a non-oppositional angle/opinion;
C = argues AGAINST the Western framing. Judge what the Chinese text actually does.

FEW-SHOT:
- EN "Donald Trump has declared victory in presidential election" / ZH "特朗普发表讲话宣布胜选"
  -> {"is_bridge":"yes","bridge_type":"A",...}  (same fact, same framing)
- EN "Imane Khelif poses with her Gold medal at 2024 Paris Olympics" / ZH "2024巴黎奥运会,中国健儿摘金"
  -> {"is_bridge":"yes","bridge_type":"B",...}  (same event, recentred on China)
- EN "Top CCP Official Responds to Boycott Xinjiang Cotton Campaign" / ZH "西方抵制中国棉花的真正原因!
  他们抵制的不是棉花,是对棉花定价权的争夺!" -> {"is_bridge":"yes","bridge_type":"C",...}  (rebuts the framing)
- EN "Trump calls on Fed to cut rates ahead of tariffs" / ZH "潘展乐巴黎夺金" -> {"is_bridge":"no",...}

rationale: one concise sentence (<=25 words) justifying the decision.
confidence: integer 1 (unsure) .. 3 (certain).

Output ONLY: {"is_bridge":"yes|no","bridge_type":"A|B|C|","rationale":"...","confidence":1-3}"""

USER_TMPL = """EVENT: {event_id}
ENGLISH ({en_platform}): {en_text}
CHINESE ({zh_platform}): {zh_text}"""


def _raw_call(client, pair: dict) -> str:
    user = USER_TMPL.format(**pair)
    if PROVIDER == "anthropic":
        msg = client.messages.create(
            model=MODEL, max_tokens=200, system=SYSTEM,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    # openai (chat completions; JSON mode)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"})
    return resp.choices[0].message.content


def make_client():
    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    from openai import OpenAI
    return OpenAI()


def classify(client, pair: dict) -> dict:
    txt = (_raw_call(client, pair) or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        out = json.loads(txt)
    except json.JSONDecodeError:
        out = {"is_bridge": "", "bridge_type": "", "rationale": "PARSE_FAIL:" + txt[:80],
               "confidence": 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="annotation_set.jsonl")
    ap.add_argument("--out", default="llm_labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--events", nargs="*")
    args = ap.parse_args()

    client = make_client()
    print(f"provider={PROVIDER} model={MODEL}")

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.events:
        rows = [r for r in rows if r["event_id"] in args.events]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["pair_id"] for l in out_path.read_text().splitlines() if l.strip()}
    fh = out_path.open("a", encoding="utf-8")
    for i, r in enumerate(rows):
        if r["pair_id"] in done:
            continue
        lab = classify(client, r)
        rec = {**{k: r[k] for k in ("pair_id", "event_id", "sim", "band")},
               "en_text": r["en_text"], "zh_text": r["zh_text"], **lab}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}")
        time.sleep(0.3)
    fh.close()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
