# Cross-Lingual Bridge Annotation Guidelines (v1 draft, 2026-06-11)

## What you are labeling
Each item is an **English post** (from X/Reddit, the *anchor*) paired with a **Chinese post**
(from Weibo/Zhihu/Bilibili/Douyin) about the **same hot-topic event**. Decide:

1. **Is this a cross-lingual bridge?** — Do the two posts refer to the *same specific event /
   sub-event / claim*, such that the Chinese post plausibly carries that content across the
   language boundary (by translating, reframing, or responding to it)? `yes` / `no`.
2. **If yes, which type?** — A, B, or C (definitions below).
3. **Confidence** 1 (unsure) – 3 (certain).

> The similarity score shown is only why the pair was *suggested*. It is NOT the answer.
> A high-similarity pair can be `no` (same topic, unrelated sub-event); a low-similarity pair
> can be a strong Type C. Judge the **content relationship**, not the number.

## Decision tree
```
Same specific event/claim referenced by both?
├─ no  → is_bridge = no   (leave type blank)
└─ yes → is_bridge = yes
         ├─ ZH ≈ faithful rendering of the EN content (same facts, same framing) → A
         ├─ ZH same event but different angle/emphasis/selection/affect            → B
         └─ ZH explicitly pushes back on / rebuts the EN framing or Western account → C
```

## The three bridge types

### Type A — Direct translation bridge
Same content, different language. The Chinese post conveys essentially the same information and
framing as the English one (a translation, near-translation, or faithful 搬运/repost).
**Real example** (sim 0.86, us_election_2024):
- EN: "Donald Trump has declared victory in presidential election as he addresses…"
- ZH: "特朗普发表讲话宣布胜选" (Trump gives speech declaring election victory)

### Type B — Reframing bridge
Same event, **different angle, emphasis, or selection**. The core event is shared but the Chinese
post highlights a different actor/aspect, or carries a different affect, without directly rebutting
the English framing.
**Real example** (sim 0.63, paris_olympics_ceremony):
- EN: "Imane Khelif poses with her Gold medal after the 2024 Paris Olympics"
- ZH: "2024巴黎奥运会，中国🇨🇳健儿摘金" (Chinese athletes take gold at Paris 2024)
→ same event (Paris medals) reframed around China's athletes.

### Type C — Counter-narrative bridge
The Chinese post **explicitly responds to or contradicts** the English platform's framing — rebuttal,
skepticism, "the real reason is…", accusation of hypocrisy, etc.
**Real examples**:
- (xinjiang_cotton) EN: "Top CCP Official Responds to Boycott Xinjiang Cotton Campaign" ↔
  ZH: "西方国家抵制中国棉花的真正原因！…他们抵制的不是棉花,是对棉花定价权的争夺！"
  (The *real* reason the West boycotts Xinjiang cotton… it's about pricing power, not cotton.)
- (trump_shooting) EN: "Secret Service ramped up security after intel of Iranian plot…" ↔
  ZH: "离了大谱,而且疑点重重——美国特勤局有超过8300人,每年预算超30亿美元,却让…"
  (Absurd and full of holes — 8,300 agents, $3B budget, yet they let…)

## Not a bridge (`is_bridge = no`)
- **Same topic, different sub-event.** EN about Trump's Fed comments; ZH about a *different* tariff
  sub-story with no shared specific claim → no (even if similarity is mid/high).
- **Generic commentary** that could attach to anything in the event, with no traceable link to the
  specific English post's content.
- **Coincidental keyword overlap** (e.g. both mention "Iran" but about unrelated incidents).

## Edge cases & conventions
- **Bilingual anchors:** some English-side posts are from translation/搬运 accounts and already
  contain Chinese text. Still judge content relation; these are usually Type A.
- **A vs B:** if any *new framing, omission, or affective spin* is added beyond translation → B.
  Pure restatement → A.
- **B vs C:** B = different angle; C = the angle is *against* the Western framing specifically.
  If the ZH post argues with / debunks / accuses, choose C.
- **Direction:** primary direction is EN→ZH (English content migrating to Chinese platforms).
  Annotate the relationship regardless of which post is chronologically first.
- **Many-to-one:** an English post may bridge to several Chinese posts (and vice versa). Label
  each pair independently.

## Process
- Pilot: 2 annotators each label the same ~150 pairs (2 events) → compute Cohen's κ on
  (is_bridge) and (A/B/C | bridge). Target κ ≥ 0.6 before scaling; revise this doc on disagreements.
- Then split the remaining ~2,000 pairs; keep ~10% double-annotated for ongoing agreement.
- Data: `data-collection/annotation_set.csv` (or .jsonl). Fill `is_bridge`, `bridge_type`,
  `confidence`, `note`.
