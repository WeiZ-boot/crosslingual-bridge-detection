# (b) no/B adjudication — diagnosing & raising the label ceiling

## Disagreement structure (Claude vs GPT-5.5, all 2,798 overlapping pairs)
63% agree (1,771/2,798). The 1,027 disagreements route through B (the majority class) on all three
boundaries, ~equally: **B↔A 353 · B↔C 348 · no↔B 277**. So it's not "just no/B" — Claude uses B as a
catch-all where GPT-5.5 commits to A/C/no.

## Adjudication study (24 disagreements, 8 per cell, re-judged with sharpened rules)
| boundary | pattern | verdict |
|---|---|---|
| **B (Claude) vs A (GPT)** | Claude calls faithful restatements "B" because the ZH is longer / adds a minor detail (e.g., same Mar-a-Lago party, same carrier specs, same Epstein-list explainer). | **~4/8 definitional → A** (Claude too loose on B); ~4/8 genuine B (real added analysis/opinion, e.g. Gu citizenship *analysis*). |
| **B (Claude) vs C (GPT)** | Claude too STRICT on C — labeled mockery/sarcasm/arguing-the-claim-is-wrong as "B" (e.g. mocks U-2 as impotent, rebuts groyper theory, "BRICS coin is useless"). | **~6/8 definitional → C** (Claude under-calls C); ~2/8 genuine B (analytical disagreement ≠ stance rebuttal). |
| **no (GPT) vs B (Claude)** | "Same figure/topic, different specific sub-event": game-build vs real carrier, different DJI products, separate Erika speeches → GPT `no`; but same-aftermath/same-topic → Claude B. | **~3/8 definitional → no**; ~5/8 genuine ambiguity (the fuzziest boundary). |

**Overall: ~52% of disagreements are DEFINITIONAL** (a clear label exists once the rule is sharpened),
~48% genuine ambiguity. So the moderate κ (≈0.45) largely reflects **guideline under-specification, not
irreducible task subjectivity.**

## Implication for the ceiling
If re-annotated with the refined guideline below, agreement ≈ 63% + 0.52×37% ≈ **~82%** (κ would rise
from ~0.45 toward ~0.7). And because the resolved cases are mostly **C** (from B→C) and **A** (from
B→A), the refined-guideline gold gains exactly the classes we care about — a higher, and C-richer,
ceiling for every model.

## Refined guideline (the fix — resolves the systematic disagreements)
1. **A vs B:** length or a minor extra fact does NOT make it B. Label **A** whenever the ZH conveys the
   same facts + same stance (translation, official statement/obituary, factual explainer/spec/list).
   Reserve **B** for a *substantive* added angle: opinion, analysis, China-recentring, or selective
   emphasis that changes the takeaway.
2. **B vs C:** **C** = the ZH opposes the EN's stance in ANY of these forms — rebuttal, **mockery/
   sarcasm toward the Western side**, hypocrisy accusation, or **asserting the EN's claim is false/
   infeasible**. (Previously under-applied.) Pure analytical disagreement with no oppositional stance
   stays **B**.
3. **no vs B:** a bridge requires the **same specific sub-event/claim**, not merely the same person or
   broad topic. Different product, different speech, fan-art vs real event, or a tangential
   sub-detail → **no**. (Genuinely fuzzy ~half the time; mark low-confidence.)

## Status / recommendation
This is a **sample-validated diagnosis + guideline**, not yet a full re-label. Realizing the ~82%
ceiling requires a re-annotation pass over the 1,027 disagreements with these rules (a marathon), then
re-freezing the gold and re-running `fewshot_eval.py` / `phase4b_bench.py` — model C-F1 should rise
because (a) noise removed and (b) the gold gains C. Highest-value paper takeaway either way: **the
ceiling is guideline-limited, not task-limited** — with a sharpened codebook, cross-lingual
counter-narrative is more reliably annotatable than κ≈0.45 suggests.
