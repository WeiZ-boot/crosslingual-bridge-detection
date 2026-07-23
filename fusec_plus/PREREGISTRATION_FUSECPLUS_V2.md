# Pre-registration — FUSE-C+ V2 arbiter validation (2026-07-09)

**Status:** LOCKED before any V2 signal was inspected in combination. The V1 run (2026-07-09,
`fusecplus_validation_result.json`) printed DO NOT SHIP (ΔC-F1 +0.033, p=0.14). This document
pre-registers exactly two confirmatory tests for the V2 arbiter. They will be run ONCE each,
by `fusecplus_validate_v2.py`, when all inputs exist. No other test on these signals counts.

## New signals (collection in progress at lock time; contents not yet seen)

1. **Self-consistency votes** (`llm_votes.jsonl`): k=8 samples of the ORIGINAL strong few-shot
   elicitation (`fewshot_eval.classify`, default temperature) per gold pair. Vote fractions =
   calibrated confidence for the strong arm. Only a 3-pair variance smoke test was observed
   before lock (votes do vary; no aggregate statistics were computed).
2. **Student seed ensemble** (`student_probs.json` + `_seed1` + `_seed2`): three independent
   train_fusec.py runs (FUSEC_SEED=0/1/2); arbiter uses the MEAN calibrated distribution.

## Frozen protocol (identical to V1 unless stated)

- Gold: `gold_consensus.jsonl` (v2.3, n=2798, 33 events). Comparison target: the original
  strong few-shot single-call preds `fewshot_gpt54nano_preds.json` — the paper's LLM baseline.
- Arbiter: LR (C=0.5, balanced) and MLP(32), 5-fold GroupKFold OOF by event; the shipped
  arbiter is whichever has higher OOF C-F1 (as in V1).
- Features: student-ensemble 4-dist; strong-pred one-hot; no-reasoning logprob 4-dist
  (`llm_probs.jsonl`); vote 4-fractions; sim(5); margins (student, logprob, vote top1−top2);
  vote entropy; disagree flag; s=B∧m=C flag; logprob-C; vote-C; interactions
  (logprob-C×s-margin, vote-C×s-margin).
- Paired cluster bootstrap over events, 5,000 resamples, seed 20260101, two-sided p.

## Test A — headline (full gold)

Arbiter vs strong single-call LLM. **SHIP the full-gold "outperforms" claim iff**
ΔC-F1 > 0 with p < 0.05 and 95% CI excluding 0, AND no significant B-precision regression
(i.e. NOT (ΔB-prec < 0 with CI entirely < 0)).

**Honesty secondary (reported, non-gating, claim-qualifying):** (i) arbiter vs self-consistency
majority-vote LLM; (ii) SC-majority vs single-call. If the arbiter beats the single-call LLM but
not SC-majority, any manuscript claim must name the single-call system as the baseline and
report the SC-majority comparison alongside.

## Test B — scoped (expert core854)

Same OOF arbiter predictions restricted to the 854 audited-∩-gold pairs, scored against TRUE
expert labels (`expert_audit_sheet.csv` `your_label` column joined via `expert_audit_key.jsonl`
audit_id→pair_id; see CORE_LABEL_CORRECTION.md). Paired cluster bootstrap over the events
present in the core. Same bar as Test A. Passing B but not A supports only the scoped claim
"outperforms on the expert-verified core854".

## Decision rule

- A passes → `SHIP (full-gold claim)`; report B as supporting evidence either way.
- A fails, B passes → `SHIP (core854-scoped claim only)`.
- Neither passes → `DO NOT SHIP` — the manuscript keeps the near-parity framing; the remaining
  legitimate route is gold-set expansion (~60+ events), not further feature engineering.

No re-runs with modified features, folds, seeds, or bootstrap parameters. If a data-integrity
bug is found, fix it, note it here, and rerun; a bug fix is not a protocol change.
