# FUSE-C+ — the learned arbiter (pre-registered)

Everything needed to reproduce the FUSE-C+ rows of the paper's benchmark table (Table II),
their paired tests, and the descriptive CIs — from frozen inputs, with no LLM access required.
Re-collecting the signals from scratch (optional) needs an OpenAI API key.

## What FUSE-C+ is

A class-balanced logistic arbiter over calibrated confidence signals from both the frozen
FUSE-C student and the `gpt-5.4-nano` assistant (single-call prediction, k=8 self-consistency
vote distribution, no-reasoning class-token log-probability distribution, similarity summary,
derived margins/flags → ψ(x) ∈ ℝ³¹). Trained and evaluated by event-grouped 5-fold
out-of-fold prediction on the gold. Protocol, feature set, decision criterion, and comparison
targets were locked in `PREREGISTRATION_FUSECPLUS_V2.md` **before** the signals were inspected.

**Pre-registered verdict: DO NOT SHIP the "outperforms the LLM" claim** — full-gold ΔC-F1
+0.043 vs the single-call LLM, CI [−0.001, +0.090], p = 0.055 (`results/fusecplus_v2_result.json`).
The paper reports the result at the edge of significance, with the significant secondary
findings (core854 macro +0.033, p = 0.025; B-precision variant +0.038, p = 0.010) labeled as such.
Do not quote these artifacts as an LLM win.

## Frozen inputs in this directory

| file | contents |
|---|---|
| `llm_probs.jsonl` | per-pair class-token log-probability distributions (no-reasoning pass) |
| `llm_votes.jsonl` | per-pair k=8 self-consistency vote counts (standard few-shot elicitation) |
| `student_probs.json`, `_seed1`, `_seed2` | calibrated student posteriors, three re-trained seeds |

These are pair_id-keyed probabilities/labels only — no post text (Tier-1 compatible).

## Additional data files required (place alongside, from the Tier-1/2 data release)

`gold_consensus.jsonl`, `fewshot_gpt54nano_preds.json`, `llm_labels_v2.jsonl`,
`pair_emb_v2.npz`, and `expert_audit/expert_audit_key.jsonl` +
`expert_audit/expert_audit_sheet.csv` (true expert labels).

## Reproduce the paper's numbers (no API key)

```bash
python fusecplus_validate_v2.py     # Table II FUSE-C+/self-consistency rows + pre-registered tests
python arbiter_standalone_cis.py    # descriptive CIs; verifies exact reproduction of the frozen
                                    # out-of-fold predictions before resampling
```

`fusecplus_validate_v2.py` is the pre-registered harness: run it once, take the verdict as
printed. `arbiter_standalone_cis.py` asserts bit-identical point estimates against
`results/fusecplus_v2_result.json` before computing any interval, and also emits the core854/
core500 descriptive cells.

## Re-collect the signals from scratch (optional; needs `OPENAI_API_KEY`)

```bash
# Stage-0/1: rebuild features and re-train the student ensemble (CPU, ~2h per seed)
CPHOT_RAW_DIR=/path/to/cphot/raw python rebuild_features.py
FUSEC_SEED=0 python train_fusec.py && FUSEC_SEED=1 python train_fusec.py && FUSEC_SEED=2 python train_fusec.py

# assistant signals (checkpointed / resumable; failures are retried on re-run, never silently kept)
python run_llm_probs.py --model gpt-5.4-nano --gold gold_consensus.jsonl   # ~2.8k calls
python run_llm_selfconsist.py --k 8 --workers 16                           # ~22k calls
```

Model-API caveat (documented in the paper): `gpt-5.4-nano` exposes log-probabilities only under
`reasoning_effort="none"`, where its hard labels degrade — the log-probability distribution is
therefore a confidence feature only, never a prediction. The re-trained ensemble's per-seed
macro-F1 (0.31–0.33) runs below the deployed student (0.359) because Stage-0 features are
re-derived from raw artifacts; this makes the arbiter comparison conservative (paper §4.4).
