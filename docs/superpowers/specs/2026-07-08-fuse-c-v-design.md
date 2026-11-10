# FUSE-C-V Design

## Objective

Build a same-budget enhancement of FUSE-C that can outperform the standalone few-shot LLM without using more than one LLM call per pair. The new framework, FUSE-C-V, treats the few-shot LLM as one input view and uses local FUSE-C evidence to verify or correct it.

## Core Claim

FUSE-C-V should test whether local structural evidence can correct systematic LLM errors in cross-lingual bridge detection. The target comparison is not zero-call FUSE-C versus the few-shot LLM, but:

```text
Few-shot LLM, one call per pair
vs.
FUSE-C-V, one same LLM call per pair plus local verifier features
```

The primary success criterion is higher C-F1 than the few-shot LLM on the full gold, with paired cluster-bootstrap testing. A stronger result would also improve macro-F1 and replicate on expert-audited cores.

## Architecture

FUSE-C-V has three inputs for each English-Chinese pair:

1. Local FUSE-C evidence:
   - student class probabilities or predicted label
   - student confidence margin
   - stance, temporal, and similarity feature blocks when available
2. One few-shot LLM prediction:
   - predicted no/A/B/C label
   - confidence if available
   - optional structured explanation fields in a later extension
3. Agreement features:
   - whether FUSE-C and the LLM agree
   - whether the LLM predicts C while the student does not
   - whether the student predicts no while the LLM predicts a bridge
   - entropy or margin features for both systems when available

The verifier is a small supervised model, initially logistic regression or a shallow MLP, trained only on event-disjoint training data. It outputs the final no/A/B/C label.

## Stage 1: Artifact-Based Meta-Fusion Verifier

The first implementation adds an evaluation script that builds a verifier dataset from existing artifacts. It must not assume private files are present. It should check for required files and fail with a clear message when they are missing.

Expected inputs, when available:

- `gold_consensus.jsonl`
- event ids or event-cluster metadata
- few-shot LLM predictions
- FUSE-C or cross-encoder predictions/probabilities
- local feature arrays such as pair embeddings or temporal features

The script should support degraded modes:

- If probabilities are available, use probabilities and margins.
- If only labels are available, use one-hot labels and agreement flags.
- If local view features are absent, report that full FUSE-C-V cannot be trained rather than silently substituting unrelated features.

Outputs:

- `results/fuse_c_v_result.json`
- `results/fuse_c_v_preds.json`
- printed metrics: accuracy, macro-F1, per-class F1, C-F1
- paired cluster-bootstrap comparison against the few-shot LLM when event clusters are available

## Stage 2: Explanation-Aware Verifier

If Stage 1 does not beat the few-shot LLM, add a one-call richer prompt. The LLM call should return a label plus structured fields:

- same specific claim: yes/no/uncertain
- stance relation: supports/opposes/reframes/unrelated
- B-vs-C rationale
- confidence

This still respects the same one-call budget. The verifier can then learn when to trust the LLM label and when to override it using local structural evidence.

## Evaluation Plan

Primary comparison:

```text
Few-shot LLM vs. FUSE-C-V
```

Primary metric:

```text
C-F1
```

Secondary metrics:

```text
macro-F1, per-class F1, confusion matrix
```

Statistical test:

Use paired cluster bootstrap over event clusters when event ids are available. Report raw and, if part of a family of comparisons, Holm-corrected p-values.

Ablations:

- LLM-only
- local FUSE-C-only
- LLM + local labels only
- LLM + local probabilities/margins
- LLM + full local view features

The minimum publishable improvement is FUSE-C-V beating the few-shot LLM in C-F1 on full gold. The stronger claim requires the improvement to replicate on expert-audited cores.

## Constraints

- No additional LLM calls beyond the standalone few-shot LLM baseline.
- No gold-event leakage into verifier training, calibration, feature standardization, or model selection.
- Do not train on expert-core labels unless running a separate diagnostic explicitly labeled as expert-core supervised.
- Keep `gpt-5.4-nano` terminology clear: standalone few-shot LLM in baseline rows, assistant only inside cascade rows.
- If required artifacts are missing from the repo, the script must explain which files are needed and how to generate or place them.

## Initial Implementation Scope

Implement Stage 1 first:

1. Add a reusable metric/bootstrap utility if none exists.
2. Add `eval/fuse_c_v.py`.
3. Add artifact validation and clear error messages.
4. Train a small event-disjoint verifier when artifacts are present.
5. Save predictions and result JSON.
6. Document the command in `README.md`.

Stage 2 should be added only after Stage 1 is measurable.

## Risks

The main scientific risk is that same-budget FUSE-C-V may not beat the few-shot LLM. If this happens, the result is still useful as an error analysis, but the paper should not claim LLM-superior performance. The next step would be the explanation-aware verifier, which keeps the same call budget but gives the verifier richer evidence.

The main engineering risk is missing raw artifacts. The current repository contains result summaries but not the full evaluation files needed to train the verifier. The implementation must make this limitation visible.
