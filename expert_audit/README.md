# Expert Audit — 250-pair blind validation of gold v2

Human-expert audit of the adjudicated gold set (`../gold_consensus.jsonl`, 2,798 pairs).
Author acts as **blind judge**: labels are hidden, row order randomized (seed 42).

## Sample design

- **750 pairs** in three batches of 250
- **Batch 1** (audit_id 0–249, seed 42) + **Batch 2** (250–499, seed 43): stratified validation sample — C=160/B=140/no=120/A=80, 50% adjudicated / 50% agree source, all 33 events. **DONE 2026-07-03**: 71.4% agreement, κ=0.612; consensus 84.8% vs adjudicated 58.0%; 143 corrections applied → gold v2.1. Results in `audit_results_summary.json`.
- **Batch 3** (500–749, seed 44, `batch: 3` in key): targeted cleanup of the noisiest remaining gold — ALL 175 unaudited adjudicated pairs currently labeled C (so every adjudicated C becomes expert-verified) + 75 unaudited Bno-cell pairs (weakest adjudication cell, 55% in batches 1–2). Labels shown in the key are current gold v2.1 labels. **DONE 2026-07-03**: 48.8% confirmed, C→B ×86 → gold v2.2.

## Route A — extended-consensus expansion (batches 4–5, added 2026-07-04)

Goal: validate and C-verify the **extended consensus** tier (4,953 cross-codebook agreement pairs
from the full refined-codebook Claude re-annotation, `extended_consensus.jsonl`) so it can enter an
expanded gold v3 for LLM-evaluation/analysis use (benchmark gold v2.2 stays frozen for trained systems).

- **Batch 4** (audit_id 750–999, seed 45, stratified: C=60/B=80/A=60/no=50, 58 events): *validation
  sample* for the extended tier. Its agreement rate is the tier's quality number — compare against
  84.8% (original consensus tier). Headline-eligible; kept separate from batches 1–2 stats.
- **Batch 5** (audit_id 1000–1444): *exhaustive C cleanup*, biased — the remaining 341
  extended-consensus C pairs + the 104 never-audited original-consensus C pairs. After this batch,
  **every C in the expanded gold is expert-verified**.
- Workflow unchanged: fill `your_label (no/A/B/C)` from row 750 down; scorer reports each batch
  in its proper role. Batch 4 first (it gates whether Route A proceeds), then batch 5.

## Files

| File | Purpose |
|---|---|
| `expert_audit_sheet.csv` | **Work here.** Fill `your_label (no/A/B/C)`; use `notes` for ambiguous pairs |
| `expert_audit_blind.jsonl` | Same content, JSONL, for programmatic annotation |
| `expert_audit_key.jsonl` | Answer key (gold label + source). **Do not open until done** |
| `expert_audit_score.py` | Scorer — run any time, partial audits OK |

## Label guideline (ADJUDICATION.md 3-rule summary)

- **no** — same broad topic only, different sub-event, or keyword coincidence
- **A** — faithful restatement: same facts AND framing, no added opinion/angle (minor detail/length difference is still A)
- **B** — reframing: same sub-event but adds a non-oppositional angle/analysis/China-recentring, without rebutting the EN framing
- **C** — counter-narrative: rebuts / mocks / inverts the EN framing or asserts the opposite stance. Mockery, sarcasm, and claiming-the-EN-assertion-is-false all count as C. If EN is already anti-US/critical and ZH merely reinforces it → B, not C

## Workflow

1. Annotate in 3–4 sittings (~60–80 pairs each) to avoid fatigue drift
2. `python3 expert_audit_score.py` → overall agreement, Cohen's κ, per-class, per-source, confusion
3. The per-source split (agree vs adjudicated) is the key diagnostic for the paper
