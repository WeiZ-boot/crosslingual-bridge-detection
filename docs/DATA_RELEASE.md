# Data Release Design — ToS-Compliant Distribution Plan

*2026-07-03. Closes PAPER_DRAFT.md TODO #7. Governs what ships publicly, what ships under
agreement, and what is re-derivable from released tooling.*

## Design principle

Release everything **we** created (labels, adjudications, audit, metadata, tooling) publicly;
release platform **content** only in the form each platform's terms permit. Where rehydration is
unreliable (Chinese platforms; deleted posts), close the gap with a gated full-text tier rather
than by violating ToS.

## Per-platform ruling

| platform | posts used | public release form | rationale |
|---|---|---|---|
| Reddit | EN anchors | **post/comment IDs** + rehydration script (official API / Arctic Shift) | API terms restrict bulk text redistribution; ID-sharing is the established academic norm |
| X | EN anchors | **post IDs only** | Developer Policy: only Post/User IDs may be redistributed; unlimited volume explicitly permitted for academic non-commercial research (verified 2026-07) |
| YouTube | EN comments | **video ID + comment IDs** | API ToS prohibit storing/redistributing comment text beyond limited windows |
| Hacker News | EN posts | **full text** | Public Firebase API, permissive norms (HN corpus is already redistributed at scale, e.g., BigQuery public datasets) |
| Bluesky | EN posts | **AT-URIs (DID + rkey)** + rehydration via public API | Firehose is public-by-design and full-text datasets exist, but community norms are contested; URI-level release is the conservative choice |
| Weibo | ZH posts | **post IDs + SHA-256 text hash + excerpt policy (below)** | No redistribution provision; rehydration unreliable (deletion/censorship) |
| Bilibili | ZH posts | same as Weibo | same |
| Zhihu | ZH posts | same as Weibo | same |
| Douyin | ZH posts | same as Weibo | same |

**Chinese-platform excerpt policy.** For the 2,798 gold pairs only, the public release includes a
≤50-character ZH excerpt (the highest-attention span used in annotation) under fair-quotation
norms, plus the SHA-256 hash of the full normalized text so gated-tier recipients can verify
integrity. Silver-set (13,992) ZH text is IDs+hash only.

## Three release tiers

**Tier 1 — public (repository + HF dataset).**
- All pair-level artifacts: `pair_id`, `event_id`, both platform IDs, timestamps, direction +
  lead-time tags, BGE-M3 similarity, silver label, dual annotations, adjudicated label, gold v2.2
  label + provenance (`agree`/`adjudicated`/`expert`-tiers), audit batch membership.
- The full refined-codebook Claude re-annotation (13,992 pairs) as a second annotation layer,
  plus the derived *extended consensus* flag (4,953 cross-codebook agreements) and the two golds:
  `gold_consensus.jsonl` (benchmark v2.3, 2,798, event-disjoint, all C expert-verified) and
  `gold_v3_extended.jsonl` (6,451, 70 events, all C expert-verified; LLM-eval/analysis — not
  event-disjoint). The tier's 1,300 unaudited A pairs are flagged training-only (audit: 0.53
  reliability on that cell).
- Full text where permitted (HN), IDs/URIs elsewhere, ZH gold excerpts per policy above.
- Codebook (`ANNOTATION_GUIDELINES.md`, `ADJUDICATION.md`), expert-audit protocol + summary
  statistics, all evaluation scripts, figures, and frozen result JSONs.
- Rehydration script: given the ID manifest, reconstructs the corpus from platform APIs where
  available; reports per-platform recovery rate.

**Tier 2 — gated full text (research DUA).**
- Full EN+ZH text for the 2,798-pair gold and, on justified request, the 13,992-pair candidate
  set. Distribution: signed data-use agreement (non-commercial research, no redistribution, no
  user-level profiling, deletion on request), via institutional exchange. This is the standard
  mitigation where rehydration decay would otherwise destroy benchmark reproducibility — which is
  acute here: censorship-driven deletion on Chinese platforms is not random (it correlates with
  exactly the contested content the C class captures), so ID-only release would silently bias
  future rehydrated evaluations toward A/B pairs.

**Tier 3 — re-collection tooling.**
- Collection configurations (event windows, query terms per event, MediaCrawler configs, Arctic
  Shift queries) sufficient to re-collect from scratch. This documents provenance and makes the
  pipeline auditable even if both content tiers become unavailable.

## What is never released
- Usernames/handles, user IDs on ZH platforms, profile metadata, follower data (were stripped at
  collection; pair artifacts reference posts, not accounts).
- Embeddings of full text for ID-only platforms are released as **similarity scores only** (not
  raw vectors) to avoid approximate text reconstruction.

## Paper text implications
- §9 Ethics: replace the current one-line release sentence with the tiered design (done in draft v2).
- §8 Limitations: note rehydration-decay bias (censorship deletes C-rich content first) as the
  motivation for Tier 2 — this is a *finding-adjacent* point worth one sentence.

## Open items before camera-ready
- [ ] Draft the DUA text (institutional template + the four conditions above).
- [ ] Verify Arctic Shift's current redistribution stance for Reddit IDs+hashes.
- [ ] Decide hosting (HF gated dataset vs institutional server) for Tier 2.
- [ ] Per-platform recovery-rate dry run of the rehydration script (report in the released README).
