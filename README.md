# Earlier modular FUSE-C implementation

This directory organizes earlier code into collection, pipeline, evaluation and analysis folders. It **does not contain the complete current nine-assistant human-reference comparison**.

Use [current experiment index](../release/EXPERIMENTS.md) and `../data-collection/` for the principal manuscript experiment. Earlier result claims in historical documents are not current reference statistics.

| Directory | Role |
|---|---|
| `collection/` | Historical collection tools |
| `pipeline/` | Candidate preparation, annotation and features |
| `eval/` | Earlier baselines, transfer evaluation and verifier prototype |
| `analysis/` | Temporal/direction analysis and plotting |
| `tests/` | Earlier modular-code tests |
| `expert_audit/` | Earlier audit scorer and private artifacts |
| `docs/`, `results/`, `figures/` | Historical records, not automatically released |

The public code snapshot has no dataset. Inspect each script's working-directory and relative-input assumptions. The old dependency list is not a lockfile; current release dependency groups and smoke tests are documented under `../release/`.

No license has been selected by this preparation step. Corpus release and platform-content permissions are separate decisions.
