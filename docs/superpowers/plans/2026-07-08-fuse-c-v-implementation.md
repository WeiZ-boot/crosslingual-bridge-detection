# FUSE-C-V Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a same-budget verifier script that combines one few-shot LLM prediction per pair with local FUSE-C evidence and evaluates whether the verifier beats the standalone few-shot LLM.

**Architecture:** Implement `eval/fuse_c_v.py` as a self-contained evaluation script with importable helpers. It validates artifacts, constructs verifier features from LLM labels and local predictions/probabilities, trains an event-disjoint small classifier, evaluates against gold labels, and writes JSON outputs. Tests use built-in `unittest` and synthetic data so they run without private artifacts.

**Tech Stack:** Python 3, standard library `unittest`, `numpy`, `scikit-learn`.

---

## File Structure

- Create: `tests/test_fuse_c_v.py`
  - Unit tests for artifact validation, feature construction, metrics, cluster bootstrap, and event-disjoint split behavior.
- Create: `eval/fuse_c_v.py`
  - CLI script and importable helpers for FUSE-C-V Stage 1.
- Modify: `README.md`
  - Add the verifier command and required artifact list.

## Task 1: Missing-Artifact Validation

**Files:**
- Create: `tests/test_fuse_c_v.py`
- Create: `eval/fuse_c_v.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
import unittest
from pathlib import Path

from eval.fuse_c_v import ArtifactError, require_files


class FuseCVArtifactTests(unittest.TestCase):
    def test_require_files_reports_all_missing_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "present.json").write_text("{}")

            with self.assertRaises(ArtifactError) as ctx:
                require_files(root, ["present.json", "missing.json", "nested/also_missing.json"])

            msg = str(ctx.exception)
            self.assertIn("missing.json", msg)
            self.assertIn("nested/also_missing.json", msg)
            self.assertNotIn("present.json", msg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: FAIL or ERROR because `eval.fuse_c_v` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `eval/fuse_c_v.py`:

```python
from __future__ import annotations

from pathlib import Path


class ArtifactError(RuntimeError):
    """Raised when required verifier artifacts are absent."""


def require_files(root: Path, rel_paths: list[str]) -> None:
    missing = [p for p in rel_paths if not (root / p).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise ArtifactError(f"Missing required FUSE-C-V artifacts:\n  - {joined}")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Task 2: Metric Helpers

**Files:**
- Modify: `tests/test_fuse_c_v.py`
- Modify: `eval/fuse_c_v.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fuse_c_v.py`:

```python
from eval.fuse_c_v import classification_report


class FuseCVMetricsTests(unittest.TestCase):
    def test_classification_report_returns_macro_and_c_f1(self):
        y_true = ["no", "A", "B", "C", "C"]
        y_pred = ["no", "B", "B", "C", "no"]

        report = classification_report(y_true, y_pred)

        self.assertAlmostEqual(report["perclass"]["C"], 2 / 3)
        self.assertIn("macroF1", report)
        self.assertIn("acc", report)
        self.assertEqual(report["n"], 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: FAIL because `classification_report` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `eval/fuse_c_v.py`:

```python
from sklearn.metrics import f1_score

CLASSES = ["no", "A", "B", "C"]


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    per = f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    return {
        "n": len(y_true),
        "acc": float(sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)),
        "macroF1": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "perclass": {c: float(v) for c, v in zip(CLASSES, per)},
        "C_F1": float(per[CLASSES.index("C")]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Task 3: Feature Builder

**Files:**
- Modify: `tests/test_fuse_c_v.py`
- Modify: `eval/fuse_c_v.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fuse_c_v.py`:

```python
import numpy as np
from eval.fuse_c_v import build_feature_matrix


class FuseCVFeatureTests(unittest.TestCase):
    def test_build_feature_matrix_combines_labels_probs_and_agreement(self):
        pair_ids = ["p1", "p2"]
        llm = {"p1": "C", "p2": "B"}
        local = {"p1": "B", "p2": "B"}
        local_probs = {
            "p1": [0.1, 0.1, 0.7, 0.1],
            "p2": [0.1, 0.1, 0.2, 0.6],
        }

        X, names = build_feature_matrix(pair_ids, llm, local, local_probs)

        self.assertEqual(X.shape[0], 2)
        self.assertIn("agree", names)
        self.assertIn("llm_is_C", names)
        self.assertIn("local_margin", names)
        self.assertEqual(float(X[0, names.index("agree")]), 0.0)
        self.assertEqual(float(X[1, names.index("agree")]), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: FAIL because `build_feature_matrix` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `eval/fuse_c_v.py`:

```python
import numpy as np


def _one_hot(label: str) -> list[float]:
    return [1.0 if label == c else 0.0 for c in CLASSES]


def _margin(probs: list[float]) -> float:
    arr = np.sort(np.asarray(probs, dtype=float))
    return float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0


def build_feature_matrix(
    pair_ids: list[str],
    llm_labels: dict[str, str],
    local_labels: dict[str, str],
    local_probs: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = (
        [f"llm_is_{c}" for c in CLASSES]
        + [f"local_is_{c}" for c in CLASSES]
        + ["agree", "llm_is_C", "local_is_C", "llm_C_local_not_C", "local_no_llm_bridge"]
    )
    if local_probs is not None:
        names += [f"local_p_{c}" for c in CLASSES] + ["local_margin"]

    rows = []
    for pid in pair_ids:
        llm = llm_labels[pid]
        local = local_labels[pid]
        row = _one_hot(llm) + _one_hot(local)
        row += [
            float(llm == local),
            float(llm == "C"),
            float(local == "C"),
            float(llm == "C" and local != "C"),
            float(local == "no" and llm != "no"),
        ]
        if local_probs is not None:
            probs = [float(v) for v in local_probs[pid]]
            row += probs + [_margin(probs)]
        rows.append(row)
    return np.asarray(rows, dtype=float), names
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Task 4: Event-Disjoint Verifier Training

**Files:**
- Modify: `tests/test_fuse_c_v.py`
- Modify: `eval/fuse_c_v.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fuse_c_v.py`:

```python
from eval.fuse_c_v import event_disjoint_split


class FuseCVSplitTests(unittest.TestCase):
    def test_event_disjoint_split_never_shares_events(self):
        events = ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"]
        train_idx, test_idx = event_disjoint_split(events, test_size=0.25, seed=7)

        train_events = {events[i] for i in train_idx}
        test_events = {events[i] for i in test_idx}

        self.assertTrue(train_events)
        self.assertTrue(test_events)
        self.assertTrue(train_events.isdisjoint(test_events))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: FAIL because `event_disjoint_split` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `eval/fuse_c_v.py`:

```python
def event_disjoint_split(events: list[str], test_size: float = 0.25, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(events)))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_size)))
    test_events = set(unique[:n_test])
    train_idx, test_idx = [], []
    for i, event in enumerate(events):
        (test_idx if event in test_events else train_idx).append(i)
    return train_idx, test_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Task 5: CLI Wiring and JSON Outputs

**Files:**
- Modify: `eval/fuse_c_v.py`
- Modify: `tests/test_fuse_c_v.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fuse_c_v.py`:

```python
from eval.fuse_c_v import train_and_eval_verifier


class FuseCVTrainEvalTests(unittest.TestCase):
    def test_train_and_eval_verifier_improves_over_llm_on_synthetic_pattern(self):
        pair_ids = [f"p{i}" for i in range(12)]
        events = [f"e{i // 2}" for i in range(12)]
        y_true = ["C" if i % 3 == 0 else "B" for i in range(12)]
        llm = {pid: ("C" if i % 2 == 0 else "B") for i, pid in enumerate(pair_ids)}
        local = {pid: y_true[i] for i, pid in enumerate(pair_ids)}

        result = train_and_eval_verifier(pair_ids, events, y_true, llm, local, seed=3)

        self.assertIn("fewshot_llm", result)
        self.assertIn("fuse_c_v", result)
        self.assertGreaterEqual(result["fuse_c_v"]["C_F1"], result["fewshot_llm"]["C_F1"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: FAIL because `train_and_eval_verifier` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `eval/fuse_c_v.py`:

```python
import json
from sklearn.linear_model import LogisticRegression


def train_and_eval_verifier(
    pair_ids: list[str],
    events: list[str],
    y_true: list[str],
    llm_labels: dict[str, str],
    local_labels: dict[str, str],
    local_probs: dict[str, list[float]] | None = None,
    seed: int = 0,
) -> dict:
    X, feature_names = build_feature_matrix(pair_ids, llm_labels, local_labels, local_probs)
    train_idx, test_idx = event_disjoint_split(events, seed=seed)
    y = np.asarray(y_true)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx]).tolist()
    llm_pred = [llm_labels[pair_ids[i]] for i in test_idx]
    gold = y[test_idx].tolist()
    return {
        "feature_names": feature_names,
        "test_pair_ids": [pair_ids[i] for i in test_idx],
        "fewshot_llm": classification_report(gold, llm_pred),
        "fuse_c_v": classification_report(gold, pred),
        "predictions": {pair_ids[i]: pred[j] for j, i in enumerate(test_idx)},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Task 6: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update reproduction commands**

Add after the few-shot LLM command:

```markdown
# same-budget verifier: one few-shot LLM prediction per pair plus local FUSE-C evidence
python eval/fuse_c_v.py --root eval --out results
```

- [ ] **Step 2: Add artifact note**

Add:

```markdown
`eval/fuse_c_v.py` expects per-pair artifacts rather than summary JSONs: gold rows with event ids,
few-shot LLM predictions, and local FUSE-C predictions or probabilities. If these files are absent,
the script reports the missing paths and exits without fabricating results.
```

- [ ] **Step 3: Run tests**

Run:

```bash
python3 -m unittest tests.test_fuse_c_v -v
```

Expected: PASS.

## Self-Review

Spec coverage:
- Same-budget verifier: Task 5.
- Artifact validation: Task 1.
- Feature construction from LLM/local evidence: Task 3.
- Event-disjoint training: Task 4.
- Metrics and outputs: Tasks 2 and 5.
- Documentation: Task 6.

Known implementation limit:
- The plan implements Stage 1 with label/probability artifacts. It does not implement Stage 2 explanation-aware prompting.
