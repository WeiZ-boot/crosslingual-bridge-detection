from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


CLASSES = ["no", "A", "B", "C"]
DEFAULT_REQUIRED = ["gold_consensus.jsonl", "fewshot_llm_preds.json", "local_fuse_c_preds.json"]


class ArtifactError(RuntimeError):
    """Raised when required verifier artifacts are absent."""


def require_files(root: Path, rel_paths: list[str]) -> None:
    missing = [p for p in rel_paths if not (root / p).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise ArtifactError(f"Missing required FUSE-C-V artifacts:\n  - {joined}")


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_gold(path: Path) -> tuple[list[str], list[str], list[str]]:
    pair_ids, events, labels = [], [], []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_ids.append(row["pair_id"])
            events.append(row["event_id"])
            labels.append(row["label"])
    return pair_ids, events, labels


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    per = f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    return {
        "n": len(y_true),
        "acc": float(sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)),
        "macroF1": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "perclass": {c: float(v) for c, v in zip(CLASSES, per)},
        "C_F1": float(per[CLASSES.index("C")]),
    }


def _one_hot(label: str) -> list[float]:
    return [1.0 if label == c else 0.0 for c in CLASSES]


def _margin(probs: list[float]) -> float:
    arr = np.sort(np.asarray(probs, dtype=float))
    return float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0


def event_disjoint_split(
    events: list[str], test_size: float = 0.25, seed: int = 0
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(events)))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_size)))
    test_events = set(unique[:n_test])
    train_idx, test_idx = [], []
    for i, event in enumerate(events):
        (test_idx if event in test_events else train_idx).append(i)
    return train_idx, test_idx


def build_feature_matrix(
    pair_ids: list[str],
    llm_labels: dict[str, str],
    local_labels: dict[str, str],
    local_probs: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = (
        [f"llm_is_{c}" for c in CLASSES]
        + [f"local_is_{c}" for c in CLASSES]
        + ["agree", "llm_C_local_not_C", "local_no_llm_bridge"]
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
            float(llm == "C" and local != "C"),
            float(local == "no" and llm != "no"),
        ]
        if local_probs is not None:
            probs = [float(v) for v in local_probs[pid]]
            row += probs + [_margin(probs)]
        rows.append(row)
    return np.asarray(rows, dtype=float), names


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


def run_from_artifacts(root: Path, out: Path, seed: int = 0) -> dict:
    require_files(root, DEFAULT_REQUIRED)
    pair_ids, events, labels = load_gold(root / "gold_consensus.jsonl")
    llm_preds = load_json(root / "fewshot_llm_preds.json")
    local_preds = load_json(root / "local_fuse_c_preds.json")

    result = train_and_eval_verifier(pair_ids, events, labels, llm_preds, local_preds, seed=seed)

    out.mkdir(parents=True, exist_ok=True)
    result_without_predictions = {k: v for k, v in result.items() if k != "predictions"}
    (out / "fuse_c_v_result.json").write_text(
        json.dumps(result_without_predictions, indent=2, sort_keys=True) + "\n"
    )
    (out / "fuse_c_v_preds.json").write_text(
        json.dumps(result["predictions"], indent=2, sort_keys=True) + "\n"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the FUSE-C-V verifier.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        result = run_from_artifacts(args.root, args.out, seed=args.seed)
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    metrics = result["fuse_c_v"]
    print(
        f"fuse_c_v n={metrics['n']} acc={metrics['acc']:.3f} "
        f"macroF1={metrics['macroF1']:.3f} C_F1={metrics['C_F1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
