import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from eval.fuse_c_v import (
    ArtifactError,
    build_feature_matrix,
    classification_report,
    event_disjoint_split,
    main,
    require_files,
    run_from_artifacts,
    train_and_eval_verifier,
)


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

    def test_main_returns_2_when_required_artifacts_are_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "empty"
            out = Path(td) / "out"
            root.mkdir()

            err = StringIO()
            with redirect_stderr(err):
                code = main(["--root", str(root), "--out", str(out)])

            self.assertEqual(code, 2)
            self.assertIn("gold_consensus.jsonl", err.getvalue())

    def test_run_from_artifacts_writes_metrics_and_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            out = Path(td) / "out"
            root.mkdir()

            pair_ids = [f"p{i}" for i in range(12)]
            gold_rows = [
                {"pair_id": pid, "event_id": f"e{i // 2}", "label": "C" if i % 3 == 0 else "B"}
                for i, pid in enumerate(pair_ids)
            ]
            (root / "gold_consensus.jsonl").write_text(
                "\n".join(json.dumps(row) for row in gold_rows) + "\n"
            )
            (root / "fewshot_llm_preds.json").write_text(
                json.dumps({pid: ("C" if i % 2 == 0 else "B") for i, pid in enumerate(pair_ids)})
            )
            (root / "local_fuse_c_preds.json").write_text(
                json.dumps({pid: gold_rows[i]["label"] for i, pid in enumerate(pair_ids)})
            )

            result = run_from_artifacts(root, out, seed=3)

            self.assertTrue((out / "fuse_c_v_result.json").exists())
            self.assertTrue((out / "fuse_c_v_preds.json").exists())
            self.assertIn("fewshot_llm", result)
            self.assertIn("fuse_c_v", result)


class FuseCVMetricsTests(unittest.TestCase):
    def test_classification_report_returns_macro_and_c_f1(self):
        y_true = ["no", "A", "B", "C", "C"]
        y_pred = ["no", "B", "B", "C", "no"]

        report = classification_report(y_true, y_pred)

        self.assertAlmostEqual(report["perclass"]["C"], 2 / 3)
        self.assertIn("macroF1", report)
        self.assertIn("acc", report)
        self.assertEqual(report["n"], 5)


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
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(float(X[0, names.index("agree")]), 0.0)
        self.assertEqual(float(X[1, names.index("agree")]), 1.0)


class FuseCVSplitTests(unittest.TestCase):
    def test_event_disjoint_split_never_shares_events(self):
        events = ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"]
        train_idx, test_idx = event_disjoint_split(events, test_size=0.25, seed=7)

        train_events = {events[i] for i in train_idx}
        test_events = {events[i] for i in test_idx}

        self.assertTrue(train_events)
        self.assertTrue(test_events)
        self.assertTrue(train_events.isdisjoint(test_events))


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


if __name__ == "__main__":
    unittest.main()
