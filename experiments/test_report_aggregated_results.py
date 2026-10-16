from pathlib import Path
import tempfile
import unittest

import pandas as pd

from report_aggregated_results import build_tables, latex_table, load_comparisons, markdown_table


def result_rows(classifier, policies, scores, instances="0;1"):
    common = dict(dataset="example", mr_classifier=classifier, base_explainer="shap",
                  attribution_type="backpropagated", aggregation="avg", perturbation_policy="gaussian",
                  label="predicted", distance="euclidean", reference_strategy="average-scores", seed=0,
                  args="{'budget': 10, 'sigma': 3}", count_basis="shared_minimum",
                  instance_ids=instances, n_perturbed_points=";".join("2" for _ in instances.split(";")))
    return [dict(common, reference_policy=policy, **{"f_minus_f0-mean": score})
            for policy, score in zip(["+".join(policies), *policies], scores)]


class ReportTests(unittest.TestCase):
    def test_two_and_three_policy_groups_preserve_their_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = result_rows("model", ["a", "b"], [0.5, 0.2, 0.3])
            rows += result_rows("model", ["a", "b", "c"], [0.8, 0.4, 0.6, 0.7])
            pd.DataFrame(rows[:3]).to_csv(Path(directory) / "two.csv", index=False)
            pd.DataFrame(rows[3:]).to_csv(Path(directory) / "three.csv", index=False)
            pd.DataFrame({"irrelevant": [1]}).to_csv(Path(directory) / "two-instances.csv", index=False)
            records, _ = load_comparisons(directory)
            dataset, classifier, legends = build_tables(records)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(classifier), 2)
            self.assertEqual(len(legends["Policies"]), 3)
            self.assertEqual(sorted(classifier.P1), [0.2, 0.4])
            self.assertEqual(classifier.P3.isna().sum(), 1)

    def test_dataset_is_macro_average_not_instance_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(result_rows("small", ["a", "b"], [0.8, 0.1, 0.2], "0")).to_csv(Path(directory) / "small.csv", index=False)
            pd.DataFrame(result_rows("large", ["a", "b"], [0.4, 0.5, 0.1], "0;1;2")).to_csv(Path(directory) / "large.csv", index=False)
            records, _ = load_comparisons(directory)
            dataset, _, _ = build_tables(records)
            self.assertAlmostEqual(dataset.Combined.iloc[0], 0.6)
            self.assertAlmostEqual(dataset["Delta best"].iloc[0], 0.3)

    def test_rejects_missing_baselines_mismatched_coverage_and_duplicates(self):
        for failure in ("missing", "coverage", "duplicate", "nonfinite"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                rows = result_rows("model", ["a", "b"], [0.5, 0.2, 0.3])
                if failure == "missing":
                    rows.pop()
                elif failure == "coverage":
                    rows[1]["instance_ids"] = "0"
                elif failure == "nonfinite":
                    rows[0]["f_minus_f0-mean"] = float("inf")
                pd.DataFrame(rows).to_csv(Path(directory) / "run.csv", index=False)
                if failure == "duplicate":
                    pd.DataFrame(rows).to_csv(Path(directory) / "copy.csv", index=False)
                with self.assertRaises(ValueError):
                    load_comparisons(directory)

    def test_renderers_escape_labels_and_show_missing_values(self):
        frame = pd.DataFrame({"dataset_name": ["a&b|c"], "Score": [float("nan")]})
        self.assertIn(r"a&b\|c", markdown_table(frame))
        self.assertIn("—", markdown_table(frame))
        self.assertIn(r"dataset\_name", latex_table(frame))
        self.assertIn(r"a\&b|c", latex_table(frame))


if __name__ == "__main__":
    unittest.main()
