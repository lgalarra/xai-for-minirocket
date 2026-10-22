from pathlib import Path
import tempfile
import unittest

import pandas as pd

from report_aggregated_results import (
    build_aggregation_comparison,
    build_tables,
    latex_aggregation_comparison,
    latex_table,
    load_comparisons,
    markdown_table,
)


def result_rows(classifier, policies, scores, instances="0;1", aggregation="avg",
                perturbation="gaussian", dataset="example"):
    common = dict(dataset=dataset, mr_classifier=classifier, base_explainer="shap",
                  attribution_type="backpropagated", aggregation=aggregation, perturbation_policy=perturbation,
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

    def test_aggregation_comparison_uses_min_baselines_and_macro_averages(self):
        policies = ["opposite_class_closest_instance", "opposite_class_medoid",
                    "opposite_class_farthest_instance"]
        with tempfile.TemporaryDirectory() as directory:
            rows = result_rows("model-a", policies, [0.4, 0.1, 0.2, 0.3],
                               aggregation="min", perturbation="instance_to_reference")
            rows += result_rows("model-b", policies, [0.6, 0.3, 0.4, 0.5],
                                aggregation="min", perturbation="instance_to_reference")
            rows += result_rows("model-a", policies, [0.35, 0.1, 0.2, 0.3],
                                perturbation="instance_to_reference")
            rows += result_rows("model-b", policies, [0.55, 0.3, 0.4, 0.5],
                                perturbation="instance_to_reference")
            pd.DataFrame(rows).to_csv(Path(directory) / "runs.csv", index=False)
            records, _ = load_comparisons(directory)
            comparison = build_aggregation_comparison(records)
            self.assertEqual(list(comparison.columns), ["Dataset", "Pert.", "M", "C", "F", "min", "avg"])
            self.assertEqual(comparison["Pert."].iloc[0], "O")
            self.assertAlmostEqual(comparison.M.iloc[0], 0.3)
            self.assertAlmostEqual(comparison.C.iloc[0], 0.2)
            self.assertAlmostEqual(comparison.F.iloc[0], 0.4)
            self.assertAlmostEqual(comparison["min"].iloc[0], 0.5)
            self.assertAlmostEqual(comparison["avg"].iloc[0], 0.45)

            by_classifier = build_aggregation_comparison(records, by_classifier=True)
            self.assertEqual(
                list(by_classifier.columns),
                ["Dataset", "Classifier", "Pert.", "M", "C", "F", "min", "avg"],
            )
            self.assertEqual(list(by_classifier.Classifier), ["model-a", "model-b"])
            self.assertAlmostEqual(by_classifier["min"].iloc[0], 0.4)
            self.assertAlmostEqual(by_classifier["avg"].iloc[1], 0.55)

    def test_aggregation_comparison_uses_shared_classifier_coverage(self):
        policies = ["opposite_class_closest_instance", "opposite_class_medoid",
                    "opposite_class_farthest_instance"]
        with tempfile.TemporaryDirectory() as directory:
            rows = result_rows("model-a", policies, [0.4, 0.1, 0.2, 0.3], aggregation="min")
            rows += result_rows("model-a", policies, [0.5, 0.1, 0.2, 0.3])
            rows += result_rows("model-b", policies, [0.9, 0.7, 0.8, 0.6])
            pd.DataFrame(rows).to_csv(Path(directory) / "runs.csv", index=False)
            records, _ = load_comparisons(directory)
            comparison = build_aggregation_comparison(records)
            self.assertAlmostEqual(comparison["min"].iloc[0], 0.4)
            self.assertAlmostEqual(comparison["avg"].iloc[0], 0.5)

    def test_aggregation_comparison_latex_uses_multirow_for_datasets(self):
        frame = pd.DataFrame([
            {"Dataset": "example_set", "Pert.": "O", "M": 0.1},
            {"Dataset": "example_set", "Pert.": "N", "M": 0.2},
            {"Dataset": "second", "Pert.": "O", "M": 0.3},
        ])
        rendered = latex_aggregation_comparison(frame)
        self.assertIn(
            r"\begin{tabular}{@{}l@{\hspace{3pt}}c@{\hspace{3pt}}c@{}}",
            rendered,
        )
        self.assertIn(r"\toprule", rendered)
        self.assertIn(r"\bottomrule", rendered)
        self.assertNotIn(r"\hline", rendered)
        self.assertEqual(rendered.count(r"\midrule"), 2)
        self.assertEqual(rendered.count(r"\multirow{2}{*}{example\_set}"), 1)
        self.assertIn(r" & N & \textbf{0.200}", rendered)

    def test_aggregation_comparison_latex_marks_scores_below_threshold(self):
        frame = pd.DataFrame([
            {"Dataset": "example", "Pert.": "O", "M": 0.0009,
             "C": -0.0009, "F": -0.0014, "min": 0.001, "avg": float("nan")},
        ])
        rendered = latex_aggregation_comparison(frame)
        self.assertEqual(rendered.count(r"$< 0.001$"), 2)
        self.assertIn("-0.001", rendered)
        self.assertIn("0.001", rendered)
        self.assertIn("--", rendered)

    def test_aggregation_comparison_latex_compacts_score_columns(self):
        frame = pd.DataFrame([
            {"Dataset": "example", "Pert.": "O", "M": 0.1,
             "C": 0.2, "F": 0.3, "min": 0.4, "avg": 0.5},
        ])
        rendered = latex_aggregation_comparison(frame)
        self.assertIn(
            r"\begin{tabular}{@{}l@{\hspace{3pt}}c@{\hspace{3pt}}c"
            r"@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c"
            r"@{\hspace{3pt}}c@{}}",
            rendered,
        )
        self.assertIn(r"\textbf{0.500}", rendered)

        frame.insert(1, "Classifier", "LR")
        appendix_rendered = latex_aggregation_comparison(frame, compact=False)
        self.assertIn(r"\begin{tabular}{llcccccc}", appendix_rendered)
        self.assertIn(r"\multirow{1}{*}{LR}", appendix_rendered)

    def test_classifier_comparison_uses_multirow_for_perturbations(self):
        frame = pd.DataFrame([
            {"Dataset": "example", "Classifier": "LR", "Pert.": "O", "M": 0.1},
            {"Dataset": "example", "Classifier": "LR", "Pert.": "N", "M": 0.2},
        ])
        rendered = latex_aggregation_comparison(frame, compact=False)
        self.assertEqual(rendered.count(r"\multirow{2}{*}{LR}"), 1)
        self.assertIn(r" &  & N & \textbf{0.200}", rendered)
        self.assertNotIn(r"\cmidrule", rendered)

    def test_classifier_comparison_separates_classifiers_without_crossing_dataset(self):
        frame = pd.DataFrame([
            {"Dataset": "example", "Classifier": "LR", "Pert.": "O", "M": 0.1},
            {"Dataset": "example", "Classifier": "LR", "Pert.": "N", "M": 0.2},
            {"Dataset": "example", "Classifier": "RF", "Pert.": "O", "M": 0.3},
            {"Dataset": "example", "Classifier": "RF", "Pert.": "N", "M": 0.4},
        ])
        rendered = latex_aggregation_comparison(frame, compact=False)
        self.assertEqual(rendered.count(r"\cmidrule(lr){2-4}"), 1)

    def test_aggregation_comparison_collapses_starlight_datasets(self):
        policies = ["opposite_class_closest_instance", "opposite_class_medoid",
                    "opposite_class_farthest_instance"]
        with tempfile.TemporaryDirectory() as directory:
            rows = result_rows("model", policies, [0.2, 0.1, 0.2, 0.3], aggregation="min",
                               dataset="starlight-c1")
            rows += result_rows("model", policies, [0.6, 0.3, 0.4, 0.5], aggregation="min",
                                dataset="starlight-c2")
            pd.DataFrame(rows).to_csv(Path(directory) / "starlight.csv", index=False)
            records, _ = load_comparisons(directory)
            comparison = build_aggregation_comparison(records)
            self.assertEqual(len(comparison), 1)
            self.assertEqual(comparison.Dataset.iloc[0], "starlight")
            self.assertAlmostEqual(comparison["min"].iloc[0], 0.4)


if __name__ == "__main__":
    unittest.main()
