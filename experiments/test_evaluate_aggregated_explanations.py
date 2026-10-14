import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_aggregated_explanations import (
    combine_variants, count_basis, evaluate, load_inputs, parse_args, parse_reference_group, reference_targets,
)


class AggregatedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "medoid": {"reference": np.array([[[1., 3.]]]),
                       "variants": {"backpropagated": np.array([[[-2., 4.]]])}},
            "closest": {"reference": np.array([[[5., 7.]]]),
                        "variants": {"backpropagated": np.array([[[3., -1.]]])}},
        }

    def test_group_parsing_and_default(self):
        self.assertEqual(parse_reference_group("medoid+closest"), ["medoid", "closest"])
        for value in ("", "medoid+", "medoid+medoid"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_reference_group(value)
        args = parse_args(["double-freq-test", "predicted", "shap", "medoid+closest", "both", "RandomForestClassifier"])
        self.assertEqual(args.aggregation, "avg")
        self.assertTrue(args.compare_individual)

    def test_individual_comparison_can_be_disabled(self):
        arguments = ["ford-a", "predicted", "shap", "medoid+closest", "both", "LogisticRegression"]
        self.assertFalse(parse_args(arguments + ["--no-compare-individual"]).compare_individual)
        self.assertTrue(parse_args(arguments + ["--compare-individual"]).compare_individual)

    def test_loader_selects_policy_rows_from_sparse_legacy_metadata(self):
        frame = pd.DataFrame([
            {"instance_id": 7, "channel": "C", "reference_0": "first.csv"},
            {"instance_id": 7, "channel": "C", "reference_1": "second.csv"},
        ])

        class Importer:
            def __init__(self, dataset):
                pass

            def get_metadata(self, *args):
                return frame

            @staticmethod
            def get_series_from_metadata(selected, policies):
                self.assertEqual(len(selected), 1)
                policy = policies[0]
                values = np.ones((1, 1, 2))
                return values, np.array([1]), {policy: values}, {policy: values}, {policy: [None]}, {policy: {}}, {policy: {}}

        evaluator = SimpleNamespace(DataImporter=Importer,
                                    get_reference_policy_indices=lambda policies: [(0 if policies[0] == "medoid" else 1, policies[0])],
                                    has_explanations=lambda values: np.asarray(values, dtype=object).flat[0] is not None)
        args = parse_args(["example", "predicted", "shap", "medoid+closest", "both", "model"])
        instance_ids, inputs = load_inputs(evaluator, args, args.reference_policy)
        self.assertEqual(instance_ids, [7])
        self.assertEqual(set(inputs), {"medoid", "closest"})
        frame.loc[len(frame)] = {"instance_id": 2, "channel": "C", "reference_0": "earlier.csv"}
        args.start = 7
        instance_ids, _ = load_inputs(evaluator, args, args.reference_policy)
        self.assertEqual(instance_ids, [7])
        args.start = 8
        with self.assertRaisesRegex(ValueError, "Missing"):
            load_inputs(evaluator, args, args.reference_policy)

    def test_output_name_uses_experiment_arguments_and_allows_override(self):
        arguments = ["ford-a", "predicted", "shap", "medoid+closest", "both", "LogisticRegression"]
        args = parse_args(arguments + ["--aggregation", "sum"])
        self.assertEqual(args.output.parent.name, "aggregated-results")
        self.assertEqual(args.output.name, "ford-a__predicted__shap__medoid+closest__LogisticRegression__sum__both.csv")
        for index, value in enumerate(["double-freq-test", "training", "gradients", "medoid", "gaussian", "RandomForestClassifier"]):
            changed = arguments.copy()
            changed[index] = value
            self.assertNotEqual(parse_args(changed + ["--aggregation", "sum"]).output, args.output)
        self.assertNotEqual(parse_args(arguments).output, args.output)
        self.assertEqual(parse_args(arguments + ["--output", "custom/results.csv"]).output, Path("custom/results.csv"))

    def test_inclusive_instance_interval(self):
        arguments = ["ford-a", "predicted", "shap", "medoid+closest", "both", "LogisticRegression"]
        args = parse_args(arguments + ["--start", "86", "--end", "99"])
        self.assertEqual(args.instances, list(range(86, 100)))
        args = parse_args(arguments + ["--start", "0", "--end", "0"])
        self.assertEqual(args.instances, [0])
        args = parse_args(arguments + ["--start", "86"])
        self.assertEqual(args.start, 86)
        self.assertIsNone(args.end)
        self.assertIsNone(args.instances)

    def test_invalid_instance_intervals(self):
        arguments = ["ford-a", "predicted", "shap", "medoid+closest", "both", "LogisticRegression"]
        for options in (["--start", "-1"], ["--end", "2"],
                        ["--start", "1", "--instances", "1"],
                        ["--start", "2", "--end", "1"],
                        ["--start", "-1", "--end", "2"],
                        ["--start", "1", "--end", "2", "--instances", "1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(arguments + options)
            self.assertEqual(error.exception.code, 2)

    def test_aggregate_preserves_signed_values_and_instance_axes(self):
        expected = {"max": [3., 4.], "min": [-2., -1.], "avg": [0.5, 1.5], "sum": [1., 3.]}
        for aggregation, values in expected.items():
            result = combine_variants(self.inputs, ["medoid", "closest"], aggregation)
            np.testing.assert_array_equal(result["backpropagated"], [[values]])

    def test_reference_target_semantics(self):
        policies = ["medoid", "closest"]
        targets = reference_targets(self.inputs, policies, "average-scores")
        self.assertEqual([name for name, _ in targets], policies)
        np.testing.assert_array_equal(targets[0][1], [[[1., 3.]]])
        np.testing.assert_array_equal(reference_targets(self.inputs, policies, "mean-reference")[0][1], [[[3., 5.]]])
        self.assertEqual(len(reference_targets(self.inputs, policies, "first")), 1)

    def test_missing_variant_is_not_partially_aggregated(self):
        self.inputs["medoid"]["variants"]["p2p"] = np.ones((1, 1, 2))
        result = combine_variants(self.inputs, ["medoid", "closest"], "max")
        self.assertNotIn("p2p", result)
        self.inputs["closest"]["variants"] = {}
        with self.assertRaises(ValueError):
            combine_variants(self.inputs, ["medoid", "closest"], "max")

    def test_count_basis_uses_p2p_then_smallest_nonzero_variant(self):
        variants = {"backpropagated": np.ones((2, 1, 4)), "p2p": np.array([[[1, 0, 0, 0]], [[1, 1, 0, 0]]])}
        np.testing.assert_array_equal(count_basis(variants), [1, 2])
        variants["segmented_n10"] = variants.pop("p2p")
        np.testing.assert_array_equal(count_basis(variants), [1, 2])

    def test_evaluation_averages_scores_and_reuses_random_draws(self):
        draws = []

        def compute(classifier, series, reference, explanations, explainer, policy, parameters, budget):
            draws.append(np.random.random(budget))
            metric = np.repeat(reference.mean(axis=(1, 2)) ** 2, budget)
            return metric, metric, 0., 1

        def append(results, prefix, metric, normalized, change):
            results["f_minus_f0-mean"] = [np.mean(metric)]

        evaluator = SimpleNamespace(
            get_n_perturbed_points=lambda counts, percentile: np.ceil(counts * (100 - percentile) / 100).astype(int),
            compute_perturbation_metrics=compute,
            add_metric_columns=lambda results, prefix: None,
            append_metrics=append,
        )
        for values in self.inputs.values():
            values.update(series=np.ones((1, 1, 2)), labels=np.array([1]))
        args = parse_args(["example", "predicted", "shap", "medoid+closest", "both", "model", "--compare-individual"])
        state_before = np.random.get_state()
        summary, instances, raw = evaluate(evaluator, None, self.inputs, [7], args)
        np.testing.assert_array_equal(np.random.get_state()[1], state_before[1])
        combined = summary[summary.reference_policy == "medoid+closest"]
        np.testing.assert_array_equal(combined["f_minus_f0-mean"], [20., 20.])
        np.testing.assert_array_equal(draws[0], draws[1])
        self.assertEqual(len(instances), 6)
        self.assertEqual(len(raw), 44)


if __name__ == "__main__":
    unittest.main()
