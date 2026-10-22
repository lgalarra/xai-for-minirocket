import argparse
import unittest

import numpy as np

from predict_and_explain import Explanation, aggregate_explanations, parse_reference_policy, parse_reference_policies


class ReferencePolicyAggregationTests(unittest.TestCase):
    @staticmethod
    def explanation(coefficients, alphas, elapsed=1.0, reference_convolution=None):
        return Explanation({
            "coefficients": np.asarray(coefficients),
            "minirocket_coefficients": np.asarray(alphas),
            "time_elapsed": elapsed,
            "selected_features": [0],
            "reference_traces": [{
                "conv_sum": np.asarray(reference_convolution if reference_convolution is not None else [0, 0])
            }],
        })

    def test_parses_aggregation_and_further_alias(self):
        specification = parse_reference_policy(
            "max(opposite_class_medoid+opposite_class_further_instance)"
        )
        self.assertEqual(specification.aggregation, "max")
        self.assertEqual(specification.policies, (
            "opposite_class_medoid", "opposite_class_farthest_instance"
        ))
        self.assertEqual(
            specification.name,
            "max(opposite_class_medoid+opposite_class_farthest_instance)",
        )

    def test_parses_comma_separated_simple_and_aggregated_policies(self):
        specifications = parse_reference_policies(
            "global_medoid,avg(opposite_class_medoid+opposite_class_closest_instance)"
        )
        self.assertEqual([specification.aggregation for specification in specifications], [None, "avg"])

    def test_rejects_invalid_aggregated_policies(self):
        for value in (
            "max(opposite_class_medoid)",
            "max(opposite_class_medoid+opposite_class_medoid)",
            "median(opposite_class_medoid+global_medoid)",
        ):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_reference_policy(value)

    def test_aggregates_signed_attributions_pointwise(self):
        specification = parse_reference_policy(
            "max(opposite_class_medoid+opposite_class_farthest_instance)"
        )
        first = self.explanation([[-4, 2]], [-3, 5], elapsed=1.5, reference_convolution=[-2, 4])
        second = self.explanation([[3, -1]], [4, -2], elapsed=2.0, reference_convolution=[3, -1])
        result = aggregate_explanations([first, second], specification).explanation

        np.testing.assert_array_equal(result["coefficients"], [[3, 2]])
        np.testing.assert_array_equal(result["minirocket_coefficients"], [-3, 5])
        np.testing.assert_array_equal(result["reference_traces"][0]["conv_sum"], [3, 4])
        self.assertEqual(result["time_elapsed"], 3.5)
        self.assertEqual(result["aggregation"], "max")
        self.assertIsNone(result["selected_features"])


if __name__ == "__main__":
    unittest.main()
