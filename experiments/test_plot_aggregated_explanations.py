import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from plot_aggregated_explanations import aggregate_explanations, main


class AggregationTests(unittest.TestCase):
    def test_signed_pointwise_reductions(self):
        explanations = [np.array([[-4, 2], [6, -8]]), np.array([[2, 4], [-2, -4]])]
        expected = {
            "max": [[2, 4], [6, -4]], "min": [[-4, 2], [-2, -8]],
            "sum": [[-2, 6], [4, -12]], "avg": [[-1, 3], [2, -6]],
        }
        for aggregation, values in expected.items():
            with self.subTest(aggregation=aggregation):
                np.testing.assert_array_equal(aggregate_explanations(explanations, aggregation), values)

    def test_invalid_arrays(self):
        for explanations in ([], [[1], [1, 2]], [[np.nan]], [[np.inf]], [[]]):
            with self.subTest(explanations=explanations), self.assertRaises(ValueError):
                aggregate_explanations(explanations, "avg")

    def test_export_to_image_and_missing_policy_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "data/example/model/shap/predicted/euclidean"
            directory.mkdir(parents=True)
            policies = ["global_medoid", "global_centroid"]
            for policy_index, policy in enumerate(policies):
                rows = []
                for channel in ("A", "B"):
                    series_path = root / f"series_{channel}.csv"
                    beta_path = root / f"beta_{policy}_{channel}.csv"
                    pd.Series([1., 3., 2.]).to_csv(series_path, header=False)
                    pd.Series([-1., policy_index, 2.]).to_csv(beta_path, header=False)
                    rows.append({
                        "instance_id": 7, "channel": channel, "label": 1,
                        "series": series_path.name,
                        f"reference_{policy_index}": "unused.csv",
                        f"beta_{policy_index}_attributions": beta_path.name,
                    })
                pd.DataFrame(rows).to_csv(directory / f"metadata_ref_policy_{policy}.csv", index=False)
            arguments = [
                "--dataset", "example", "--classifier", "model", "--explainer", "shap",
                "--reference-policies", ",".join(policies), "--aggregation", "avg",
                "--export-root", str(root), "--output-dir", str(root / "plots"),
            ]
            main(arguments)
            image = next((root / "plots").glob("*.png"))
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata_path = directory / "metadata_ref_policy_global_centroid.csv"
            pd.read_csv(metadata_path).iloc[:1].to_csv(metadata_path, index=False)
            with self.assertRaisesRegex(ValueError, "coverage"):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
