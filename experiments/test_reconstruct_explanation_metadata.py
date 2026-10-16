from pathlib import Path
import tempfile
import unittest

import pandas as pd

from reconstruct_explanation_metadata import map_path, stage_frame


class ReconstructionTests(unittest.TestCase):
    def test_mapping_rejects_unexpected_paths(self):
        self.assertEqual(map_path('/remote/experiments/lrp-data/example/A.csv'), 'data/example/A.csv')
        for value in ['data/example/A.csv', 'lrp-data/../secret.csv']:
            with self.assertRaises(ValueError):
                map_path(value)

    def test_required_paths_remapped_lrp_removed_optional_missing_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = 'example/model/shap/predicted/euclidean'
            paths = {'series': 'data/example/A/A_instance_0.csv',
                     'reference_0': f'data/{config}/A/A_reference_hash.csv',
                     'beta_0_attributions': f'data/{config}/A/beta_instance_0_0.csv'}
            for path in paths.values():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('0,1\n')
            row = {column: path.replace('data/', 'lrp-data/', 1) for column, path in paths.items()}
            row.update(instance_id=0, channel='A', beta_lrp_0_attributions='unused',
                       beta_p2p_0_attributions=f'lrp-data/{config}/A/missing.csv')
            frame, missing = stage_frame(pd.DataFrame([row]), root, config, 0)
            self.assertNotIn('beta_lrp_0_attributions', frame)
            self.assertTrue(pd.isna(frame.beta_p2p_0_attributions.iloc[0]))
            self.assertEqual(len(missing), 1)
            self.assertEqual(frame.beta_0_attributions.iloc[0], paths['beta_0_attributions'])
            (root / paths['beta_0_attributions']).unlink()
            with self.assertRaises(FileNotFoundError):
                stage_frame(pd.DataFrame([row]), root, config, 0)


if __name__ == '__main__':
    unittest.main()
