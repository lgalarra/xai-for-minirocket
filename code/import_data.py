import os.path
import pickle

import numpy as np
import pandas as pd
from export_data import CHANNELS
from reference import REFERENCE_POLICIES
from export_data import (DataExporter, SEGMENTED_EXPLANATION_SEGMENTS, TSHAP_CONFIGS,
                         get_beta_segmented_column, get_beta_tshap_column, get_tshap_key)


class DataImporter:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_path = f"data/{dataset_name}"

    def get_attributions_path(self, classifier_name, explainer_method, label, distance=None):
        path = f"{self.data_path}/{classifier_name}/{explainer_method}/{label}"
        if distance is not None:
            return path + f"/{distance}"
        return path

    def get_metadata(self, classifier_name, explainer_method, label, distance, reference_policy=None) -> pd.DataFrame:
        attributions_path = self.get_attributions_path(classifier_name, explainer_method, label, distance)
        if distance == 'euclidean' and not os.path.exists(attributions_path):
            attributions_path = self.get_attributions_path(classifier_name, explainer_method, label)
        if reference_policy is not None:
            policy_metadata_path = (
                f"{attributions_path}/"
                f"{DataExporter.get_metadata_filename_for_reference_policy(reference_policy)}"
            )
            if os.path.exists(policy_metadata_path):
                return pd.read_csv(policy_metadata_path)

        legacy_metadata_path = f"{attributions_path}/{DataExporter.METADATA_FILE}"
        if os.path.exists(legacy_metadata_path):
            return pd.read_csv(legacy_metadata_path)

        metadata_frames = []
        for policy in REFERENCE_POLICIES:
            policy_metadata_path = (
                f"{attributions_path}/"
                f"{DataExporter.get_metadata_filename_for_reference_policy(policy)}"
            )
            if os.path.exists(policy_metadata_path):
                metadata_frames.append(pd.read_csv(policy_metadata_path))
        if metadata_frames:
            return pd.concat(metadata_frames, ignore_index=True)

        return pd.read_csv(legacy_metadata_path)


    @staticmethod
    def _read_optional_series(df_instance_id_channel, column):
        if column not in df_instance_id_channel.columns:
            return None
        values = df_instance_id_channel[column].dropna().values
        if len(values) == 0:
            return None
        return pd.read_csv(values[0], header=None, index_col=0).iloc[:, 0].values

    @staticmethod
    def get_series_from_metadata(metadata_df, reference_policies=None):
        if reference_policies is None:
            policy_indices = list(enumerate(REFERENCE_POLICIES))
        else:
            if isinstance(reference_policies, str):
                reference_policies = [reference_policies]

            unknown = [p for p in reference_policies if p not in REFERENCE_POLICIES]
            if unknown:
                raise ValueError(f"Unknown reference policies: {unknown}")

            policy_indices = [(REFERENCE_POLICIES.index(p), p) for p in reference_policies]

        groups = metadata_df.groupby('instance_id')
        references = {}
        instances = []
        explanations = {}
        p2p_explanations = {}
        segmented_explanations = {}
        tshap_explanations = {}
        ys = []

        for i, reference_policy in policy_indices:
            references[REFERENCE_POLICIES[i]] = []
            explanations[REFERENCE_POLICIES[i]] = []
            p2p_explanations[REFERENCE_POLICIES[i]] = []
            segmented_explanations[reference_policy] = {
                num_segments: [] for num_segments in SEGMENTED_EXPLANATION_SEGMENTS
            }
            tshap_explanations[reference_policy] = {
                get_tshap_key(*config): [] for config in TSHAP_CONFIGS
            }

        for instance_id, df_instance in groups:
            instances.append([])
            for i, reference_policy in policy_indices:
                explanations[reference_policy].append([])
                references[reference_policy].append([])
                p2p_explanations[reference_policy].append([])
                for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
                    segmented_explanations[reference_policy][num_segments].append([])
                for window_size_percent, stride in TSHAP_CONFIGS:
                    tshap_explanations[reference_policy][get_tshap_key(window_size_percent, stride)].append([])

            for channel, df_instance_id_channel in df_instance.groupby('channel'):
                channel_values = pd.read_csv(df_instance_id_channel['series'].values[0], header=None, index_col=0).iloc[:, 0]
                instances[len(instances) - 1].append(channel_values.values)
                for i, reference_policy in policy_indices:
                    reference_channel_values = DataImporter._read_optional_series(
                        df_instance_id_channel, f'reference_{i}'
                    )

                    references[REFERENCE_POLICIES[i]][len(references[REFERENCE_POLICIES[i]]) - 1].append(
                        reference_channel_values)

                    beta_channel_values = DataImporter._read_optional_series(
                        df_instance_id_channel, f'beta_{i}_attributions'
                    )
                    explanations[REFERENCE_POLICIES[i]][len(explanations[REFERENCE_POLICIES[i]]) - 1].append(
                        beta_channel_values)
                    
                    beta_p2p_channel_values = DataImporter._read_optional_series(
                        df_instance_id_channel, f'beta_p2p_{i}_attributions'
                    )
                    

                    p2p_explanations[REFERENCE_POLICIES[i]][len(p2p_explanations[REFERENCE_POLICIES[i]]) - 1].append(beta_p2p_channel_values)

                    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
                        beta_segmented_channel_values = DataImporter._read_optional_series(
                            df_instance_id_channel, get_beta_segmented_column(i, num_segments)
                        )
                        segmented_explanations[reference_policy][num_segments][
                            len(segmented_explanations[reference_policy][num_segments]) - 1
                        ].append(beta_segmented_channel_values)

                    for window_size_percent, stride in TSHAP_CONFIGS:
                        key = get_tshap_key(window_size_percent, stride)
                        beta_tshap_channel_values = DataImporter._read_optional_series(
                            df_instance_id_channel, get_beta_tshap_column(i, window_size_percent, stride)
                        )
                        tshap_explanations[reference_policy][key][
                            len(tshap_explanations[reference_policy][key]) - 1
                        ].append(beta_tshap_channel_values)

            ys.append(df_instance['label'].iloc[0])

        def arrayify(value):
            if isinstance(value, dict):
                return {k: arrayify(v) for k, v in value.items()}
            try:
                return np.array(value)
            except Exception:
                return value


        return (np.array(instances), np.array(ys), arrayify(references), arrayify(explanations),
                arrayify(p2p_explanations), arrayify(segmented_explanations),
                arrayify(tshap_explanations))

    def import_classifier(self, mr_classifier):
        return pickle.load(open(f"{self.data_path}/{mr_classifier}/{mr_classifier}.pkl", 'rb'))
