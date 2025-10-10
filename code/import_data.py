import pickle

import numpy as np
import pandas as pd
from export_data import CHANNELS
from reference import REFERENCE_POLICIES


class DataImporter:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_path = f"data/{dataset_name}"

    def get_attributions_path(self, classifier_name, explainer_method, label):
        return f"{self.data_path}/{classifier_name}/{explainer_method}/{label}"

    def get_metadata(self, classifier_name, explainer_method, label) -> pd.DataFrame:
        attributions_path = self.get_attributions_path(classifier_name, explainer_method, label)
        return pd.read_csv(f"{attributions_path}/metadata.csv")

    def fetch_computed_attributions(self, instance_id, classifier_name, label, reference_policy,
                                    explainer_method, type='backpropagated'):
        attributions_path = self.get_attributions_path(classifier_name, explainer_method, label)
        metadata_path = f"{attributions_path}/metadata.csv"

    @staticmethod
    def get_series_from_metadata(metadata_df):
        groups = metadata_df.groupby('instance_id')
        references = {}
        instances = []
        explanations = {}
        p2p_explanations = {}
        segmented_explanations = {}
        ys = []

        for i in range(len(REFERENCE_POLICIES)):
            references[REFERENCE_POLICIES[i]] = []
            explanations[REFERENCE_POLICIES[i]] = []
            p2p_explanations[REFERENCE_POLICIES[i]] = []
            segmented_explanations[REFERENCE_POLICIES[i]] = []

        for instance_id, df_instance in groups:
            instances.append([])
            for i in range(len(REFERENCE_POLICIES)):
                explanations[REFERENCE_POLICIES[i]].append([])
                references[REFERENCE_POLICIES[i]].append([])
                p2p_explanations[REFERENCE_POLICIES[i]].append([])
                segmented_explanations[REFERENCE_POLICIES[i]].append([])

            for channel, df_instance_id_channel in df_instance.groupby('channel'):
                channel_values = pd.read_csv(df_instance_id_channel['series'].values[0], header=None, index_col=0).iloc[:, 0]
                instances[len(instances) - 1].append(channel_values.values)
                for i in range(len(REFERENCE_POLICIES)):
                    reference_channel_values = pd.read_csv(df_instance_id_channel[f'reference_{i}'].values[0],
                                                           header=None, index_col=0).iloc[:, 0]

                    references[REFERENCE_POLICIES[i]][len(references[REFERENCE_POLICIES[i]]) - 1].append(
                        reference_channel_values.values)

                    beta_channel_values = pd.read_csv(df_instance_id_channel[f'beta_{i}_attributions'].values[0],
                                                      header=None, index_col=0).iloc[:, 0]
                    explanations[REFERENCE_POLICIES[i]][len(explanations[REFERENCE_POLICIES[i]]) - 1].append(
                        beta_channel_values.values)

                    beta_p2p_channel_values = pd.read_csv(df_instance_id_channel[f'beta_p2p_{i}_attributions'].values[0],
                                                          header=None, index_col=0).iloc[:, 0]

                    p2p_explanations[REFERENCE_POLICIES[i]][len(p2p_explanations[REFERENCE_POLICIES[i]]) - 1].append(beta_p2p_channel_values.values)

                    beta_segmented_channel_values = pd.read_csv(df_instance_id_channel[f'beta_segmented_{i}_attributions'].values[0],
                                                                header=None, index_col=0).iloc[:, 0]

                    segmented_explanations[REFERENCE_POLICIES[i]][len(segmented_explanations[REFERENCE_POLICIES[i]]) - 1].append(beta_segmented_channel_values.values)

            ys.append(df_instance['label'].iloc[0])

        for series_dict in [references, explanations, p2p_explanations, segmented_explanations]:
            for _k, v in series_dict.items():
                series_dict[_k] = np.array(v)

        return np.array(instances), np.array(ys), references, explanations, p2p_explanations, segmented_explanations

    def import_classifier(self, mr_classifier):
        return pickle.load(open(f"{self.data_path}/{mr_classifier}/{mr_classifier}.pkl", 'rb'))