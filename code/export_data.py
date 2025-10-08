import os
import re

import numpy as np
import pandas as pd

from classifier import MinirocketClassifier
from reference import REFERENCE_POLICIES, REFERENCE_POLICIES_LABELS
import json

from utils import COGNITIVE_CIRCLES_CHANNELS, cognitive_circles_get_sorted_channels_from_df, COGNITIVE_CIRCLES_UNITS

METADATA_COLUMNS =(['series', 'label', 'label_type', 'label_probability', 'channel', 'group', 'annotation',
                    'global_medoid_id'] + [f'reference_{i}' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_label' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_label_probability' for i in range(len(REFERENCE_POLICIES))]
                   + ['beta_attributions'] )

UNITS = {
    "ford-a": ["dB"],
    "cognitive-circles": [COGNITIVE_CIRCLES_UNITS[x] for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')]
}

DESCRIPTIONS = {
    "ford-a": ["Noise intensity"],
    "cognitive-circles": [COGNITIVE_CIRCLES_CHANNELS[x] for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')]
}

#('X', 'X'), ('V', 'velocity'), ('VA', 'angular_velocity'),
#                           ('DR', 'radial_velocity'), ('Y', 'Y'), ('D', 'radius'),  ('A', 'acceleration')


METADATA_SCHEMA = {col : [] for col in METADATA_COLUMNS}

class DataExporter(object):
    def __init__(self, dataset_name: str, mr_classifier_name: str, explainer_method: str, label_type: str):
        self.output_path = DataExporter.create_output_folder_for_export(dataset_name,
                                                                        mr_classifier_name, explainer_method,
                                                                        label_type)
        self.output_dataset_path = 'data/' + dataset_name
        self.dataset_name = dataset_name
        self.metadata_dict = {}

    @staticmethod
    def create_output_folder_for_export(dataset_name: str, mr_classifier_name: str, explainer_method: str,
                                        label_type: str):
        output_folder = get_output_folder_for_export(dataset_name, mr_classifier_name, explainer_method,
                                                     label_type)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    @staticmethod
    def save_classifier(classifier: MinirocketClassifier, dataset_name: str = None):
        mr_classifier_name = classifier.classifier.__class__.__name__
        os.makedirs(f'data/{dataset_name}/', exist_ok=True)
        classifier.save(f'data/{dataset_name}/{mr_classifier_name}.pkl')

    def prepare_export(self, DATASET_FETCH_INFO: tuple):
        (_, features) = DATASET_FETCH_INFO
        for (f, _) in features:
            os.makedirs(f'{self.output_path}/' + f, exist_ok=True)
        pd.DataFrame(METADATA_SCHEMA).to_csv(f'{self.output_path}/metadata.csv', mode='w',
                                             index=False, header=True)

    def export_instance_and_explanations(self, instance_id, y_i,
                                         features: list,
                                         configuration: tuple,
                                         explanations_dict: dict):
        (dataset_name, mr_classifier_name, explainer_method, label_type) = configuration
        if configuration not in self.metadata_dict:
            self.metadata_dict[configuration] = METADATA_SCHEMA.copy()

        some_reference_policy = next(iter(explanations_dict))
        (explanation, _, _) = explanations_dict[some_reference_policy]
        instance = explanation.get_instance()
        for channel_idx, channel in enumerate(instance):
            for idx, reference_policy in enumerate(REFERENCE_POLICIES):
                (explanation, explanation_p2p, segmented_explanation) = explanations_dict[reference_policy]
                betas = explanation.get_attributions_in_original_dimensions()
                betas_p2p = explanation_p2p.get_attributions_in_original_dimensions()
                betas_segmented = segmented_explanation.get_distributed_explanations_in_original_space()
                reference = explanation.explanation['reference']
                reference_code = hash(reference[channel_idx].data.tobytes())
                reference_filename = f'{self.output_path}/{features[channel_idx][0]}/{features[channel_idx][0]}_reference_{reference_code}.csv'
                self.metadata_dict[configuration][f'reference_{idx}'] = reference_filename
                self.metadata_dict[configuration][f'reference_{idx}'] = REFERENCE_POLICIES_LABELS[reference_policy]
                if os.path.exists(reference_filename):
                    pd.Series(reference[channel_idx]).to_csv(reference_filename, header=False)
                self.metadata_dict[configuration][f'reference_{idx}_label'] = explanation.explanation['reference_prediction']
                self.metadata_dict[configuration][f'reference_{idx}_label_probability'] = explanation.explanation['reference_logit']
                attr_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/beta_instance_{instance_id}.csv'
                pd.Series(betas[channel_idx]).to_csv(attr_channel_filename, header=False)
                attr_p2p_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/betap2p_instance_{instance_id}.csv'
                pd.Series(betas_p2p[channel_idx]).to_csv(attr_p2p_channel_filename, header=False)
                attr_segmented_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/betasegmented_instance_{instance_id}.csv'
                pd.Series(betas_segmented[channel_idx]).to_csv(attr_segmented_channel_filename, header=False)

            channel_filename = f'{self.output_dataset_path}/{features[channel_idx][0]}/{features[channel_idx][0]}_instance_{instance_id}.csv'
            self.metadata_dict[configuration]['series'] = channel_filename
            if os.path.exists(channel_filename):
                pd.Series(channel).to_csv(channel_filename, header=False)

            self.metadata_dict[configuration]['label'] = y_i
            self.metadata_dict[configuration]['label_type'] = label_type
            self.metadata_dict[configuration]['predicted_class'] = explanation.explanation['instance_prediction']
            self.metadata_dict[configuration]['predicted_class_probability'] = explanation.explanation['instance_logit']

        flush_metadata(self.metadata_dict[configuration], self.output_path)

    def export_metametadata(self):
        metametadata = {}
        for i in range(len(UNITS[self.dataset_name])):
            metametadata['units'] = DESCRIPTIONS[self.dataset_name][i]
            metametadata['references'] = REFERENCE_POLICIES_LABELS
        with open(f'{self.output_dataset_path}/metametadata.json', 'w') as f:
            json.dump(metametadata, f)

def fetch_computed_attributions(root_folder: str, dataset_name: str, x: np.ndarray, type: str, reference_policy: str,
                                explainer_method: str):
    pass


def extract_dataset_name(dataset_name: str) -> str:
    return re.sub(r'-c\d+$', '',  dataset_name)


def flush_metadata(metadata_entries: dict, folder_path: str):
    pd.DataFrame(metadata_entries).to_csv(f'{folder_path}/metadata.csv', mode='w', index=False, header=False)

def get_output_folder_for_export(dataset_name: str, mr_classifier_name: str, explainer_method: str, label_type: str) -> str:
    return 'data/' + dataset_name + '/' + mr_classifier_name + '/' + explainer_method + '/' + label_type



