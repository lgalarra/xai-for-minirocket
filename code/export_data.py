import copy
import hashlib
import os
import re

import numpy as np
import pandas as pd

from classifier import MinirocketClassifier
from reference import REFERENCE_POLICIES, REFERENCE_POLICIES_LABELS
import json

from utils import COGNITIVE_CIRCLES_CHANNELS, cognitive_circles_get_sorted_channels_from_df, COGNITIVE_CIRCLES_UNITS

METADATA_COLUMNS =(['instance_id', 'series', 'label', 'label_type', 'label_probability', 'channel', 'group', 'annotation']
                   + [f'reference_{i}' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_label' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_label_probability' for i in range(len(REFERENCE_POLICIES))]
                   + [f'beta_{i}_attributions' for i in range(len(REFERENCE_POLICIES))]
                   + [f'beta_p2p_{i}_attributions' for i in range(len(REFERENCE_POLICIES))]
                   + [f'beta_segmented_{i}_attributions' for i in range(len(REFERENCE_POLICIES))]
                   )

UNITS = {
    "ford-a": ["dB"],
    "cognitive-circles": [COGNITIVE_CIRCLES_UNITS[x] for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')],
    "starlight-c1": ["m"], "starlight-c2": ["m"], "starlight-c3": ["m"],
    "abnormal-heartbeat-c0": ["m"], "abnormal-heartbeat-c1": ["m"],
    "abnormal-heartbeat-c2": ["m"], "abnormal-heartbeat-c3": ["m"],
    "abnormal-heartbeat-c4": ["m"],
}

DESCRIPTIONS = {
    "ford-a": ["Noise intensity"],
    "cognitive-circles": [COGNITIVE_CIRCLES_CHANNELS[x] for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')],
    "starlight-c1": ["Amplitude Change"], "starlight-c2": ["Amplitude Change"], "starlight-c3": ["Amplitude Change"],
    "abnormal-heartbeat-c0": ["Amplitude Change"],
    "abnormal-heartbeat-c1": ["Amplitude Change"],
    "abnormal-heartbeat-c2": ["Amplitude Change"],
    "abnormal-heartbeat-c3": ["Amplitude Change"],
    "abnormal-heartbeat-c4": ["Amplitude Change"]
}

CHANNELS = {'ford-a': ['C'],
            'cognitive-circles': [x for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')],
            'starlight-c1': ['B'], 'starlight-c2': ['B'], 'starlight-c3': ['B'],
            'abnormal-heartbeat-c0': ['A'], 'abnormal-heartbeat-c1': ['A'], 'abnormal-heartbeat-c2': ['A'],
            'abnormal-heartbeat-c3': ['A'], 'abnormal-heartbeat-c4': ['A']
            }

#('X', 'X'), ('V', 'velocity'), ('VA', 'angular_velocity'),
#                           ('DR', 'radial_velocity'), ('Y', 'Y'), ('D', 'radius'),  ('A', 'acceleration')
CLASSES = {'ford-a': ['No problem', 'Problem'], 'cognitive-circles': ['Easy', 'Difficult'],
           'starlight-c1': ['Star Type 1', 'Other'], 'starlight-c2': ['Star Type 2', 'Other'],
           'starlight-c3': ['Star Type 3', 'Other'],
           'abnormal-heartbeat-c0': ['Artifact', 'Other'],
           'abnormal-heartbeat-c1': ['Extra Stole', 'Other'],
           'abnormal-heartbeat-c2': ['Murmur', 'Other'],
           'abnormal-heartbeat-c3': ['Normal', 'Other'],
           'abnormal-heartbeat-c4': ['ExtraHLS', 'Other']
           }

METADATA_SCHEMA = {col : [] for col in METADATA_COLUMNS}


def get_group_id(dataset_name, instance_id) -> int:
    if dataset_name == "ford-a":
        return -1
    return None

def get_annotation(dataset_name, instance_id) -> str:
    if dataset_name == "ford-a":
        return f"{instance_id}"
    return None

class DataExporter(object):

    METADATA_FILE = 'metadata.csv'
    def __init__(self, dataset_name: str, mr_classifier_name: str, explainer_method: str, label_type: str):
        self.output_path = DataExporter.create_output_folder_for_export(dataset_name,
                                                                        mr_classifier_name, explainer_method,
                                                                        label_type)
        self.metadata_file = f'{self.output_path}/{DataExporter.METADATA_FILE}'
        self.output_dataset_path = 'data/' + dataset_name
        self.dataset_name = dataset_name
        self.mr_classifier_name = mr_classifier_name
        self.explainer_method = explainer_method
        self.label_type = label_type

    @staticmethod
    def create_output_folder_for_export(dataset_name: str, mr_classifier_name: str, explainer_method: str,
                                        label_type: str):
        output_folder = get_output_folder_for_export(dataset_name, mr_classifier_name, explainer_method,
                                                     label_type)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    @staticmethod
    def get_classifier_path(mr_classifier_name: str, dataset_name: str = None):
        os.makedirs(f'data/{dataset_name}/', exist_ok=True)
        return f'data/{dataset_name}/{mr_classifier_name}.pkl'

    @staticmethod
    def save_classifier(classifier: MinirocketClassifier, dataset_name: str = None):
        mr_classifier_name = classifier.classifier.__class__.__name__
        os.makedirs(f'data/{dataset_name}/', exist_ok=True)
        classifier.save(f'data/{dataset_name}/{mr_classifier_name}.pkl')

    def prepare_export(self, DATASET_FETCH_INFO: tuple):
        (_, features) = DATASET_FETCH_INFO
        for (f, _) in features:
            os.makedirs(f'{self.output_path}/' + f, exist_ok=True)
            os.makedirs(f'{self.output_dataset_path}/' + f, exist_ok=True)

        pd.DataFrame(METADATA_SCHEMA).to_csv(f'{self.metadata_file}', mode='w',
                                             index=False, header=True)

    def export_instance_and_explanations(self, instance_id, y_i,
                                         features: list,
                                         explanations_dict: dict,
                                         studied_reference_policies=REFERENCE_POLICIES,
                                         topk=None):
        (dataset_name, mr_classifier_name, explainer_method, label_type) = \
            (self.dataset_name, self.mr_classifier_name, self.explainer_method, self.label_type)
        metadata_dict = copy.deepcopy(METADATA_SCHEMA)
        some_reference_policy = next(iter(explanations_dict))
        (explanation, _, _) = explanations_dict[some_reference_policy]
        instance = explanation.get_instance()
        for channel_idx, channel in enumerate(instance):
            metadata_dict['instance_id'].append(instance_id)
            for idx, reference_policy in enumerate(studied_reference_policies):
                (explanation, explanation_p2p, segmented_explanation) = explanations_dict[reference_policy]
                betas = explanation.get_attributions_in_original_dimensions()
                betas_p2p = None
                betas_segmented = None
                if explanation_p2p is not None:
                    betas_p2p = explanation_p2p.get_attributions_in_original_dimensions()
                if segmented_explanation is not None:
                    betas_segmented = segmented_explanation.get_distributed_explanations_in_original_space()
                reference = explanation.explanation['reference']
                reference_code = hashlib.md5(reference[channel_idx].data.tobytes()).hexdigest()
                reference_filename = f'{self.output_path}/{features[channel_idx][0]}/{features[channel_idx][0]}_reference_{reference_code}.csv'
                metadata_dict[f'reference_{idx}'].append(reference_filename)

                if not os.path.exists(reference_filename):
                    pd.Series(reference[channel_idx]).to_csv(reference_filename, header=False)

                metadata_dict[f'reference_{idx}_label'].append(explanation.explanation['reference_prediction'])
                metadata_dict[f'reference_{idx}_label_probability'].append(explanation.explanation['reference_logit'])

                if topk is None:
                    attr_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/beta_instance_{instance_id}_{idx}.csv'
                else:
                    attr_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/beta_instance_{instance_id}_{idx}-topk-{topk}.csv'

                pd.Series(betas[channel_idx]).to_csv(attr_channel_filename, header=False)
                metadata_dict[f'beta_{idx}_attributions'].append(attr_channel_filename)

                if betas_p2p is not None:
                    attr_p2p_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/betap2p_instance_{instance_id}_{idx}.csv'
                    pd.Series(betas_p2p[channel_idx]).to_csv(attr_p2p_channel_filename, header=False)
                else:
                    attr_p2p_channel_filename = None
                metadata_dict[f'beta_p2p_{idx}_attributions'].append(attr_p2p_channel_filename)

                if betas_segmented is not None:
                    attr_segmented_channel_filename = f'{self.output_path}/{features[channel_idx][0]}/betasegmented_instance_{instance_id}_{idx}.csv'
                    pd.Series(betas_segmented[channel_idx]).to_csv(attr_segmented_channel_filename, header=False)
                else:
                    attr_segmented_channel_filename = None
                metadata_dict[f'beta_segmented_{idx}_attributions'].append(attr_segmented_channel_filename)

            for idx, _ in enumerate(REFERENCE_POLICIES):
                if len(metadata_dict[f'reference_{idx}']) == 0:
                    metadata_dict[f'reference_{idx}'].append(None)
                    metadata_dict[f'reference_{idx}_label'].append(None)
                    metadata_dict[f'reference_{idx}_label_probability'].append(None)
                    metadata_dict[f'beta_{idx}_attributions'].append(None)
                    metadata_dict[f'beta_p2p_{idx}_attributions'].append(None)
                    metadata_dict[f'beta_segmented_{idx}_attributions'].append(None)

            channel_filename = f'{self.output_dataset_path}/{features[channel_idx][0]}/{features[channel_idx][0]}_instance_{instance_id}.csv'
            metadata_dict['series'].append(channel_filename)
            if not os.path.exists(channel_filename):
                pd.Series(channel).to_csv(channel_filename, header=False)

            metadata_dict['label'].append(y_i)
            metadata_dict['label_type'].append(label_type)
            metadata_dict['label_probability'].append(explanation.explanation['instance_logit'])
            metadata_dict['channel'].append(CHANNELS[dataset_name][channel_idx])
            metadata_dict['group'].append(get_group_id(dataset_name, instance_id))
            metadata_dict['annotation'].append(get_annotation(dataset_name, instance_id))

        alphas = explanation.explanation['minirocket_coefficients']
        alphas_file = f'{self.output_dataset_path}/{mr_classifier_name}/alphas_instance_{instance_id}.csv'
        if not os.path.exists(alphas_file):
            pd.Series(alphas).to_csv(alphas_file, header=False)

        print(f'Flushing {metadata_dict} in {self.metadata_file}')
        flush_metadata(metadata_dict, self.metadata_file)

    def export_metametadata(self):
        metametadata = {}
        for i in range(len(UNITS[self.dataset_name])):
            metametadata['units'] = DESCRIPTIONS[self.dataset_name][i]
            metametadata['references'] = REFERENCE_POLICIES_LABELS
            metametadata['classes'] = CLASSES[self.dataset_name]
        with open(f'{self.output_dataset_path}/metametadata.json', 'w') as f:
            json.dump(metametadata, f)


def extract_dataset_name(dataset_name: str) -> str:
    return re.sub(r'-c\d+$', '',  dataset_name)


def flush_metadata(metadata_entries: dict, output_path: str):
    pd.DataFrame(metadata_entries).to_csv(output_path, mode='a', index=False, header=False)

def get_output_folder_for_export(dataset_name: str, mr_classifier_name: str, explainer_method: str, label_type: str) -> str:
    return 'data/' + dataset_name + '/' + mr_classifier_name + '/' + explainer_method + '/' + label_type



