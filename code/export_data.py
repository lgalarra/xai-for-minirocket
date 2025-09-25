import os
import re

import numpy as np
import pandas as pd
from skbase.base import BaseEstimator
from reference import REFERENCE_POLICIES

from explainer import Explanation

METADATA_COLUMNS =(['series', 'label', 'label_type', 'predicted_class', 'predicted_class_probability', 'channel', 'group', 'annotation',
                    'global_medoid_id'] + [f'reference_{i}' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_predicted_class' for i in range(len(REFERENCE_POLICIES))]
                   + [f'reference_{i}_predicted_class_probability' for i in range(len(REFERENCE_POLICIES))]
                   + ['beta_attributions'] )


METADATA_SCHEMA = {col : [] for col in METADATA_COLUMNS}

def fetch_computed_attributions(root_folder: str, dataset_name: str, x: np.ndarray, type: str, reference_policy: str,
                                explainer_method: str):
    pass


def extract_dataset_name(dataset_name: str) -> str:
    return re.sub(r'-c\d+$', '',  dataset_name)

def prepare_output_folder_for_export(DATASET_FETCH_FUNCTIONS: dict):
    for dataset_name, (_, features) in DATASET_FETCH_FUNCTIONS.items():
        extract_dataset_name(dataset_name)
        os.makedirs('data/' + dataset_name, exist_ok=True)
        for (f, _) in features:
            os.makedirs('data/' + dataset_name + '/' + f, exist_ok=True)
        pd.DataFrame(METADATA_SCHEMA).to_csv('data/metadata.csv', mode='w', index=False, header=True)


def update_metadata(metadata_entry: dict):
    pd.DataFrame(metadata_entry).to_csv('data/metadata.csv', mode='a', index=False, header=False)


def export_instance_and_explanations(i, y_i, dataset_name,
                                     features: list,
                                     explanation: Explanation,
                                     explanation_p2p: Explanation,
                                     segmented_explanation: Explanation,
                                     explainer_method: str,
                                     mr_classifier: BaseEstimator,
                                     label_type: str = 'training'):
    instance = explanation.get_instance()
    betas = explanation.get_attributions_in_original_dimensions()
    betas_p2p = explanation_p2p.get_attributions_in_original_dimensions()
    betas_segmented = segmented_explanation.get_distributed_explanations_in_original_space()
    metadata = METADATA_SCHEMA.copy()
    for channel_idx, channel in enumerate(instance):
        channel_filename = f'data/{dataset_name}/{features[channel_idx][0]}/{features[channel_idx][0]}_instance_{i}.csv'
        metadata['series'] = channel_filename
        pd.Series(channel).to_csv(channel_filename, header=False)
        metadata['label'] = y_i
        metadata['label_type'] = label_type
        metadata['predicted_class'] = explanation.explanation['instance_prediction']
        metadata['predicted_class_probability'] = explanation.explanation['instance_logit']
        attr_channel_filename = f'data/{dataset_name}/{features[channel_idx][0]}/beta_{mr_classifier.__class__.__name__}_{explainer_method}_{features[channel_idx][0]}_instance_{i}.csv'
        pd.Series(betas[channel_idx]).to_csv(attr_channel_filename, header=False)
        attr_p2p_channel_filename = f'data/{dataset_name}/{features[channel_idx][0]}/betap2p_{mr_classifier.__class__.__name__}_{explainer_method}_{features[channel_idx][0]}_instance_{i}.csv'
        pd.Series(betas_p2p[channel_idx]).to_csv(attr_p2p_channel_filename, header=False)
        attr_segmented_channel_filename = f'data/{dataset_name}/{features[channel_idx][0]}/betasegmented_{mr_classifier.__class__.__name__}_{explainer_method}_{features[channel_idx][0]}_instance_{i}.csv'
        pd.Series(betas_segmented[channel_idx]).to_csv(attr_segmented_channel_filename, header=False)
        update_metadata(metadata_entry=metadata)