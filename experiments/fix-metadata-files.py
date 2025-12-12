## Similar to the script that carries out the perturbations but it should now iterate over the test instances
## find their reference and fill the schema
import copy
import glob
import hashlib
import os
import pickle
import sys

import pandas as pd
from numba.np.arrayobj import populate_array

from export_data import METADATA_COLUMNS, flush_metadata, get_group_id, get_annotation
from import_data import DataImporter
from reference import REFERENCE_POLICIES
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)

MR_CLASSIFIERS = {
        "starlight-c1": [pickle.load(open("data/starlight-c1/LogisticRegression.pkl", "rb")),
                          pickle.load(open("data/starlight-c1/RandomForestClassifier.pkl", "rb"))
                          ],
        "starlight-c2": [pickle.load(open("data/starlight-c2/LogisticRegression.pkl", "rb")),
                          pickle.load(open("data/starlight-c2/RandomForestClassifier.pkl", "rb"))
                          ],
        "starlight-c3": [pickle.load(open("data/starlight-c3/LogisticRegression.pkl", "rb")),
                          pickle.load(open("data/starlight-c3/RandomForestClassifier.pkl", "rb"))
                        ],
#        "cognitive-circles": [pickle.load(open("data/cognitive-circles/LogisticRegression.pkl", "rb")),
#                          pickle.load(open("data/cognitive-circles/RandomForestClassifier.pkl", "rb"))
#                          ],
    "ford-a": [
            pickle.load(open("data/ford-a/LogisticRegression.pkl", "rb")),
            pickle.load(open("data/ford-a/RandomForestClassifier.pkl", "rb"))
    ]
}

LABELS = ['training', 'predicted']
DATASET_FETCH_FUNCTIONS = {
    "ford-a": "get_forda_for_classification()",
    "starlight-c1": "get_starlightcurves_for_classification('1')",
    "starlight-c2": "get_starlightcurves_for_classification('2')",
    "starlight-c3": "get_starlightcurves_for_classification('3')",
    #"cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
}
EXPLAINERS = ['shap', 'gradients', 'extreme_feature_coalitions', 'stratoshap-k1']


def fix_metadata_for_channel(channel_path):
    glob.glob(channel_path + 'beta_instance_')

METADATA_FILE = 'metadata-fixed.csv'
METADATA_SCHEMA = {col : [] for col in METADATA_COLUMNS}


def populate_metadata_entry(metadata_dict, x, y_label, y_i, y_pred, reference_policy_idx, channel_id, channel,
                            exp_path, exp_p2p_path, exp_segmented_path,
                            explainer, classifier) -> dict:
    reference = explainer.get_reference(x, y_label, REFERENCE_POLICIES[reference_policy_idx])
    reference_code = hashlib.md5(x[channel_id].data.tobytes()).hexdigest()
    reference_filename = f'{os.path.dirname(exp_path)}/{channel}_reference_{reference_code}.csv'
    if not os.path.exists(reference_filename):
        print(f'File {reference_filename} for {exp_path} does not exists', file=sys.stderr)
        print('Regenerating the file')
        pd.Series(reference[channel_id]).to_csv(reference_filename, header=False)

    metadata_dict[f'reference_{reference_policy_idx}'].append(reference_filename)
    metadata_dict[f'reference_{reference_policy_idx}_label'].append(classifier.predict(reference.reshape(-1, 1))[0])
    metadata_dict[f'reference_{reference_policy_idx}_label_probability'].append(classifier.predict_proba(reference.reshape(-1, 1))[0][y_pred])
    metadata_dict[f'beta_{reference_policy_idx}_attributions'].append(exp_path)
    metadata_dict[f'beta_p2p_{reference_policy_idx}_attributions'].append(exp_p2p_path)
    metadata_dict[f'beta_segmented_{reference_policy_idx}_attributions'].append(exp_segmented_path)


def fix_metadata(path, channels, X_test, y_test, X_train, y_train, classifier):
    channels_dict = {}
    pd.DataFrame(METADATA_SCHEMA).to_csv(f'{path}/{METADATA_FILE}', mode='w',
                                         index=False, header=True)
    for channel in channels:
        channel_path = os.path.join(path, channel)
        channels_dict[channel] = channel_path

    dataset = os.path.normpath(path).split(os.sep)[1]
    explainer = classifier.get_explainer(X=X_train, y=y_train)
    for instance_id, x in enumerate(X_test):
        for channel_id, channel in enumerate(channels):
            metadata_dict = copy.deepcopy(METADATA_SCHEMA)
            n_ref = 0
            y_i = y_test[instance_id]
            y_pred = classifier.predict(x.reshape(-1, 1))[0]
            y_label = y_i if 'training' in path else y_pred
            for reference_policy_idx in range(len(REFERENCE_POLICIES)):
                exp_path = channels_dict[channel] + f'/beta_instance_{instance_id}_{reference_policy_idx}.csv'
                exp_p2p_path = channels_dict[channel] + f'/betap2p_instance_{instance_id}_{reference_policy_idx}.csv'
                exp_segmented_path = channels_dict[channel] + f'/betasegmented_instance_{instance_id}_{reference_policy_idx}.csv'
                if os.path.exists(exp_path) and os.path.exists(exp_p2p_path) and os.path.exists(exp_segmented_path):
                    populate_metadata_entry(metadata_dict, x, y_label, y_i, y_pred, reference_policy_idx, channel_id, channel,
                                            exp_path, exp_p2p_path, exp_segmented_path,
                                            explainer, classifier)
                    n_ref += 1
            if n_ref < len(REFERENCE_POLICIES):
                print(f'Problem with instance {instance_id} and channel {channel}, {path}')
                continue

            metadata_dict['instance_id'].append(instance_id)
            metadata_dict['label_type'].append(os.path.basename(path))
            metadata_dict['channel'].append(channel)
            metadata_dict['group'].append(get_group_id(os.path.split(path)[1], instance_id))
            metadata_dict['annotation'].append(get_annotation(os.path.split(path)[1], instance_id))
            metadata_dict['label'].append(y_label)
            metadata_dict['label_probability'].append(classifier.predict_proba(x.reshape(-1, 1))[0][y_pred])
            channel_filename = f'data/{dataset}/{channel}/{channel}_instance_{instance_id}.csv'
            metadata_dict['series'].append(channel_filename)


            flush_metadata(metadata_dict, f'{path}/{METADATA_FILE}')



for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
    (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
    data_importer = DataImporter(dataset_name)
    for classifier in MR_CLASSIFIERS[dataset_name]:
        classifier_name = classifier.classifier.__class__.__name__
        print('Classifier', classifier_name)
        for label in LABELS:
            for explainer_method in EXPLAINERS:
                path = data_importer.get_attributions_path(classifier_name, explainer_method, label)
                if os.path.exists(path):
                    channels = [x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
                    print(channels)
                    fix_metadata(path, channels, X_test, y_test, X_train, y_train, classifier)
#                metadata_df = data_importer.get_metadata(classifier_name, explainer_method, label)
#                X_test, y_test, references_dict, explanations_dict, p2p_explanations_dict, segmented_explanations_dict = (
#                    DataImporter.get_series_from_metadata(metadata_df)
#                )
#                print('Label, explainer_method: ', label, explainer_method)
