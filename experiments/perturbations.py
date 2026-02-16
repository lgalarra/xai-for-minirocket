#!/usr/bin/env python
# coding: utf-8
import copy
import itertools
# In[41]:
import os
import pickle
import joblib
import sys

import numpy as np
import pandas as pd

from pertutils import get_perturbations
from export_data import DataExporter
from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import importlib

import minirocket_multivariate_variable as mmv
from import_data import DataImporter
from reference import REFERENCE_POLICIES

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df, get_abnormal_hearbeat_for_classification)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score


def compute_difference(classifier, X_test, X_perturbed, X_reference, budget) -> (np.array, np.array, float):
    X_test_expanded = np.repeat(X_test, budget, axis=0)
    X_reference_expanded = np.repeat(X_reference, budget, axis=0)
    y = classifier.predict(X_test_expanded)
    y_pert = classifier.predict(X_perturbed)
    probs_before = classifier.predict_proba(X_test_expanded)
    probs_after = classifier.predict_proba(X_perturbed)
    probs_reference = classifier.predict_proba(X_reference_expanded)
    delta = probs_before[np.arange(len(y)), y] - probs_after[np.arange(len(y)), y]
    delta_instance_ref = (probs_before[np.arange(len(y)), y]  - probs_reference[np.arange(len(y)), y])
    delta_norm = delta / delta_instance_ref
    delta_bin = np.abs(y - y_pert)
    return delta, delta_norm, np.mean(delta_bin)


if __name__ == '__main__':
    THE_DATASET = None
    if len(sys.argv) > 1:
        THE_DATASET = sys.argv[1]

    THE_LABEL = None
    if len(sys.argv) > 2:
        THE_LABEL = sys.argv[2]

    THE_EXPLAINER = None
    if len(sys.argv) > 3:
        THE_EXPLAINER = sys.argv[3]

    THE_REFERENCE_POLICY = None
    if len(sys.argv) > 4:
        THE_REFERENCE_POLICY = sys.argv[4]

    THE_PERTURBATION = None
    if len(sys.argv) > 5:
        THE_PERTURBATION = sys.argv[5]

    THE_CLASSIFIER = None
    if len(sys.argv) > 6:
        THE_CLASSIFIER = sys.argv[6]

    DATASETS = ['starlight-c1', 'starlight-c2', 'starlight-c3', 'cognitive-circles', 'ford-a']
    if THE_DATASET is None:
        MR_CLASSIFIERS = {dataset: [
            [pickle.load(open(f"data/{dataset}/LogisticRegression.pkl", "rb")),
             pickle.load(open(f"data/{dataset}/RandomForestClassifier.pkl", "rb")),
             pickle.load(open(f"data/{dataset}/MLPClassifier.pkl", "rb"))
             ]
        ]
                          for dataset in DATASETS}
    else:
        MR_CLASSIFIERS = {
            THE_DATASET: [pickle.load(open(f"data/{THE_DATASET}/LogisticRegression.pkl", "rb")),
                          pickle.load(open(f"data/{THE_DATASET}/RandomForestClassifier.pkl", "rb")),
                          pickle.load(open(f"data/{THE_DATASET}/MLPClassifier.pkl", "rb"))
                        ]
        }

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": "get_forda_for_classification()",
        "starlight-c1": "get_starlightcurves_for_classification('1')",
        "starlight-c2": "get_starlightcurves_for_classification('2')",
        "starlight-c3": "get_starlightcurves_for_classification('3')",
		"abnormal-heartbeat-c1": "get_abnormal_hearbeat_for_classification('1')",
        "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        "handoutlines": "get_handoutlines_for_classification('1')"
    }
    EXPLAINERS = ['extreme_feature_coalitions', 'shap', 'gradients', 'stratoshap-k1']
    #EXPLAINERS = ['gradients']
    BUDGET = 10
    PERTURBATIONS = {
                    'instance_to_reference': {'percentile_cut': [50, 75, 90],
                     'interpolation': [0.25, 0.5, 0.75, 1.0], 'budget': [1]
                    },
                    'gaussian' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.5, 2.0, 1.5, 1.0],
                                  'budget': [BUDGET]
                    },
                     'reference_to_instance': {'percentile_cut': [90, 75, 50],
                                               'interpolation': [0.25, 0.5, 0.75, 1.0],
                                               'budget': [1]
                    },
                    'reference_to_instance_positive': {'percentile_cut': [90, 75, 50],
                                  'interpolation': [0.25, 0.5, 0.75, 1.0],
                                  'budget': [1]
                    }
    }

    # In[42]:
    OUTPUT_FILE = 'perturbation-results.csv'
    DataExporter.METADATA_FILE = 'metadata.csv'
    if THE_DATASET is not None:
        OUTPUT_FILE = f'perturbation-results-{THE_DATASET}.csv'
    if THE_LABEL is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_LABEL}.csv')
    if THE_EXPLAINER is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_EXPLAINER}.csv')
    if THE_REFERENCE_POLICY is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_REFERENCE_POLICY}.csv')
    if THE_PERTURBATION is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_PERTURBATION}.csv')
        PERTURBATIONS = {THE_PERTURBATION : PERTURBATIONS[THE_PERTURBATION]}
    if THE_CLASSIFIER is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_CLASSIFIER}.csv')


    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'f_minus_f0' : [], 'f_minus_f0-mean': [], 'f_minus_f0-std': [], 'f_minus_f0-change_ratio': [],
                 'f_minus_f0_norm': [], 'f_minus_f0_norm-mean': [], 'f_minus_f0_norm-std': [],
                 'p2p_f_minus_f0': [], 'p2p_f_minus_f0-mean': [], 'p2p_f_minus_f0-std': [], 'p2p_f_minus_f0-change_ratio': [],
                 'p2p_f_minus_f0_norm': [], 'p2p_f_minus_f0_norm-mean': [], 'p2p_f_minus_f0_norm-std': [],
                 'segmented_f_minus_f0': [], 'segmented_f_minus_f0-mean': [], 'segmented_f_minus_f0-std': [], 'segmented_f_minus_f0-change_ratio': [],
                 'segmented_f_minus_f0_norm': [], 'segmented_f_minus_f0_norm-mean': [], 'segmented_f_minus_f0_norm-std': [],
                 'args': [], 'perturbation_policy': [] }
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    for dataset_name in DATASET_FETCH_FUNCTIONS.keys() if THE_DATASET is None else [THE_DATASET]:
        dataset_fetch_function = DATASET_FETCH_FUNCTIONS[dataset_name]
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        data_importer = DataImporter(dataset_name)
        for classifier in MR_CLASSIFIERS[dataset_name]:
            classifier_name = classifier.classifier.__class__.__name__
            if THE_CLASSIFIER is not None and classifier_name != THE_CLASSIFIER:
                continue
            print('Classifier', classifier_name)
            for label in LABELS if THE_LABEL is None else [THE_LABEL]:
                for explainer_method in EXPLAINERS if THE_EXPLAINER is None else [THE_EXPLAINER]:
                    metadata_df = data_importer.get_metadata(classifier_name, explainer_method, label)
                    (X_test, y_test, references_dict, explanations_dict, p2p_explanations_dict,
                     segmented_explanations_dict) = (
                        DataImporter.get_series_from_metadata(metadata_df)
                    )
                    print('Label, explainer_method: ', label, explainer_method)
                    for perturbation_policy, all_args in PERTURBATIONS.items():
                        for combo in itertools.product(*all_args.values()):
                            args = dict(zip(all_args.keys(), combo))
                            print('Perturbation', perturbation_policy, 'Args: ', args)
                            for reference_policy in REFERENCE_POLICIES if THE_REFERENCE_POLICY is None else [THE_REFERENCE_POLICY]:
                                df_results = copy.deepcopy(df_schema)
                                X_reference = explanations_dict[reference_policy]
                                print('Backpropagated explanations')
                                args['y'] = y_test if label == 'training' else classifier.predict(X_test)

                                X_perturbed, n_perturbed_points = get_perturbations(X_test, references_dict[reference_policy],
                                                                explanations_dict[reference_policy],
                                                                explainer_method=explainer_method,
                                                                policy=perturbation_policy, **args)
                                X_p2p_perturbed = None
                                if p2p_explanations_dict[reference_policy][0][0] is not None:
                                    print('P2p explanations')
                                    X_p2p_perturbed, _ = get_perturbations(X_test, references_dict[reference_policy],
                                                                    p2p_explanations_dict[reference_policy],
                                                                        explainer_method=explainer_method,
                                                                        policy=perturbation_policy, **args)

                                print('Segmented explanations')
                                args['n_perturbed_points'] = n_perturbed_points
                                X_segmented_perturbed, _ = get_perturbations(X_test, references_dict[reference_policy],
                                                                segmented_explanations_dict[reference_policy],
                                                                          explainer_method=explainer_method,
                                                                          policy=perturbation_policy, **args)
                                del args['y']
                                del args['n_perturbed_points']
                                if perturbation_policy == 'reference_to_instance_positive':
                                    metric, norm_metric, change_ratio = compute_difference(classifier, X_reference, X_perturbed,
                                                                                           X_test,
                                                                                           PERTURBATIONS[perturbation_policy]['budget'][0])
                                else:
                                    metric, norm_metric, change_ratio = compute_difference(classifier, X_test, X_perturbed,
                                                                                           X_reference,
                                                                                           PERTURBATIONS[perturbation_policy]['budget'][0])
                                df_results['f_minus_f0'].append(to_sep_list(metric))
                                df_results['f_minus_f0-mean'].append(np.mean(metric))
                                df_results['f_minus_f0-std'].append(np.std(metric))
                                df_results['f_minus_f0_norm'].append(to_sep_list(norm_metric))
                                df_results['f_minus_f0_norm-mean'].append(np.mean(norm_metric))
                                df_results['f_minus_f0_norm-std'].append(np.std(norm_metric))
                                df_results['f_minus_f0-change_ratio'].append(change_ratio)

                                metric = norm_metric = change_ratio = [-1.0]
                                if X_p2p_perturbed is not None:
                                    if perturbation_policy == 'reference_to_instance_positive':
                                        metric, norm_metric, change_ratio = compute_difference(classifier, X_reference, X_p2p_perturbed, X_test, PERTURBATIONS[perturbation_policy]['budget'][0])
                                    else:
                                        metric, norm_metric, change_ratio = compute_difference(classifier, X_test, X_p2p_perturbed, X_reference, PERTURBATIONS[perturbation_policy]['budget'][0])
                                df_results['p2p_f_minus_f0'].append(to_sep_list(metric))
                                df_results['p2p_f_minus_f0-mean'].append(np.mean(metric))
                                df_results['p2p_f_minus_f0-std'].append(np.std(metric))
                                df_results['p2p_f_minus_f0_norm'].append(to_sep_list(norm_metric))
                                df_results['p2p_f_minus_f0_norm-mean'].append(np.mean(norm_metric))
                                df_results['p2p_f_minus_f0_norm-std'].append(np.std(norm_metric))
                                df_results['p2p_f_minus_f0-change_ratio'].append(change_ratio)

                                if perturbation_policy == 'reference_to_instance_positive':
                                    metric, norm_metric, change_ratio = compute_difference(classifier, X_reference, X_segmented_perturbed, X_test, PERTURBATIONS[perturbation_policy]['budget'][0])

                                else:
                                    metric, norm_metric, change_ratio = compute_difference(classifier, X_test, X_segmented_perturbed, X_reference, PERTURBATIONS[perturbation_policy]['budget'][0])
                                df_results['segmented_f_minus_f0'].append(to_sep_list(metric))
                                df_results['segmented_f_minus_f0-mean'].append(np.mean(metric))
                                df_results['segmented_f_minus_f0-std'].append(np.std(metric))
                                df_results['segmented_f_minus_f0_norm'].append(to_sep_list(norm_metric))
                                df_results['segmented_f_minus_f0_norm-mean'].append(np.mean(norm_metric))
                                df_results['segmented_f_minus_f0_norm-std'].append(np.std(norm_metric))
                                df_results['segmented_f_minus_f0-change_ratio'].append(change_ratio)


                                df_results['timestamp'].append(pd.Timestamp.now())
                                df_results['base_explainer'].append(explainer_method)
                                df_results['mr_classifier'].append(classifier_name)
                                df_results['reference_policy'].append(reference_policy)
                                df_results['label'].append(label)
                                df_results['dataset'].append(dataset_name)
                                df_results['args'].append(f'{args}')
                                df_results['perturbation_policy'].append(perturbation_policy)
                                pd.DataFrame(df_results).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
