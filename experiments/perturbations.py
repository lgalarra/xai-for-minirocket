#!/usr/bin/env python
# coding: utf-8
import copy
import itertools
# In[41]:
import os
import pickle
import sys

import numpy as np
import pandas
import pandas as pd

from experiments.exputils import to_sep_list
from experiments.pertutils import get_perturbations
from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import importlib

import minirocket_multivariate_variable as mmv
from import_data import DataImporter

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score


def compute_difference(classifier, X_test, X_perturbed, X_reference, budget) -> (np.array, np.array, float):
    X_test_expanded = np.repeat(X_test, budget, axis=0)
    X_reference_expanded = np.repeat(X_reference, budget, axis=0)
    y = classifier.predict(X_test_expanded)
    probs_before = classifier.predict_proba(X_test_expanded)
    probs_after = classifier.predict_proba(X_perturbed)
    probs_reference = classifier.predict_proba(X_reference_expanded)
    delta = probs_before[np.arange(len(y)), y] - probs_after[np.arange(len(y)), y]
    delta_norm = delta / (probs_before[np.arange(len(y)), y]  - probs_reference[np.arange(len(y)), y])
    delta_bin = np.abs(classifier.predict(X_test_expanded) - classifier.predict(X_perturbed))
    return delta, delta_norm, np.mean(delta_bin)


if __name__ == '__main__':
    MR_CLASSIFIERS = {
        "ford-a": [pickle.load(open("data/ford-a/LogisticRegression.pkl", "rb"))]
    }
    ## We will restrict to one or two
    REFERENCE_POLICIES = ['opposite_class_medoid', 'opposite_class_centroid',
                          'global_medoid', 'global_centroid', 'opposite_class_farthest_instance',
                          'opposite_class_closest_instance']

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": "get_forda_for_classification()",
        "startlight-c1": "get_starlightcurves_for_classification('1')",
        "startlight-c2": "get_starlightcurves_for_classification('2')",
        "startlight-c3": "get_starlightcurves_for_classification('3')",
        "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
    }
    EXPLAINERS = ['shap', 'extreme_feature_coalitions', 'stratoshap-k1']
    BUDGET = 100
    PERTURBATIONS = {
                    'instance_to_reference': {'percentile_cut': [90, 75, 50],
                                  'interpolation': [0.5, 1.0],
                                  'noise_level': [0.2, 0.4, 0.6, 0.8, 1.0], 'budget': [1]
                    },
                    'gaussian' : {'percentile_cut': [90, 75, 50],
                                   'sigma' : [0.05, 0.1, 0.2],
                                  'budget': [BUDGET]
                    },
                     'reference_to_instance': {'percentile_cut': [90, 75, 50],
                                               'interpolation': [0.5, 1.0],
                                               'noise_level': [0.2, 0.4, 0.6, 0.8, 1.0],
                                               'budget': [1]
                    }
    }

    # In[42]:
    OUTPUT_FILE = 'perturbation-results.csv'


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

    for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        data_importer = DataImporter(dataset_name)
        for classifier in MR_CLASSIFIERS[dataset_name]:
            classifier_name = classifier.classifier.__class__.__name__
            for label in LABELS:
                for explainer_method in EXPLAINERS:
                    metadata_df = data_importer.get_metadata(classifier_name, explainer_method, label)
                    X_test, y_test, references_dict, explanations_dict, p2p_explanations_dict, segmented_explanations_dict = (
                        DataImporter.get_series_from_metadata(metadata_df)
                    )
                    for perturbation_policy, all_args in PERTURBATIONS.items():
                        for combo in itertools.product(*all_args.values()):
                            args = dict(zip(all_args.keys(), combo))
                            for reference_policy in REFERENCE_POLICIES:
                                df_results = copy.deepcopy(df_schema)
                                X_reference = explanations_dict[reference_policy]
                                X_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                X_reference, policy=perturbation_policy, **args)

                                X_p2p_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                X_reference, policy=perturbation_policy, **args)

                                X_segmented_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                X_reference, policy=perturbation_policy, **args)

                                metric, norm_metric, change_ratio = compute_difference(classifier, X_test, X_perturbed, X_reference, PERTURBATIONS[perturbation_policy]['budget'][0])
                                df_results['f_minus_f0'].append(to_sep_list(metric))
                                df_results['f_minus_f0-mean'].append(np.mean(metric))
                                df_results['f_minus_f0-std'].append(np.std(metric))
                                df_results['f_minus_f0_norm'].append(to_sep_list(norm_metric))
                                df_results['f_minus_f0_norm-mean'].append(np.mean(norm_metric))
                                df_results['f_minus_f0_norm-std'].append(np.std(norm_metric))
                                df_results['f_minus_f0-change_ratio'].append(change_ratio)

                                metric, norm_metric, change_ratio = compute_difference(classifier, X_test, X_p2p_perturbed, X_reference, PERTURBATIONS[perturbation_policy]['budget'][0])
                                df_results['p2p_f_minus_f0'].append(to_sep_list(metric))
                                df_results['p2p_f_minus_f0-mean'].append(np.mean(metric))
                                df_results['p2p_f_minus_f0-std'].append(np.std(metric))
                                df_results['p2p_f_minus_f0_norm'].append(to_sep_list(norm_metric))
                                df_results['p2p_f_minus_f0_norm-mean'].append(np.mean(norm_metric))
                                df_results['p2p_f_minus_f0_norm-std'].append(np.std(norm_metric))
                                df_results['p2p_f_minus_f0-change_ratio'].append(change_ratio)

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
