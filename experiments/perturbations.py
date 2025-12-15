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
import pandas
import pandas as pd

from experiments.exputils import to_sep_list
from experiments.pertutils import get_perturbations
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
                   cognitive_circles_get_sorted_channels_from_df)
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
    MR_CLASSIFIERS = {
#        "starlight-c1": [pickle.load(open("data/starlight-c1/LogisticRegression.pkl", "rb")),
#                          pickle.load(open("data/starlight-c1/RandomForestClassifier.pkl", "rb"))
#                          ],
#        "starlight-c2": [pickle.load(open("data/starlight-c2/LogisticRegression.pkl", "rb")),
#                          pickle.load(open("data/starlight-c2/RandomForestClassifier.pkl", "rb"))
#                          ],
        #"starlight-c3": [pickle.load(open("data/starlight-c3/LogisticRegression.pkl", "rb")),
        #                  pickle.load(open("data/starlight-c3/RandomForestClassifier.pkl", "rb"))
        #                  ],
#        "cognitive-circles": [pickle.load(open("data/cognitive-circles/LogisticRegression.pkl", "rb")),
#                          pickle.load(open("data/cognitive-circles/RandomForestClassifier.pkl", "rb"))
#                          ],
        "ford-a": [
            pickle.load(open("data/ford-a/LogisticRegression.pkl", "rb")),
            #       pickle.load(open("data/ford-a/RandomForestClassifier.pkl", "rb"))
        ]
    }

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": "get_forda_for_classification()",
        #"starlight-c1": "get_starlightcurves_for_classification('1')",
        #"starlight-c2": "get_starlightcurves_for_classification('2')",
        #"starlight-c3": "get_starlightcurves_for_classification('3')",
        #"cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
    }
    #EXPLAINERS = ['extreme_feature_coalitions', 'shap', 'gradients', 'stratoshap-k1']
    EXPLAINERS = ['gradients']
    BUDGET = 10
    PERTURBATIONS = {
#                    'instance_to_reference': {'percentile_cut': [50, 75, 90],
#                     'interpolation': [0.25, 0.5, 0.75, 1.0], 'budget': [1]
#                    },
                    'gaussian' : {'percentile_cut': [50, 75, 90],
                                    'sigma' : [1.0, 0.75, 0.5, 0.25],
                                  'budget': [BUDGET]
                    },
#                     'reference_to_instance': {'percentile_cut': [90, 75, 50],
#                                               'interpolation': [0.25, 0.5, 0.75, 1.0],
#                                               'budget': [1]
#                    }
    }

    # In[42]:
    OUTPUT_FILE = 'perturbation-results.csv'
    DataExporter.METADATA_FILE = 'metadata-fixed.csv'


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
            print('Classifier', classifier_name)
            for label in LABELS:
                for explainer_method in EXPLAINERS:
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
                            for reference_policy in REFERENCE_POLICIES:
                                df_results = copy.deepcopy(df_schema)
                                X_reference = explanations_dict[reference_policy]
                                print('Backpropagated explanations')
                                X_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                explanations_dict[reference_policy],
                                                                explainer_method=explainer_method,
                                                                policy=perturbation_policy, **args)
                                print('P2p explanations')
                                args['p2p'] = True
                                args['y'] = y_test if label == 'training' else classifier.predict(X_test)
                                X_p2p_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                p2p_explanations_dict[reference_policy],
                                                                    explainer_method=explainer_method,
                                                                    policy=perturbation_policy, **args)
                                del args['p2p']
                                del args['y']
                                print('Segmented explanations')
                                X_segmented_perturbed = get_perturbations(X_test, references_dict[reference_policy],
                                                                segmented_explanations_dict[reference_policy],
                                                                          explainer_method=explainer_method,
                                                                          policy=perturbation_policy, **args)

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
