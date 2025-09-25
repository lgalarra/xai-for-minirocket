#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
import pickle
import sys

import numpy as np
import pandas
import pandas as pd

from experiments.pertutils import get_perturbations
from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import importlib

import minirocket_multivariate_variable as mmv

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score
from export_data import export_instance_and_explanations, prepare_output_folder_for_export, fetch_computed_attributions

if __name__ == '__main__':
    MR_CLASSIFIERS = {
        "startlight-c1": [pickle.load("data/startlight-c1_LogisticRegression.pkl")]
    }
    ## We will restrict to one or two
    REFERENCE_POLICIES = ['opposite_class_medoid', 'opposite_class_centroid',
                          'global_medoid', 'global_centroid', 'opposite_class_farthest_instance',
                          'opposite_class_closest_instance']

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "startlight-c1": "get_starlightcurves_for_classification('1')",
        "startlight-c2": "get_starlightcurves_for_classification('2')",
        "startlight-c3": "get_starlightcurves_for_classification('3')",
        "ford-a": "get_forda_for_classification()",
        "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
    }
    EXPLAINERS = ['extreme_feature_coalitions', 'stratoshap-k1', 'shap']
    BUDGET = 100
    PERTURBATIONS = {'gaussian' : {'percentile_cut': [10, 25, 50],
                                   'noise_ratio' : [0.05, 0.1, 0.2]
                                   },
                     'instance_to_reference': {'percentile_cut': [10, 25, 50],
                                               'interpolation': [0.5, 1.0]
                                              },
                     'reference_to_instance': {'percentile_cut': [10, 25, 50],
                                               'interpolation': [0.5, 1.0]
                                               }
                     }

    # In[42]:
    OUTPUT_FILE = 'perturbation-results.csv'


    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'classifier_accuracy': [], 'r2s': [], 'r2s-mean': [], 'r2s-std': [],
                 'local_accuracy': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
                 'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
                 'runtimes-segmented-seconds': [], 'runtimes-segmented-mean': [], 'runtimes-segmented-std': [],
                 'complexity': [], 'complexity-mean': [], 'complexity-std': [],
                 'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': []}
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        for mr_classifier in MR_CLASSIFIERS[dataset_name]:
            for reference_policy in REFERENCE_POLICIES:
                for label in LABELS:
                    for explainer_method in EXPLAINERS:
                        for idx, x in enumerate(X_test):
                            betas, x_reference = fetch_computed_attributions('data/', dataset_name, x,
                                                                type='backpropagated',
                                                                reference_policy=reference_policy,
                                                                explainer_method=explainer_method)
                            betas_p2p, _ = fetch_computed_attributions('data/', dataset_name, x,
                                                                    type='p2p',
                                                                    reference_policy=reference_policy,
                                                                    explainer_method=explainer_method)
                            betas_segment, _ = fetch_computed_attributions('data/', dataset_name, x,
                                                                        type='segmented', reference_policy=reference_policy,
                                                                        explainer_method=explainer_method)
                            print(f"Running: {dataset_name} {mr_classifier.__class__.__name__} {label} {explainer_method} {reference_policy}")
                            results_df = df_schema.copy()
                            results_df['timestamp'].append(pd.Timestamp.now())
                            results_df['base_explainer'].append(explainer_method)
                            results_df['mr_classifier'].append(mr_classifier.__class__.__name__)
                            results_df['reference_policy'].append(reference_policy)
                            results_df['label'].append(label)
                            results_df['dataset'].append(dataset_name)
                            for perturbation_policy, args in PERTURBATIONS.items():
                                X_perturbed = get_perturbations(x, x_reference, budget=BUDGET, policy=perturbation_policy, **args)
