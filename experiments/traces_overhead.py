#!/usr/bin/env python
# coding: utf-8
import copy
import itertools
# In[41]:
import os
import pickle
import time

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
from approximation import MINIROCKET_PARAMS_DICT



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
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": "get_forda_for_classification()",
        "starlight-c1": "get_starlightcurves_for_classification('1')",
        #"starlight-c2": "get_starlightcurves_for_classification('2')",
        #"starlight-c3": "get_starlightcurves_for_classification('3')",
        "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
    }

    # In[42]:
    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'f_minus_f0' : [], 'f_minus_f0-mean': [], 'f_minus_f0-std': [], 'f_minus_f0-change_ratio': [],
                 'f_minus_f0_norm': [], 'f_minus_f0_norm-mean': [], 'f_minus_f0_norm-std': [],
                 'p2p_f_minus_f0': [], 'p2p_f_minus_f0-mean': [], 'p2p_f_minus_f0-std': [], 'p2p_f_minus_f0-change_ratio': [],
                 'p2p_f_minus_f0_norm': [], 'p2p_f_minus_f0_norm-mean': [], 'p2p_f_minus_f0_norm-std': [],
                 'segmented_f_minus_f0': [], 'segmented_f_minus_f0-mean': [], 'segmented_f_minus_f0-std': [], 'segmented_f_minus_f0-change_ratio': [],
                 'segmented_f_minus_f0_norm': [], 'segmented_f_minus_f0_norm-mean': [], 'segmented_f_minus_f0_norm-std': [],
                 'args': [], 'perturbation_policy': [] }
    #final_df = pd.DataFrame(df_schema.copy())
    #pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    print('Dataset\tInstance\tTime RT\tTime NRT')
    for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
        import minirocket_multivariate as mv
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        minirocket_params = mmv.fit_minirocket_parameters(X_train, **MINIROCKET_PARAMS_DICT[dataset_name])
        n, C, L = X_train.shape
        for idx, x_train in enumerate(X_train):
            start = time.perf_counter()
            mmv.transform_prime(x_train, parameters=minirocket_params)
            time_elapsed1 = time.perf_counter() - start
            Xi = x_train.astype(np.float32)  # (C, L)
            Li = np.array([L], dtype=np.int32)
            start = time.perf_counter()
            mmv.transform(Xi, Li, minirocket_params)
            time_elapsed2 = time.perf_counter() - start
            print(f'{dataset_name}\t{idx}\t{time_elapsed1}\t{time_elapsed2}')

    for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
        import minirocket_multivariate as mv
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        minirocket_params = mmv.fit_minirocket_parameters(X_train, **MINIROCKET_PARAMS_DICT[dataset_name])
        start = time.perf_counter()
        mmv.transform_prime(X_train, parameters=minirocket_params)
        time_elapsed1 = time.perf_counter() - start
        print('Augmented:', time_elapsed1)
        start = time.perf_counter()
        mmv._transform_batch(X_train, minirocket_params)
        time_elapsed2 = time.perf_counter() - start
        print('Non-augmented:', time_elapsed2)
        print(f'{dataset_name}\tBATCH\t{time_elapsed1}\t{time_elapsed2}')

