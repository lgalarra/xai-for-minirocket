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
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import importlib

import minirocket_multivariate_variable as mmv
from import_data import DataImporter
from reference import REFERENCE_POLICIES

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df, get_abnormal_hearbeat_for_classification,
                   get_handoutlines_for_classification)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score

if __name__ == '__main__':

    DATASETS = ['starlight-c1', 'starlight-c2', 'starlight-c3', 'cognitive-circles', 'ford-a', 'handoutlines', 'abnormal-heartbeat-c1']
    MR_CLASSIFIERS = {dataset: [
        pickle.load(open(f"data/{dataset}/LogisticRegression.pkl", "rb")),
         pickle.load(open(f"data/{dataset}/RandomForestClassifier.pkl", "rb")),
         pickle.load(open(f"data/{dataset}/MLPClassifier.pkl", "rb"))
         ] for dataset in DATASETS}

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

    results = defaultdict(dict)

    # Keep track of all classifier names (for table columns)
    all_classifiers = set()

    for dataset_name in DATASET_FETCH_FUNCTIONS.keys():
        dataset_fetch_function = DATASET_FETCH_FUNCTIONS[dataset_name]
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        data_importer = DataImporter(dataset_name)
        for classifier in MR_CLASSIFIERS[dataset_name]:
            classifier_name = classifier.classifier.__class__.__name__
            all_classifiers.add(classifier_name)
            y_test_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            print(f"Accuracy on test set ({dataset_name}): {acc}")
            results[dataset_name][classifier_name] = acc
            print(f"Accuracy on test set ({dataset_name}, {classifier_name}): {acc:.4f}")

    # Sort columns for reproducibility
    all_classifiers = sorted(all_classifiers)
    print(results)
    # ---- LaTeX table generation ----
    # ---- LaTeX table generation ----

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{l" + "c" * len(all_classifiers) + r"}")
    latex.append(r"\toprule")

    # Header
    header = "Dataset & " + " & ".join(all_classifiers) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")

    # Rows
    for dataset_name in sorted(results.keys()):
        row = [dataset_name]
        for clf in all_classifiers:
            if clf in results[dataset_name]:
                row.append(f"{results[dataset_name][clf]:.3f}")
            else:
                row.append("--")
        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Test accuracy for each classifier and dataset}")
    latex.append(r"\label{tab:accuracy_results}")
    latex.append(r"\end{table}")

    # Print LaTeX table
    print("\n".join(latex))
