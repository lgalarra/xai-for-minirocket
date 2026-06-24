#!/usr/bin/env python
# coding: utf-8
import copy
import itertools
# In[41]:
import os
import pickle
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pertutils import get_perturbations, ensure_consistency
from export_data import DataExporter
from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import importlib

import minirocket_multivariate_variable as mmv
from import_data import DataImporter
from reference import REFERENCE_POLICIES
from export_data import SEGMENTED_EXPLANATION_SEGMENTS, TSHAP_CONFIGS, get_tshap_key

TSHAP_REPO_PATH = Path(__file__).resolve().parents[2] / "tshap"
if TSHAP_REPO_PATH.exists() and str(TSHAP_REPO_PATH) not in sys.path:
    sys.path.append(str(TSHAP_REPO_PATH))

from tshap.synthetic import DoubleFreqTest

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df, get_abnormal_hearbeat_for_classification,
                   get_handoutlines_for_classification)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score


def get_double_freq_test_for_classification(n_samples=100):
    synth_gen = DoubleFreqTest()
    X_train, y_train, _ = synth_gen.generate_classification_data_and_attribs(
        n_samples=n_samples,
        random_seed=0
    )
    X_test, y_test, _ = synth_gen.generate_classification_data_and_attribs(
        n_samples=int(n_samples / 5),
        random_seed=1
    )
    return (X_train.astype(np.float32), y_train.astype(int)), (X_test.astype(np.float32), y_test.astype(int))


def compute_difference(classifier, X_test, X_perturbed, X_reference, budget) -> (np.array, np.array, float):
    #print(X_test.shape, X_perturbed.shape, X_reference.shape)    
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


def metric_columns(prefix: str) -> tuple:
    return (f'{prefix}f_minus_f0', f'{prefix}f_minus_f0-mean', f'{prefix}f_minus_f0-std',
            f'{prefix}f_minus_f0-change_ratio', f'{prefix}f_minus_f0_norm',
            f'{prefix}f_minus_f0_norm-mean', f'{prefix}f_minus_f0_norm-std')


def add_metric_columns(schema: dict, prefix: str):
    for column in metric_columns(prefix):
        schema[column] = []


def append_metrics(results: dict, prefix: str, metric, norm_metric, change_ratio):
    (metric_column, mean_column, std_column, change_column,
     norm_column, norm_mean_column, norm_std_column) = metric_columns(prefix)
    results[metric_column].append(to_sep_list(metric))
    results[mean_column].append(np.mean(metric))
    results[std_column].append(np.std(metric))
    results[norm_column].append(to_sep_list(norm_metric))
    results[norm_mean_column].append(np.mean(norm_metric))
    results[norm_std_column].append(np.std(norm_metric))
    results[change_column].append(change_ratio)


def append_missing_metrics(results: dict, prefix: str):
    append_metrics(results, prefix, [-1.0], [-1.0], -1.0)


def has_explanations(explanations) -> bool:
    values = np.asarray(explanations, dtype=object)
    return values.size > 0 and any(value is not None for value in values.flat)


def compute_perturbation_metrics(classifier, X_test, X_reference, X_explanations, explainer_method,
                                 perturbation_policy, args, budget):
    X_test_for_explanations = X_test.copy()
    X_reference_for_explanations = X_reference.copy()
    X_explanations = X_explanations.copy()
    X_explanations, X_test_for_explanations, X_reference_for_explanations = ensure_consistency(
        X_explanations, X_test_for_explanations, X_reference_for_explanations
    )
    X_perturbed, n_perturbed_points = get_perturbations(
        X_test_for_explanations,
        X_reference_for_explanations,
        X_explanations,
        explainer_method=explainer_method,
        policy=perturbation_policy,
        **args
    )

    if perturbation_policy == 'reference_to_instance_positive':
        metric, norm_metric, change_ratio = compute_difference(
            classifier, X_reference_for_explanations, X_perturbed, X_test_for_explanations, budget
        )
    else:
        metric, norm_metric, change_ratio = compute_difference(
            classifier, X_test_for_explanations, X_perturbed, X_reference_for_explanations, budget
        )
    return metric, norm_metric, change_ratio, n_perturbed_points


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

    THE_DISTANCE = 'euclidean'
    if len(sys.argv) > 7:
        THE_DISTANCE = sys.argv[7]

    DATASETS = ['starlight-c1', 'starlight-c2', 'starlight-c3', 'cognitive-circles', 'ford-a',
                'handoutlines', 'abnormal-heartbeat-c1', 'double-freq-test']
    CLASSIFIERS = ['LogisticRegression', 'RandomForestClassifier', 'MLPClassifier']
    if THE_CLASSIFIER is not None:
        CLASSIFIERS = [THE_CLASSIFIER]

    def load_classifiers(dataset):
        return [pickle.load(open(f"data/{dataset}/{classifier}.pkl", "rb")) for classifier in CLASSIFIERS]

    if THE_DATASET is None:
        MR_CLASSIFIERS = {dataset: load_classifiers(dataset) for dataset in DATASETS}
    else:
        MR_CLASSIFIERS = {THE_DATASET: load_classifiers(THE_DATASET)}

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": "get_forda_for_classification()",
        "double-freq-test": "get_double_freq_test_for_classification(n_samples=200)",
        "starlight-c1": "get_starlightcurves_for_classification('1')",
        "starlight-c2": "get_starlightcurves_for_classification('2')",
        "starlight-c3": "get_starlightcurves_for_classification('3')",
		"abnormal-heartbeat-c1": "get_abnormal_hearbeat_for_classification('1')",
        "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        "handoutlines": "get_handoutlines_for_classification('1')"
    }
    EXPLAINERS = ['extreme_feature_coalitions', 'shap', 'gradients', 'stratoshap-k1']
    DISTANCES = ['euclidean', 'pca-mr']
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

    reference_policies = REFERENCE_POLICIES if THE_REFERENCE_POLICY is None else [THE_REFERENCE_POLICY]

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
    if THE_DISTANCE != 'euclidean':
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'_{THE_DISTANCE}.csv')


    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'args': [], 'perturbation_policy': [], 'distance': []}
    add_metric_columns(df_schema, '')
    add_metric_columns(df_schema, 'p2p_')
    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
        add_metric_columns(df_schema, 'segmented_' if num_segments == 10 else f'segmented_n{num_segments}_')
    for window_size_percent, stride in TSHAP_CONFIGS:
        add_metric_columns(df_schema, f'tshap_{get_tshap_key(window_size_percent, stride)}_')
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
                    for distance in DISTANCES if THE_DISTANCE is None else [THE_DISTANCE]:
                        metadata_df = data_importer.get_metadata(classifier_name, explainer_method, label, distance)
                        (X_test, y_test, references_dict, explanations_dict, p2p_explanations_dict,
                         segmented_explanations_dict, tshap_explanations_dict) = (
                            DataImporter.get_series_from_metadata(metadata_df, reference_policies=reference_policies)
                        )
                        print('Label, explainer_method, distance: ', label, explainer_method, distance)
                        for perturbation_policy, all_args in PERTURBATIONS.items():
                            for combo in itertools.product(*all_args.values()):
                                args = dict(zip(all_args.keys(), combo))
                                print('Perturbation', perturbation_policy, 'Args: ', args)
                                for reference_policy in reference_policies:
                                    df_results = copy.deepcopy(df_schema)
                                    print('Backpropagated explanations')
                                    args['y'] = y_test if label == 'training' else classifier.predict(X_test)
                                    perturbation_budget = PERTURBATIONS[perturbation_policy]['budget'][0]

                                    metric, norm_metric, change_ratio, n_perturbed_points = compute_perturbation_metrics(
                                        classifier,
                                        X_test,
                                        references_dict[reference_policy],
                                        explanations_dict[reference_policy],
                                        explainer_method,
                                        perturbation_policy,
                                        args,
                                        perturbation_budget
                                    )
                                    append_metrics(df_results, '', metric, norm_metric, change_ratio)

                                    if has_explanations(p2p_explanations_dict[reference_policy]):
                                        print('P2p explanations')
                                        metric, norm_metric, change_ratio, _ = compute_perturbation_metrics(
                                            classifier,
                                            X_test,
                                            references_dict[reference_policy],
                                            p2p_explanations_dict[reference_policy],
                                            explainer_method,
                                            perturbation_policy,
                                            args,
                                            perturbation_budget
                                        )
                                        append_metrics(df_results, 'p2p_', metric, norm_metric, change_ratio)
                                    else:
                                        append_missing_metrics(df_results, 'p2p_')

                                    args['n_perturbed_points'] = n_perturbed_points
                                    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
                                        prefix = 'segmented_' if num_segments == 10 else f'segmented_n{num_segments}_'
                                        segmented_explanations = segmented_explanations_dict[reference_policy][num_segments]
                                        if has_explanations(segmented_explanations):
                                            print(f'Segmented explanations ({num_segments} segments)')
                                            metric, norm_metric, change_ratio, _ = compute_perturbation_metrics(
                                                classifier,
                                                X_test,
                                                references_dict[reference_policy],
                                                segmented_explanations,
                                                explainer_method,
                                                perturbation_policy,
                                                args,
                                                perturbation_budget
                                            )
                                            append_metrics(df_results, prefix, metric, norm_metric, change_ratio)
                                        else:
                                            append_missing_metrics(df_results, prefix)

                                    for window_size_percent, stride in TSHAP_CONFIGS:
                                        key = get_tshap_key(window_size_percent, stride)
                                        prefix = f'tshap_{key}_'
                                        tshap_explanations = tshap_explanations_dict[reference_policy][key]
                                        if has_explanations(tshap_explanations):
                                            print(f'TSHAP explanations ({key})')
                                            metric, norm_metric, change_ratio, _ = compute_perturbation_metrics(
                                                classifier,
                                                X_test,
                                                references_dict[reference_policy],
                                                tshap_explanations,
                                                explainer_method,
                                                perturbation_policy,
                                                args,
                                                perturbation_budget
                                            )
                                            append_metrics(df_results, prefix, metric, norm_metric, change_ratio)
                                        else:
                                            append_missing_metrics(df_results, prefix)

                                    del args['y']
                                    del args['n_perturbed_points']

                                    df_results['timestamp'].append(pd.Timestamp.now())
                                    df_results['base_explainer'].append(explainer_method)
                                    df_results['mr_classifier'].append(classifier_name)
                                    df_results['reference_policy'].append(reference_policy)
                                    df_results['label'].append(label)
                                    df_results['dataset'].append(dataset_name)
                                    df_results['args'].append(f'{args}')
                                    df_results['perturbation_policy'].append(perturbation_policy)
                                    df_results['distance'].append(distance)
                                    pd.DataFrame(df_results).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
