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


def count_nonzero_attributions(explanations) -> int:
    try:
        values = np.asarray(explanations, dtype=float)
        if values.size == 0:
            return 0
        values = values[np.isfinite(values)]
        return int(np.count_nonzero(values))
    except (TypeError, ValueError):
        values = np.asarray(explanations, dtype=object)
        total = 0
        for value in values.flat:
            if value is None:
                continue
            total += count_nonzero_attributions(value)
        return total


def count_explanation_series(explanations) -> int:
    values = np.asarray(explanations, dtype=object)
    if values.size == 0:
        return 0
    if values.ndim == 0:
        return 1
    return len(values)


def get_explanation_count_info(name, explanations):
    per_series_counts = count_nonzero_attributions_per_series(explanations)
    nonzero_count = int(per_series_counts.sum())
    series_count = len(per_series_counts)
    avg_nonzero_per_series = nonzero_count / series_count if series_count > 0 else 0.0
    return name, nonzero_count, avg_nonzero_per_series, per_series_counts


def count_nonzero_attributions_per_series(explanations) -> np.ndarray:
    values = np.asarray(explanations, dtype=object)
    if values.size == 0:
        return np.array([], dtype=int)
    if values.ndim == 0:
        return np.array([count_nonzero_attributions(values.item())], dtype=int)
    return np.array([count_nonzero_attributions(value) for value in values], dtype=int)


def get_perturbation_count_basis(reference_policy, explanations_dict, p2p_explanations_dict,
                                 segmented_explanations_dict, tshap_explanations_dict):
    p2p_name, p2p_count, p2p_avg, p2p_per_series_counts = get_explanation_count_info(
        'p2p/end-to-end',
        p2p_explanations_dict[reference_policy]
    )
    if p2p_count > 0:
        return p2p_name, p2p_count, p2p_avg, p2p_per_series_counts

    counts = [get_explanation_count_info('backpropagated', explanations_dict[reference_policy])]
    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
        counts.append(get_explanation_count_info(
            f'segmented_{num_segments}',
            segmented_explanations_dict[reference_policy][num_segments]
        ))
    for window_size_percent, stride in TSHAP_CONFIGS:
        key = get_tshap_key(window_size_percent, stride)
        counts.append(get_explanation_count_info(f'tshap_{key}', tshap_explanations_dict[reference_policy][key]))

    positive_counts = [
        (name, count, avg, per_series_counts)
        for name, count, avg, per_series_counts in counts
        if count > 0
    ]
    if not positive_counts:
        raise ValueError(
            f"No non-zero explanations found for reference policy {reference_policy}. "
            "Cannot decide how many observations to perturb."
        )
    return min(positive_counts, key=lambda item: item[1])


def get_n_perturbed_points(nonzero_attribution_count: int, percentile_cut: float) -> int:
    counts = np.asarray(nonzero_attribution_count, dtype=float)
    n_points = np.ceil(counts * (100.0 - percentile_cut) / 100.0).astype(int)
    if n_points.ndim == 0:
        return int(n_points)
    return n_points


def select_perturbation_counts(n_perturbed_points, kept_indices):
    counts = np.asarray(n_perturbed_points)
    if counts.ndim == 0:
        return n_perturbed_points
    return counts[np.asarray(kept_indices, dtype=int)]


def describe_n_perturbed_points(n_perturbed_points) -> str:
    counts = np.asarray(n_perturbed_points)
    if counts.ndim == 0:
        return str(int(counts))
    if counts.size == 0:
        return "0 observations"
    return (
        f"{int(counts.sum())} observations total "
        f"(mean {counts.mean():.2f}, min {int(counts.min())}, max {int(counts.max())} per series)"
    )


def get_perturbation_args_for_explainer(all_args: dict, explainer_method: str) -> dict:
    args_for_explainer = copy.deepcopy(all_args)
    if 'percentile_cut' in args_for_explainer:
        percentile_cuts = args_for_explainer['percentile_cut']
        if 10 not in percentile_cuts:
            args_for_explainer['percentile_cut'] = [*percentile_cuts, 10]
    return args_for_explainer


def compute_perturbation_metrics(classifier, X_test, X_reference, X_explanations, explainer_method,
                                 perturbation_policy, args, budget):
    X_test_for_explanations = X_test.copy()
    X_reference_for_explanations = X_reference.copy()
    X_explanations = X_explanations.copy()
    X_explanations, X_test_for_explanations, X_reference_for_explanations, kept_indices = ensure_consistency(
        X_explanations, X_test_for_explanations, X_reference_for_explanations,
        return_kept_indices=True
    )
    args = copy.deepcopy(args)
    if 'n_perturbed_points' in args:
        args['n_perturbed_points'] = select_perturbation_counts(args['n_perturbed_points'], kept_indices)
    X_perturbed, n_perturbed_points = get_perturbations(
        X_test_for_explanations,
        X_reference_for_explanations,
        X_explanations,
        explainer_method=explainer_method,
        policy=perturbation_policy,
        **args
    )

    if perturbation_policy.startswith('reference_to_instance'):
        metric, norm_metric, change_ratio = compute_difference(
            classifier, X_reference_for_explanations, X_perturbed, X_test_for_explanations, budget
        )
    else:
        metric, norm_metric, change_ratio = compute_difference(
            classifier, X_test_for_explanations, X_perturbed, X_reference_for_explanations, budget
        )
    return metric, norm_metric, change_ratio, n_perturbed_points


def supports_explainer_for_classifier(explainer_method, classifier_name):
    return explainer_method != 'gradients' or classifier_name == 'LogisticRegression'


def supports_perturbation_for_explainer(perturbation_policy, explainer_method):
    return not perturbation_policy.startswith('gradient_gaussian') or explainer_method == 'gradients'


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
        "double-freq-test": "get_double_freq_test_for_classification(n_samples=250)",
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
                    'instance_to_reference_bottom': {'percentile_cut': [50, 75, 90],
                     'interpolation': [0.25, 0.5, 0.75, 1.0], 'budget': [1]
                    },
                    'instance_to_reference_random': {'percentile_cut': [50, 75, 90],
                     'interpolation': [0.25, 0.5, 0.75, 1.0], 'budget': [BUDGET]
                    },
                    'instance_to_reference_random_no_positive': {'percentile_cut': [50, 75, 90],
                     'interpolation': [0.25, 0.5, 0.75, 1.0], 'budget': [BUDGET]
                    },
                    'gaussian' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.5, 2.0, 1.5, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gaussian_bottom' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gaussian_random' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gaussian_random_no_positive' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gradient_gaussian' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.5, 2.0, 1.5, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gradient_gaussian_bottom' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gradient_gaussian_random' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'gradient_gaussian_random_no_positive' : {'percentile_cut': [90, 75, 50],
                                    'sigma' : [3.0, 2.0, 1.0],
                                  'budget': [BUDGET]
                    },
                    'reference_to_instance': {'percentile_cut': [90, 75, 50],
                                               'interpolation': [0.25, 0.5, 0.75, 1.0],
                                               'budget': [1]
                    },
                    'reference_to_instance_bottom': {'percentile_cut': [90, 75, 50],
                                  'interpolation': [0.25, 0.5, 0.75, 1.0],
                                  'budget': [1]
                    },
                    'reference_to_instance_random': {'percentile_cut': [90, 75, 50],
                                         'interpolation': [0.25, 0.5, 0.75, 1.0],
                                         'budget': [BUDGET]
                    },
                    'reference_to_instance_random_no_positive': {'percentile_cut': [90, 75, 50],
                                         'interpolation': [0.25, 0.5, 0.75, 1.0],
                                         'budget': [BUDGET]
                    }
    }

    reference_policies = REFERENCE_POLICIES if THE_REFERENCE_POLICY is None else [THE_REFERENCE_POLICY]

    # In[42]:
    OUTPUT_FILE = 'results/perturbation-results.csv'
    os.makedirs('results', exist_ok=True)
    DataExporter.METADATA_FILE = 'metadata.csv'
    if THE_DATASET is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{THE_DATASET}.csv')
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
                    if not supports_explainer_for_classifier(explainer_method, classifier_name):
                        print(
                            f'Skipping {explainer_method} for {classifier_name}: '
                            'gradients are only supported for LogisticRegression'
                        )
                        continue
                    for distance in DISTANCES if THE_DISTANCE is None else [THE_DISTANCE]:
                        metadata_df = data_importer.get_metadata(
                            classifier_name,
                            explainer_method,
                            label,
                            distance,
                            reference_policy=THE_REFERENCE_POLICY
                        )
                        (X_test, y_test, references_dict, explanations_dict, p2p_explanations_dict,
                         segmented_explanations_dict, tshap_explanations_dict) = (
                            DataImporter.get_series_from_metadata(metadata_df, reference_policies=reference_policies)
                        )
                        print('Label, explainer_method, distance: ', label, explainer_method, distance)
                        for perturbation_policy, all_args in PERTURBATIONS.items():
                            if not supports_perturbation_for_explainer(perturbation_policy, explainer_method):
                                print(f'Skipping {perturbation_policy} for {explainer_method}')
                                continue
                            args_for_explainer = get_perturbation_args_for_explainer(all_args, explainer_method)
                            for combo in itertools.product(*args_for_explainer.values()):
                                args = dict(zip(args_for_explainer.keys(), combo))
                                print('Perturbation', perturbation_policy, 'Args: ', args)
                                for reference_policy in reference_policies:
                                    df_results = copy.deepcopy(df_schema)
                                    args['y'] = y_test if label == 'training' else classifier.predict(X_test)
                                    perturbation_budget = args_for_explainer['budget'][0]
                                    (
                                        count_source,
                                        nonzero_attribution_count,
                                        avg_nonzero_per_series,
                                        nonzero_attribution_counts,
                                    ) = get_perturbation_count_basis(
                                        reference_policy,
                                        explanations_dict,
                                        p2p_explanations_dict,
                                        segmented_explanations_dict,
                                        tshap_explanations_dict
                                    )
                                    args['n_perturbed_points'] = get_n_perturbed_points(
                                        nonzero_attribution_counts,
                                        args['percentile_cut']
                                    )
                                    print(
                                        f'Perturbation count source: {count_source} '
                                        f'({nonzero_attribution_count} non-zero attributions, '
                                        f'{avg_nonzero_per_series:.2f} per series on average)'
                                    )
                                    print(
                                        f'Percentile cut {args["percentile_cut"]}: '
                                        f'perturbing at most {describe_n_perturbed_points(args["n_perturbed_points"])}'
                                    )

                                    print('Backpropagated explanations')
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
                                    print(f'Backpropagated perturbed observations: {n_perturbed_points}')
                                    append_metrics(df_results, '', metric, norm_metric, change_ratio)

                                    if has_explanations(p2p_explanations_dict[reference_policy]):
                                        print('P2p explanations')
                                        metric, norm_metric, change_ratio, p2p_perturbed_points = compute_perturbation_metrics(
                                            classifier,
                                            X_test,
                                            references_dict[reference_policy],
                                            p2p_explanations_dict[reference_policy],
                                            explainer_method,
                                            perturbation_policy,
                                            args,
                                            perturbation_budget
                                        )
                                        print(f'P2p perturbed observations: {p2p_perturbed_points}')
                                        append_metrics(df_results, 'p2p_', metric, norm_metric, change_ratio)
                                    else:
                                        print('P2p explanations missing')
                                        append_missing_metrics(df_results, 'p2p_')

                                    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
                                        prefix = 'segmented_' if num_segments == 10 else f'segmented_n{num_segments}_'
                                        segmented_explanations = segmented_explanations_dict[reference_policy][num_segments]
                                        if has_explanations(segmented_explanations):
                                            print(f'Segmented explanations ({num_segments} segments)')
                                            metric, norm_metric, change_ratio, segmented_perturbed_points = compute_perturbation_metrics(
                                                classifier,
                                                X_test,
                                                references_dict[reference_policy],
                                                segmented_explanations,
                                                explainer_method,
                                                perturbation_policy,
                                                args,
                                                perturbation_budget
                                            )
                                            print(
                                                f'Segmented perturbed observations ({num_segments} segments): '
                                                f'{segmented_perturbed_points}'
                                            )
                                            append_metrics(df_results, prefix, metric, norm_metric, change_ratio)
                                        else:
                                            print(f'Segmented explanations missing ({num_segments} segments)')
                                            append_missing_metrics(df_results, prefix)

                                    for window_size_percent, stride in TSHAP_CONFIGS:
                                        key = get_tshap_key(window_size_percent, stride)
                                        prefix = f'tshap_{key}_'
                                        tshap_explanations = tshap_explanations_dict[reference_policy][key]
                                        if has_explanations(tshap_explanations):
                                            print(f'TSHAP explanations ({key})')
                                            metric, norm_metric, change_ratio, tshap_perturbed_points = compute_perturbation_metrics(
                                                classifier,
                                                X_test,
                                                references_dict[reference_policy],
                                                tshap_explanations,
                                                explainer_method,
                                                perturbation_policy,
                                                args,
                                                perturbation_budget
                                            )
                                            print(f'TSHAP perturbed observations ({key}): {tshap_perturbed_points}')
                                            append_metrics(df_results, prefix, metric, norm_metric, change_ratio)
                                        else:
                                            print(f'TSHAP explanations missing ({key})')
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
