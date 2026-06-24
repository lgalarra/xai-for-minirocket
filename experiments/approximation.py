#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import copy
import pickle

from scipy.stats import kendalltau
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from explainer import Explanation, MinirocketExplainer
from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")   # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import importlib

import minirocket_multivariate_variable as mmv

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, get_handoutlines_for_classification,
                   get_abnormal_hearbeat_for_classification, COGNITIVE_CIRCLES_CHANNELS, COGNITIVE_CIRCLES_BASIC_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df, get_double_freq_test_for_classification)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score
from export_data import DataExporter
from reference import REFERENCE_POLICIES

import argparse

from tshap.synthetic import DoubleFreqTest
from tshap.tshap import tshap_explanation

def parse_args():
    parser = argparse.ArgumentParser(
        description="approximation.py [dump-data=yes|no] [dataset1,dataset2,...] "
                    "[label1,label2=predicted|training] "
                    "[model1,model2,...=LogisticRegression|RandomForestClassifier] "
                    "[start=0] [end=-1]"
    )

    parser.add_argument(
        "--dump-data",
        "-d",
        type=str,
        default="no",
        choices=["yes", "no", "true", "false", "1", "0"],
        help="Whether to export data (yes/no)."
    )

    parser.add_argument(
        "--datasets",
        "-D",
        type=lambda s: s.split(','),
        default=None,
        help="Comma-separated list of datasets."
    )

    parser.add_argument(
        "--labels",
        "-L",
        type=lambda s: s.split(','),
        default=None,
        help="Comma-separated list of labels."
    )

    parser.add_argument(
        "--models",
        "-M",
        type=lambda s: s.split(','),
        default=None,
        help="Comma-separated list of model names."
    )

    parser.add_argument(
        "--explainers",
        "-E",
        type=lambda s: s.split(','),
        default=None,
        help="Comma-separated list of explainers."
    )

    parser.add_argument(
        "--propagate_top_features",
        "-t",
        type=int,
        default=None,
        help="A positive integer that defines whether only the top-k."
    )

    parser.add_argument(
        "--reference_policy",
        "-r",
        type=lambda s: s.split(','),
        default=None,
        help="The used reference policy: 'opposite_class_closest_instance', 'opposite_class_medoid', 'opposite_class_centroid', 'global_medoid', 'global_centroid', 'opposite_class_farthest_instance'"
    )

    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=0,
        help="Start index (default: 0)."
    )

    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=sys.maxsize-1,
        help="End index (default: biggest possible integer)."
    )

    parser.add_argument(
        "--job_id",
        type=int,
        default=random.randint(1_000_000, 9_999_999),
        help="Job identifier (int). Default: random."
    )

    parser.add_argument(
        "--run_id",
        type=int,
        default=random.randint(1_000_000, 9_999_999),
        help="Run identifier (int). Default: random."
    )

    parser.add_argument(
        "--p2p_explanations",
        "-p",
        type=str,
        default="yes",
        choices=["yes", "no", "true", "false", "1", "0"],
        help="Whether to compute the p2p explanations (yes/no)."
    )

    parser.add_argument(
        "--tshap_explanations",
        "-T",
        type=str,
        default="yes",
        choices=["yes", "no", "true", "false", "1", "0"],
        help="Whether to compute the t-shap explanations (yes/no)."
    )

    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="euclidean",
        help="Distance metric used to calculate the reference instances: euclidean, pca-mr"
    )


    args = parser.parse_args()

    # post-processing for boolean
    should_export_data = args.dump_data.lower() in ("yes", "true", "1")
    compute_p2p_explanations = args.p2p_explanations.lower() in ("yes", "true", "1")
    compute_tshap_explanations = args.tshap_explanations.lower() in ("yes", "true", "1")

    return (
        should_export_data,
        args.datasets,
        args.labels,
        args.models,
        args.explainers,
        args.propagate_top_features,
        args.reference_policy,
        args.start,
        args.end,
        args.metric,
        compute_p2p_explanations,
        compute_tshap_explanations
    )

MR_CLASSIFIERS = {'LogisticRegression': LogisticRegression,
                  'RandomForestClassifier': RandomForestClassifier,
                  'MLPClassifier' : MLPClassifier}

DATASET_FETCH_FUNCTIONS = {
    "ford-a": ("get_forda_for_classification()", [('C', 'Noise intensity')]),
    "double-freq-test": ("get_double_freq_test_for_classification(n_samples=200)", [('X', 'Frequency')]),
    "abnormal-heartbeat-c1": ("get_abnormal_hearbeat_for_classification('1')", [('A', 'Amplitude Change')]),
    "starlight-c1": ("get_starlightcurves_for_classification('1')", [('B', 'Brightness')]),
    "starlight-c2": ("get_starlightcurves_for_classification('2')", [('B', 'Brightness')]),
    "starlight-c3": ("get_starlightcurves_for_classification('3')", [('B', 'Brightness')]),
    "cognitive-circles": (
        "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        [(x, COGNITIVE_CIRCLES_CHANNELS[x]) for x in
         cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')]
        ),
    "handoutlines" : ("get_handoutlines_for_classification('1')", [('X', 'X')])
}

def build_map_of_already_trained_classifiers(datasets: list, classifiers):
    return {dataset : {classifier : f'pickle.load(open("data/{dataset}/{classifier}.pkl", "rb"))'
                       for classifier in classifiers}
            for dataset in datasets
            }

MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT = build_map_of_already_trained_classifiers(['starlight-c1', 'starlight-c2', 'starlight-c3',
                                             'abnormal-heartbeat-c1', 'ford-a', 'cognitive-circles', 'handoutlines',
                                          'double-freq-test'], ['LogisticRegression', 'RandomForestClassifier', 'MLPClassifier'])

### Some notes:
MINIROCKET_PARAMS_DICT = {'ford-a': {'num_features': 5000}, 'starlight-c1': {'num_features': 5000},
                          'starlight-c2': {'num_features': 1000}, 'starlight-c3': {'num_features': 5000},
                          'handoutlines': {'num_features': 5000},
                          'cognitive-circles': {'num_features': 500},
                          'abnormal-heartbeat-c1' : {'num_features': 10000},
                          'double-freq-test': {'num_features': 5000},
                          }

SEGMENTED_EXPLANATION_SEGMENTS = (10, 20, 50, 100)
TSHAP_CONFIGS = tuple(
    (window_size_percent, stride)
    for window_size_percent in (10, 15, 20)
    for stride in (5, 20)
)

def get_segmented_runtime_columns(num_segments: int) -> tuple:
    prefix = "runtimes-segmented" if num_segments == 10 else f"runtimes-segmented-n{num_segments}"
    return f"{prefix}-seconds", f"{prefix}-mean", f"{prefix}-std"

def get_segmented_complexity_columns(num_segments: int) -> tuple:
    prefix = "complexity-segmented" if num_segments == 10 else f"complexity-segmented-n{num_segments}"
    return prefix, f"{prefix}-mean", f"{prefix}-std"

def get_tshap_key(window_size_percent: int, stride: int) -> str:
    return f"w{window_size_percent}_s{stride}"

def get_tshap_runtime_columns(window_size_percent: int, stride: int) -> tuple:
    prefix = f"runtimes-tshap-{get_tshap_key(window_size_percent, stride)}"
    return f"{prefix}-seconds", f"{prefix}-mean", f"{prefix}-std"

def get_tshap_complexity_columns(window_size_percent: int, stride: int) -> tuple:
    prefix = f"complexity-tshap-{get_tshap_key(window_size_percent, stride)}"
    return prefix, f"{prefix}-mean", f"{prefix}-std"

def compute_tshap_explanations(classifier: MinirocketClassifier, instance: np.ndarray, reference: np.ndarray,
                               y_target) -> dict:
    tshap_explanations = {}
    model_fn = lambda X: classifier.predict_proba(X)[:, int(y_target)]
    series_length = instance.shape[-1]
    for window_size_percent, stride in TSHAP_CONFIGS:
        window_length = max(1, int(series_length * window_size_percent / 100))
        start = time.perf_counter()
        tshap_window_attribs, _ = tshap_explanation(
            model_fn,
            np.array([instance]),
            baselines=np.array([reference]),
            window_length=window_length,
            stride=stride,
            roi=False
        )
        time_elapsed = time.perf_counter() - start
        key = get_tshap_key(window_size_percent, stride)
        print(
            f"Time elapsed (tshap {key}, window_length={window_length}, stride={stride}): "
            f"{time_elapsed}"
        )
        tshap_explanations[key] = {
            "coefficients": tshap_window_attribs[0],
            "time_elapsed": time_elapsed,
            "window_size_percent": window_size_percent,
            "window_length": window_length,
            "stride": stride,
        }
    return tshap_explanations


def compute_explanations(x_target, y_target, classifier: MinirocketClassifier, explainer, configuration: tuple,
                         reference_policy: str, compute_p2p_explanations=True,
                         compute_segmented_explanations=True, compute_tshap_explanations_enabled=True, top_alpha=None):
    (dataset_name, mr_classifier_name, explainer_method, label) = configuration

    explanation = list(explainer.explain_instances(x_target, y_target,
                                                   classifier_explainer=explainer_method,
                                                   reference_policy=reference_policy, top_alpha=top_alpha))[0]

    ## Point to point explanation
    reference = explanation.get_reference()
    instance = explanation.get_instance()
    if compute_p2p_explanations:
        explanation_p2p = classifier.explain_instances(instance,
                                                       reference,
                                                       explainer=explainer_method,
                                                       reference_policy=reference_policy
                                                       )
    else:
        explanation_p2p = None

    segmented_explanations = {}
    if compute_segmented_explanations:
        ## Segmented explanation
        for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
            segmented_explanations[num_segments] = MinirocketSegmentedClassifier(
                classifier.classifier,
                instance,
                reference,
                num_segments=num_segments
            ).explain_instances(
                instance,
                reference,
                explainer=explainer_method,
                reference_policy=reference_policy
            )

    tshap_explanations = {}
    if compute_tshap_explanations_enabled:
        tshap_explanations = compute_tshap_explanations(classifier, instance, reference, y_target)

    return explanation, explanation_p2p, segmented_explanations, tshap_explanations


def update(explanation, explanation_p2p, segmented_explanations, tshap_explanations,
           kendalls, runtimes_backpropagated, runtimes_p2p, runtimes_segmented, runtimes_tshap,
           complexity_backpropagated, complexity_p2p, complexity_segmented, complexity_tshap,
           local_accuracy, error, reference_policy):
    if reference_policy not in kendalls:
        error[reference_policy] = []
        kendalls[reference_policy] = []
        runtimes_backpropagated[reference_policy] = []
        runtimes_p2p[reference_policy] = []
        runtimes_segmented[reference_policy] = {num_segments: [] for num_segments in SEGMENTED_EXPLANATION_SEGMENTS}
        runtimes_tshap[reference_policy] = {get_tshap_key(*config): [] for config in TSHAP_CONFIGS}
        complexity_backpropagated[reference_policy] = []
        complexity_p2p[reference_policy] = []
        complexity_segmented[reference_policy] = {num_segments: [] for num_segments in SEGMENTED_EXPLANATION_SEGMENTS}
        complexity_tshap[reference_policy] = {get_tshap_key(*config): [] for config in TSHAP_CONFIGS}
        local_accuracy[reference_policy] = 0

    kendall = 0.0 if explanation_p2p is None else compare_explanations(explanation, explanation_p2p)
    kendalls[reference_policy].append(kendall)
    runtimes_backpropagated[reference_policy].append(explanation.get_runtime())
    runtimes_p2p[reference_policy].append(-1.0 if explanation_p2p is None else explanation_p2p.get_runtime())
    complexity_backpropagated[reference_policy].append(np.count_nonzero(explanation.explanation['coefficients']))
    complexity_p2p[reference_policy].append(-1.0 if explanation_p2p is None else np.count_nonzero(explanation_p2p.explanation['coefficients']))
    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
        segmented_explanation = segmented_explanations.get(num_segments)
        runtimes_segmented[reference_policy][num_segments].append(
            -1.0 if segmented_explanation is None else segmented_explanation.get_runtime()
        )
        complexity_segmented[reference_policy][num_segments].append(
            -1.0 if segmented_explanation is None
            else np.count_nonzero(segmented_explanation.get_distributed_explanations_in_original_space())
        )
    for window_size_percent, stride in TSHAP_CONFIGS:
        key = get_tshap_key(window_size_percent, stride)
        tshap_explanation_dict = tshap_explanations.get(key)
        runtimes_tshap[reference_policy][key].append(
            -1.0 if tshap_explanation_dict is None else tshap_explanation_dict["time_elapsed"]
        )
        complexity_tshap[reference_policy][key].append(
            -1.0 if tshap_explanation_dict is None
            else np.count_nonzero(tshap_explanation_dict["coefficients"])
        )
    (respects_local_accuracy, delta) = explanation.check_explanation_local_accuracy_wrt_minirocket()
    local_accuracy[reference_policy] += 1 if respects_local_accuracy else 0
    error[reference_policy].append(delta)

def get_classifier(mr_classifier_name: str, dataset_name: str) -> MinirocketClassifier:
    mr_params = MINIROCKET_PARAMS_DICT[dataset_name]
    mr_params['diff'] = (mr_classifier_name == 'LogisticRegression')
    model_path = DataExporter.get_classifier_path(mr_classifier_name, dataset_name)
    if os.path.exists(model_path):
        print(f'Loading existing classifier at {model_path}')
        classifier = eval(MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT[dataset_name][mr_classifier_name])
        mmv.MINIROCKET_PARAMETERS = classifier.minirocket_params
        if not hasattr(classifier, 'pca'):
            classifier._X_transform = mmv._transform_batch(classifier._X_train, parameters=mmv.MINIROCKET_PARAMETERS)
            classifier.pca = PCA(n_components=0.9, random_state=42).fit(classifier._X_transform)
    else:
        print('Training new classifier...')
        mr_classifier = MR_CLASSIFIERS[mr_classifier_name]()
        classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
        classifier.fit(X_train, y_train, **MINIROCKET_PARAMS_DICT[dataset_name])
        DataExporter.save_classifier(classifier, dataset_name)
    return classifier

if __name__ == '__main__':
    (
        should_export_data,
        datasets,
        labels,
        models,
        explainers,
        topk,
        reference_policy,
        start,
        end,
        metric,
        compute_p2p_explanations,
        compute_tshap_explanations
    ) = parse_args()

    print("should_export_data:", should_export_data)
    print("datasets:", datasets)
    print("labels:", labels)
    print("models:", models)
    print("explainers:", explainers)
    print("topk:", topk)
    print("reference_policy:", reference_policy)
    print("start:", start)
    print("end:", end)
    print("metric:", metric)
    print("compute_p2p_explanations:", compute_p2p_explanations)
    print("compute_tshap_explanations:", compute_tshap_explanations)


    LABELS = ['predicted', 'training']
    EXPLAINERS = ['extreme_feature_coalitions', 'shap', 'gradients', 'stratoshap-k1']

    if datasets is not None:
        DATASET_FETCH_FUNCTIONS = {dt: DATASET_FETCH_FUNCTIONS[dt] for dt in datasets}
    if labels is not None:
        LABELS = labels
    if models is None:
        models = MR_CLASSIFIERS.keys()
    if explainers is None:
        explainers = EXPLAINERS
    if reference_policy is not None:
        studied_reference_policies = reference_policy
    else:
        studied_reference_policies = REFERENCE_POLICIES


    # In[42]:
    OUTPUT_FILE = 'approximation-results.csv'
    if datasets is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{",".join(datasets)}.csv')
    if labels is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{",".join(labels)}.csv')
    if models is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{",".join(models)}.csv')
    if reference_policy is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{",".join(studied_reference_policies)}.csv')
    if explainers is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{",".join(explainers)}.csv')

    if topk is not None:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'topk-{topk}.csv')
        DataExporter.METADATA_FILE = DataExporter.METADATA_FILE.replace('.csv', f'-{topk}.csv')

    if end != sys.maxsize-1:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{start}-{end}.csv')

    if not should_export_data:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-NOTDUMPED.csv')

    OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'metric-{metric}.csv')

    def compare_explanations(explanation: Explanation, explanation_p2p: Explanation) -> float:
        explanation_vector = explanation.get_attributions_as_single_vector()
        explanation_p2p_vector = explanation_p2p.get_attributions_as_single_vector()
        res = kendalltau(explanation_p2p_vector, explanation_vector)
        return res.statistic
    
    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'local_accuracy': [], 'error': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
                 'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
                 'complexity': [], 'complexity-mean': [], 'complexity-std': [],
                 'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': [],
                 'kendall-taus': [], 'kendall-taus-mean': [], 'kendall-taus-std': []}
    for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
        for column in get_segmented_runtime_columns(num_segments):
            df_schema[column] = []
        for column in get_segmented_complexity_columns(num_segments):
            df_schema[column] = []
    for window_size_percent, stride in TSHAP_CONFIGS:
        for column in get_tshap_runtime_columns(window_size_percent, stride):
            df_schema[column] = []
        for column in get_tshap_complexity_columns(window_size_percent, stride):
            df_schema[column] = []
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    exporters_dict = {}
    MinirocketExplainer.REFERENCE_DISTANCE = metric

    for dataset_name, (dataset_fetch_function, features) in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        end_dataset = min(len(X_test), end)
        for mr_classifier_name in models:
            classifier = get_classifier(mr_classifier_name, dataset_name)

            y_test_pred = classifier.predict(X_test)
            print(f"Accuracy on test set ({dataset_name}): {accuracy_score(y_test, y_test_pred)}")
            for explainer_method in explainers:
                for label in LABELS:
                    configuration = (dataset_name, mr_classifier_name, explainer_method, label)
                    print(f"Evaluating configuration {configuration}")
                    if should_export_data:
                        exporter = DataExporter(dataset_name, mr_classifier_name, explainer_method, label, metric)
                        exporter.prepare_export(DATASET_FETCH_FUNCTIONS[dataset_name])
                        exporters_dict[configuration] = exporter

                    results_df_dict = {}
                    results_df = copy.deepcopy(df_schema.copy())
                    results_df['timestamp'].append(pd.Timestamp.now())
                    results_df['base_explainer'].append(explainer_method)
                    results_df['mr_classifier'].append(mr_classifier_name)
                    results_df['label'].append(label)
                    results_df['dataset'].append(dataset_name)
                    for reference_policy in studied_reference_policies:
                        results_df_dict[reference_policy] = copy.deepcopy(results_df)
                    dataset_measures = []
                    kendalls = {}
                    runtimes_backpropagated = {}
                    runtimes_p2p = {}
                    runtimes_segmented = {}
                    runtimes_tshap = {}
                    complexity_backpropagated = {}
                    complexity_p2p = {}
                    complexity_segmented = {}
                    complexity_tshap = {}
                    local_accuracy = {}
                    error = {}
                    for idx in range(start, end_dataset):
                        print(f'Instance {idx} out of {end_dataset - start} (end={end_dataset})')
                        measures_for_instance = {}
                        explanations_for_instance = {}
                        x_target = X_test[idx]
                        y_target = y_test[idx] if label == 'training' else y_test_pred[idx]
                        for reference_policy in studied_reference_policies:
                            explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                            explanation, explanation_p2p, segmented_explanations, tshap_explanations = (
                                compute_explanations(x_target, y_target, classifier, explainer, configuration,
                                                     reference_policy,
                                                     compute_p2p_explanations=(topk is None and compute_p2p_explanations),
                                                     compute_segmented_explanations=(topk is None),
                                                     compute_tshap_explanations_enabled=compute_tshap_explanations,
                                                     top_alpha=topk)
                            )
                            explanations_for_instance[reference_policy] = (explanation, explanation_p2p,
                                                                           segmented_explanations,
                                                                           tshap_explanations)

                            update(explanation, explanation_p2p, segmented_explanations, tshap_explanations,
                                   kendalls, runtimes_backpropagated, runtimes_p2p, runtimes_segmented, runtimes_tshap,
                                   complexity_backpropagated, complexity_p2p, complexity_segmented, complexity_tshap,
                                   local_accuracy, error, reference_policy)

                            measures_for_instance[reference_policy] = (kendalls[reference_policy],
                                                                       runtimes_backpropagated[reference_policy],
                                                                   runtimes_p2p[reference_policy], runtimes_segmented[reference_policy],
                                                                   runtimes_tshap[reference_policy],
                                                                   complexity_backpropagated[reference_policy], complexity_p2p[reference_policy],
                                                                   complexity_segmented[reference_policy], complexity_tshap[reference_policy],
                                                                   local_accuracy[reference_policy], error[reference_policy])
                        dataset_measures.append(measures_for_instance)
                        if should_export_data:
                            print(f"Exporting instance {idx} to {exporter.output_path} ({configuration})")
                            exporter.export_instance_and_explanations(idx, y_target, features, explanations_for_instance,
                                                                      studied_reference_policies=studied_reference_policies,
                                                                      topk=topk)

                    for reference_policy in studied_reference_policies:
                        for instance_measures in dataset_measures:
                            (kendalls, runtimes_backpropagated,
                             runtimes_p2p, runtimes_segmented, runtimes_tshap,
                             complexity_backpropagated, complexity_p2p,
                             complexity_segmented, complexity_tshap,
                             local_accuracy, error) = instance_measures[reference_policy]
                            results_df_rp = copy.deepcopy(results_df_dict[reference_policy])
                            results_df_rp['reference_policy'].append(reference_policy)
                            results_df_rp['complexity'].append(to_sep_list(complexity_backpropagated))
                            results_df_rp['complexity-mean'].append(np.mean(complexity_backpropagated))
                            results_df_rp['complexity-std'].append(np.std(complexity_backpropagated))

                            results_df_rp['complexity-p2p'].append(to_sep_list(complexity_p2p))
                            results_df_rp['complexity-p2p-mean'].append(np.mean(complexity_p2p))
                            results_df_rp['complexity-p2p-std'].append(np.std(complexity_p2p))

                            results_df_rp['runtimes-seconds'].append(to_sep_list(runtimes_backpropagated))
                            results_df_rp['runtimes-mean'].append(np.mean(runtimes_backpropagated))
                            results_df_rp['runtimes-std'].append(np.std(runtimes_backpropagated))

                            results_df_rp['runtimes-p2p-seconds'].append(to_sep_list(runtimes_p2p))
                            results_df_rp['runtimes-p2p-mean'].append(np.mean(runtimes_p2p))
                            results_df_rp['runtimes-p2p-std'].append(np.std(runtimes_p2p))

                            for num_segments in SEGMENTED_EXPLANATION_SEGMENTS:
                                (runtimes_column, runtimes_mean_column, runtimes_std_column) = get_segmented_runtime_columns(num_segments)
                                (complexity_column, complexity_mean_column, complexity_std_column) = get_segmented_complexity_columns(num_segments)
                                segmented_runtimes = runtimes_segmented[num_segments]
                                segmented_complexity = complexity_segmented[num_segments]

                                results_df_rp[runtimes_column].append(to_sep_list(segmented_runtimes))
                                results_df_rp[runtimes_mean_column].append(np.mean(segmented_runtimes))
                                results_df_rp[runtimes_std_column].append(np.std(segmented_runtimes))

                                results_df_rp[complexity_column].append(to_sep_list(segmented_complexity))
                                results_df_rp[complexity_mean_column].append(np.mean(segmented_complexity))
                                results_df_rp[complexity_std_column].append(np.std(segmented_complexity))

                            for window_size_percent, stride in TSHAP_CONFIGS:
                                key = get_tshap_key(window_size_percent, stride)
                                (runtimes_column, runtimes_mean_column, runtimes_std_column) = get_tshap_runtime_columns(window_size_percent, stride)
                                (complexity_column, complexity_mean_column, complexity_std_column) = get_tshap_complexity_columns(window_size_percent, stride)
                                tshap_runtimes = runtimes_tshap[key]
                                tshap_complexity = complexity_tshap[key]

                                results_df_rp[runtimes_column].append(to_sep_list(tshap_runtimes))
                                results_df_rp[runtimes_mean_column].append(np.mean(tshap_runtimes))
                                results_df_rp[runtimes_std_column].append(np.std(tshap_runtimes))

                                results_df_rp[complexity_column].append(to_sep_list(tshap_complexity))
                                results_df_rp[complexity_mean_column].append(np.mean(tshap_complexity))
                                results_df_rp[complexity_std_column].append(np.std(tshap_complexity))

                            results_df_rp['kendall-taus'].append(to_sep_list(kendalls))
                            results_df_rp['kendall-taus-mean'].append(np.mean(kendalls))
                            results_df_rp['kendall-taus-std'].append(np.std(kendalls))

                            results_df_rp['local_accuracy'].append(local_accuracy / len(kendalls))
                            results_df_rp['error'].append(to_sep_list(error))
                            print(results_df_rp)
                            pd.DataFrame(results_df_rp).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

    if should_export_data:
        for configuration, exporter in exporters_dict.items():
            exporter.export_metametadata()






    
    
    
    
