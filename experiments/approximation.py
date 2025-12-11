#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
import random
import sys

import numpy as np
import pandas as pd
import copy
import pickle

from scipy.stats import kendalltau
from sklearn.base import BaseEstimator

from explainer import Explanation
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
                   get_starlightcurves_for_classification,
                   get_abnormal_hearbeat_for_classification, COGNITIVE_CIRCLES_CHANNELS, COGNITIVE_CIRCLES_BASIC_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score
from export_data import DataExporter
from reference import REFERENCE_POLICIES

import argparse

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
        help="End index (default: -1)."
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


    args = parser.parse_args()

    # post-processing for boolean
    should_export_data = args.dump_data.lower() in ("yes", "true", "1")

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
    )

MR_CLASSIFIERS = {'LogisticRegression': LogisticRegression, 'RandomForestClassifier': RandomForestClassifier}

DATASET_FETCH_FUNCTIONS = {
    "ford-a": ("get_forda_for_classification()", [('C', 'Noise intensity')]),
    "abnormal-heartbeat-c0": ("get_abnormal_hearbeat_for_classification('0')", [('A', 'Amplitude Change')]),
    "abnormal-heartbeat-c1": ("get_abnormal_hearbeat_for_classification('1')", [('A', 'Amplitude Change')]),
    "abnormal-heartbeat-c2": ("get_abnormal_hearbeat_for_classification('2')", [('A', 'Amplitude Change')]),
    "abnormal-heartbeat-c3": ("get_abnormal_hearbeat_for_classification('3')", [('A', 'Amplitude Change')]),
    "abnormal-heartbeat-c4": ("get_abnormal_hearbeat_for_classification('4')", [('A', 'Amplitude Change')]),
    "starlight-c1": ("get_starlightcurves_for_classification('1')", [('B', 'Brightness')]),
    "starlight-c2": ("get_starlightcurves_for_classification('2')", [('B', 'Brightness')]),
    "starlight-c3": ("get_starlightcurves_for_classification('3')", [('B', 'Brightness')]),
    "cognitive-circles": (
        "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        [(x, COGNITIVE_CIRCLES_CHANNELS[x]) for x in
         cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')]
        )
}

def build_map_of_already_trained_classifiers(datasets: list, classifiers):
    return {dataset : {classifier : f'pickle.load(open("data/{dataset}/{classifier}.pkl", "rb"))'
                       for classifier in classifiers}
            for dataset in datasets
            }

MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT = build_map_of_already_trained_classifiers(['starlight-c1', 'starlight-c2', 'starlight-c3',
                                             'abnormal-hearbeat-c0', 'abnormal-hearbeat-c1'
                                          'abnormal-hearbeat-c2', 'abnormal-hearbeat-c3',
                                          'abnormal-hearbeat-c4', 'ford-a', 'cognitive-circles'],
                                                                                     ['LogisticRegression', 'RandomForestClassifier'])


MINIROCKET_PARAMS_DICT = {'ford-a': {'num_features': 500}, 'starlight-c1': {'num_features': 500},
                          'starlight-c2': {'num_features': 500}, 'starlight-c3': {'num_features': 500},
                          'cognitive-circles': {'num_features': 1000},
                          'abnormal-heartbeat-c0' : {'num_features': 1000},
                          'abnormal-heartbeat-c1' : {'num_features': 1000},
                          'abnormal-heartbeat-c2' : {'num_features': 1000},
                          'abnormal-heartbeat-c3' : {'num_features': 1000},
                            'abnormal-heartbeat-c4' : {'num_features': 1000},
                          }


def compute_explanations(x_target, y_target, classifier: MinirocketClassifier, explainer, configuration: tuple,
                         reference_policy: str, compute_all_explanations=False, top_alpha=None):
    (dataset_name, mr_classifier_name, explainer_method, label) = configuration

    explanation = list(explainer.explain_instances(x_target, y_target,
                                                   classifier_explainer=explainer_method,
                                                   reference_policy=reference_policy, top_alpha=top_alpha))[0]

    ## Point to point explanation
    explanation_p2p = None
    segmented_explanation = None
    if compute_all_explanations:
        reference = explanation.get_reference()
        instance = explanation.get_instance()
        explanation_p2p = classifier.explain_instances(instance,
                                                       reference,
                                                       explainer=explainer_method,
                                                       reference_policy=reference_policy
                                                       )
        ## Segmented explanation
        segmented_explanation = MinirocketSegmentedClassifier(classifier.classifier, instance,
                                                              reference).explain_instances(instance, reference,
            explainer=explainer_method,
            reference_policy=reference_policy
        )
    return explanation, explanation_p2p, segmented_explanation


def update(r2s, kendalls, runtimes_backpropagated, runtimes_p2p, runtimes_segmented, complexity_backpropagated,
           complexity_p2p, complexity_segmented, local_accuracy, error, reference_policy):
    if reference_policy not in r2s:
        r2s[reference_policy] = []
        error[reference_policy] = []
        kendalls[reference_policy] = []
        runtimes_backpropagated[reference_policy] = []
        runtimes_p2p[reference_policy] = []
        runtimes_segmented[reference_policy] = []
        complexity_backpropagated[reference_policy] = []
        complexity_p2p[reference_policy] = []
        complexity_segmented[reference_policy] = []
        local_accuracy[reference_policy] = 0

    r2, kendall = (0.0, 0.0) if explanation_p2p is None else compare_explanations(explanation, explanation_p2p)
    r2s[reference_policy].append(r2)
    kendalls[reference_policy].append(kendall)
    runtimes_backpropagated[reference_policy].append(explanation.get_runtime())
    runtimes_p2p[reference_policy].append(-1.0 if explanation_p2p is None else explanation_p2p.get_runtime())
    runtimes_segmented[reference_policy].append(-1.0 if segmented_explanation is None else segmented_explanation.get_runtime())
    complexity_backpropagated[reference_policy].append(np.count_nonzero(explanation.explanation['coefficients']))
    complexity_p2p[reference_policy].append(-1.0 if explanation_p2p is None else np.count_nonzero(explanation_p2p.explanation['coefficients']))
    complexity_segmented[reference_policy].append(
        -1.0 if segmented_explanation is None else np.count_nonzero(segmented_explanation.get_distributed_explanations_in_original_space()))
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
        end
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

    if end != -1:
        OUTPUT_FILE = OUTPUT_FILE.replace('.csv', f'-{start}-{end}.csv')
        DataExporter.METADATA_FILE.replace('.csv', f'-{start}-{end}.csv')
    
    def compare_explanations(explanation: Explanation, explanation_p2p: Explanation) -> (float, float):
        explanation_vector = explanation.get_attributions_as_single_vector()
        explanation_p2p_vector = explanation_p2p.get_attributions_as_single_vector()
        res = kendalltau(explanation_p2p_vector, explanation_vector)
        return r2_score(explanation_p2p_vector, explanation_vector), res.statistic
    
    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'r2s': [], 'r2s-mean': [], 'r2s-std': [],
                 'local_accuracy': [], 'error': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
                 'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
                 'runtimes-segmented-seconds': [], 'runtimes-segmented-mean': [], 'runtimes-segmented-std': [],
                 'complexity': [], 'complexity-mean': [], 'complexity-std': [],
                 'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': [],
                 'complexity-segmented': [], 'complexity-segmented-mean': [], 'complexity-segmented-std': [],
                 'kendall-taus': [], 'kendall-taus-mean': [], 'kendall-taus-std': []}
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    exporters_dict = {}
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
                        exporter = DataExporter(dataset_name, mr_classifier_name, explainer_method, label)
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
                    r2s = {}
                    kendalls = {}
                    runtimes_backpropagated = {}
                    runtimes_p2p = {}
                    runtimes_segmented = {}
                    complexity_backpropagated = {}
                    complexity_p2p = {}
                    complexity_segmented = {}
                    local_accuracy = {}
                    error = {}
                    for idx in range(start, end_dataset):
                        print(f'Instance {idx} out of {end_dataset - start} (end={end_dataset})')
                        measures_for_instance = {}
                        explanations_for_instance = {}
                        x_target = X_test[idx]
                        y_target = y_test[idx] if label == 'training' else y_test_pred[idx]
                        for reference_policy in REFERENCE_POLICIES:
                            explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                            explanation, explanation_p2p, segmented_explanation = (
                                compute_explanations(x_target, y_target, classifier, explainer, configuration,
                                                     reference_policy, compute_all_explanations=(topk is None),
                                                     top_alpha=topk)
                            )
                            explanations_for_instance[reference_policy] = (explanation, explanation_p2p,
                                                                           segmented_explanation)

                            update(r2s, kendalls, runtimes_backpropagated, runtimes_p2p, runtimes_segmented,
                                   complexity_backpropagated, complexity_p2p, complexity_segmented,
                                   local_accuracy, error, reference_policy)

                            measures_for_instance[reference_policy] = (kendalls[reference_policy], r2s[reference_policy],
                                                                       runtimes_backpropagated[reference_policy],
                                                                   runtimes_p2p[reference_policy], runtimes_segmented[reference_policy],
                                                                   complexity_backpropagated[reference_policy], complexity_p2p[reference_policy],
                                                                   complexity_segmented[reference_policy], local_accuracy[reference_policy], error[reference_policy])
                        dataset_measures.append(measures_for_instance)
                        if should_export_data:
                            print(f"Exporting instance {idx} to {exporter.output_path} ({configuration})")
                            exporter.export_instance_and_explanations(idx, y_target, features, explanations_for_instance,
                                                                      studied_reference_policies=studied_reference_policies,
                                                                      topk=topk)

                    for reference_policy in studied_reference_policies:
                        for instance_measures in dataset_measures:
                            (kendalls, r2s, runtimes_backpropagated,
                             runtimes_p2p, runtimes_segmented,
                             complexity_backpropagated, complexity_p2p,
                             complexity_segmented, local_accuracy, error) = instance_measures[reference_policy]
                            results_df_rp = copy.deepcopy(results_df_dict[reference_policy])
                            results_df_rp['reference_policy'].append(reference_policy)
                            results_df_rp['complexity'].append(to_sep_list(complexity_backpropagated))
                            results_df_rp['complexity-mean'].append(np.mean(complexity_backpropagated))
                            results_df_rp['complexity-std'].append(np.std(complexity_backpropagated))

                            results_df_rp['complexity-p2p'].append(to_sep_list(complexity_p2p))
                            results_df_rp['complexity-p2p-mean'].append(np.mean(complexity_p2p))
                            results_df_rp['complexity-p2p-std'].append(np.std(complexity_p2p))

                            results_df_rp['complexity-segmented'].append(to_sep_list(complexity_segmented))
                            results_df_rp['complexity-segmented-mean'].append(np.mean(complexity_segmented))
                            results_df_rp['complexity-segmented-std'].append(np.std(complexity_segmented))

                            results_df_rp['runtimes-seconds'].append(to_sep_list(runtimes_backpropagated))
                            results_df_rp['runtimes-mean'].append(np.mean(runtimes_backpropagated))
                            results_df_rp['runtimes-std'].append(np.std(runtimes_backpropagated))

                            results_df_rp['runtimes-p2p-seconds'].append(to_sep_list(runtimes_p2p))
                            results_df_rp['runtimes-p2p-mean'].append(np.mean(runtimes_p2p))
                            results_df_rp['runtimes-p2p-std'].append(np.std(runtimes_p2p))

                            results_df_rp['runtimes-segmented-seconds'].append(to_sep_list(runtimes_segmented))
                            results_df_rp['runtimes-segmented-mean'].append(np.mean(runtimes_segmented))
                            results_df_rp['runtimes-segmented-std'].append(np.std(runtimes_segmented))

                            results_df_rp['r2s'].append(to_sep_list(r2s))
                            results_df_rp['r2s-mean'].append(np.mean(r2s))
                            results_df_rp['r2s-std'].append(np.std(r2s))

                            results_df_rp['kendall-taus'].append(to_sep_list(kendalls))
                            results_df_rp['kendall-taus-mean'].append(np.mean(kendalls))
                            results_df_rp['kendall-taus-std'].append(np.std(kendalls))

                            results_df_rp['local_accuracy'].append(local_accuracy / len(r2s))
                            results_df_rp['error'].append(to_sep_list(error))
                            print(results_df_rp)
                            pd.DataFrame(results_df_rp).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

    if should_export_data:
        for configuration, exporter in exporters_dict.items():
            exporter.export_metametadata()






    
    
    
    
