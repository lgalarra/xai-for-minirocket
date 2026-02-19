#!/usr/bin/env python
# coding: utf-8
import json
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
sys.path.append('code/')
from explainer import Explanation, get_dilated_triplet_array
from experiments.exputils import to_sep_list

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
                   get_starlightcurves_for_classification,
                   get_abnormal_hearbeat_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   COGNITIVE_CIRCLES_BASIC_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score
from reference import REFERENCE_POLICIES

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="predict_and_explain.py [dump-data=yes|no] [dataset1,dataset2,...] "
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
        default=sys.maxsize - 1,
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
        "--output_path",
        "-o",
        type=str,
        default="output",
        help="Directory to save the prediction and explanation results."
    )

    parser.add_argument(
        "--change_class_propagation",
        "-c",
        type=str,
        default="no",
        choices=["yes", "no", "true", "false", "1", "0"],
        help="Back-propagate attributions in decreasing order of importance until we switch class."
    )

    args = parser.parse_args()

    # post-processing for boolean
    should_export_data = args.dump_data.lower() in ("yes", "true", "1")
    compute_p2p_explanations = args.p2p_explanations.lower() in ("yes", "true", "1")
    change_class_propagation = args.change_class_propagation in ("yes", "true", "1")
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
        compute_p2p_explanations,
        args.output_path,
        change_class_propagation,
    )


MR_CLASSIFIERS = {'LogisticRegression': LogisticRegression, 'RandomForestClassifier': RandomForestClassifier}

DATASET_FETCH_FUNCTIONS = {
    "ford-a": ("get_forda_for_classification()", [('C', 'Noise intensity')]),
    "abnormal-heartbeat-c1": ("get_abnormal_hearbeat_for_classification('1')", [('A', 'Amplitude Change')]),
    "starlight-c1": ("get_starlightcurves_for_classification('1')", [('B', 'Brightness')]),
    "starlight-c2": ("get_starlightcurves_for_classification('2')", [('B', 'Brightness')]),
    "starlight-c3": ("get_starlightcurves_for_classification('3')", [('B', 'Brightness')]),
    "cognitive-circles": (
        "get_cognitive_circles_data_for_classification('data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        [(x, COGNITIVE_CIRCLES_CHANNELS[x]) for x in
         cognitive_circles_get_sorted_channels_from_df(data_dir='data/cognitive-circles')]
    )
}


def build_map_of_already_trained_classifiers(datasets: list, classifiers):
    return {dataset: {classifier: f'pickle.load(open("experiments/data/{dataset}/{classifier}.pkl", "rb"))'
                      for classifier in classifiers}
            for dataset in datasets
            }


MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT = build_map_of_already_trained_classifiers(
    ['starlight-c1', 'starlight-c2', 'starlight-c3',
     'abnormal-heartbeat-c1', 'ford-a', 'cognitive-circles'],
    ['LogisticRegression', 'RandomForestClassifier'])

MINIROCKET_PARAMS_DICT = {'ford-a': {'num_features': 500}, 'starlight-c1': {'num_features': 500},
                          'starlight-c2': {'num_features': 500}, 'starlight-c3': {'num_features': 500},
                          'cognitive-circles': {'num_features': 1000},
                          'abnormal-heartbeat-c1': {'num_features': 1000}
                          }


def compute_explanations(x_target, y_target, classifier: MinirocketClassifier, explainer, configuration: tuple,
                         reference_policy: str, compute_p2p_explanations=True, compute_segmented_explanations=True,
                         top_alpha=None, top_alpha_that_change_class=None):
    (dataset_name, mr_classifier_name, explainer_method, label) = configuration

    explanation = list(explainer.explain_instances(x_target, y_target,
                                                   classifier_explainer=explainer_method,
                                                   reference_policy=reference_policy,
                                                   top_alpha=top_alpha, top_alpha_that_change_class=top_alpha_that_change_class))[0]

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

    if compute_segmented_explanations:
        ## Segmented explanation
        segmented_explanation = MinirocketSegmentedClassifier(classifier.classifier, instance,
                                                              reference).explain_instances(instance, reference,
                                                                                           explainer=explainer_method,
                                                                                           reference_policy=reference_policy
                                                                                           )
    else:
        segmented_explanation = None

    return explanation, explanation_p2p, segmented_explanation


def get_classifier(mr_classifier_name: str, dataset_name: str) -> MinirocketClassifier:
    mr_params = MINIROCKET_PARAMS_DICT[dataset_name]
    mr_params['diff'] = (mr_classifier_name == 'LogisticRegression')
    model_path = f'experiments/data/{dataset_name}/{mr_classifier_name}.pkl'
    if os.path.exists(model_path):
        print(f'Loading existing classifier at {model_path}')
        classifier = eval(MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT[dataset_name][mr_classifier_name])
        mmv.MINIROCKET_PARAMETERS = classifier.minirocket_params
    else:
        print('Training new classifier...')
        mr_classifier = MR_CLASSIFIERS[mr_classifier_name]()
        classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
        classifier.fit(X_train, y_train, **MINIROCKET_PARAMS_DICT[dataset_name])
        mr_classifier_name = classifier.classifier.__class__.__name__
        os.makedirs(f'experiments/data/{dataset_name}/', exist_ok=True)
        classifier.save(f'experiments/data/{dataset_name}/{mr_classifier_name}.pkl')
    return classifier

def export(idx: int, explanation: Explanation, classifier: MinirocketClassifier, base_path: str, root_path: str):
    reference_policy = explanation.explanation['reference_policy']
    betas = explanation.explanation["coefficients"]
    betas_filename = f"{base_path}/betas_backpropagated_explanations_ref_policy_{reference_policy}_instance_{idx}.csv"
    if explanation.explanation['selected_features'] is not None:
        pd.Series(explanation.explanation['selected_features']).to_csv(f"{base_path}/selected_features_{idx}-top{len(explanation.explanation['selected_features'])}.csv", header=False)
        betas_filename = betas_filename.replace('.csv', f'-changeclass-top-{len(explanation.explanation["selected_features"])}.csv')

    pd.DataFrame(betas).T.to_csv(betas_filename, header=False)
    alphas = explanation.explanation["minirocket_coefficients"]
    pd.DataFrame(alphas).to_csv(f"{base_path}/alphas_mr_explanations_ref_policy_{reference_policy}_instance_{idx}.csv", header=False)
    instance = explanation.explanation["instance"]
    pd.DataFrame(instance).T.to_csv(f"{base_path}/instance_{idx}.csv", header=False)
    instance_transformed = explanation.explanation["instance_transformed"]
    pd.DataFrame(instance_transformed).to_csv(f"{base_path}/mr_instance_{idx}.csv", header=False)
    reference = explanation.explanation["reference"]
    pd.DataFrame(reference).T.to_csv(f"{base_path}/reference_ref_policy_{reference_policy}_for_instance_{idx}.csv", header=False)
    reference_transformed = explanation.explanation["reference_transformed"]
    pd.DataFrame(reference_transformed).to_csv(f"{base_path}/mr_reference_ref_policy_{reference_policy}_for_instance_{idx}.csv", header=False)
    biases = []
    dilations = []

    trace_ids = [i for i in range(len(explanation.explanation["traces"]))]

    for tridx in trace_ids:
        trace = explanation.explanation['traces'][tridx]
        base_mask, dilated_mask = get_dilated_triplet_array(mmv.get_feature_signature(tridx, classifier.minirocket_params))
        if not os.path.exists(f"{base_path}/base_mask_feature_{tridx}.csv"):
            pd.Series(base_mask).T.to_csv(f"{root_path}/base_mask_feature_{tridx}.csv", header=False)
        if not os.path.exists(f"{base_path}/dilated_mask_feature_{tridx}.csv"):
            pd.Series(dilated_mask).T.to_csv(f"{root_path}/dilated_mask_feature_{tridx}.csv", header=False)

        convolved_instance = trace['conv_sum']
        pd.DataFrame(convolved_instance).to_csv(f"{base_path}/convolved_instance_{idx}_feature_{tridx}.csv", header=False)
        convolved_instance_after_sigma = trace['sigma']
        pd.DataFrame(convolved_instance_after_sigma).to_csv(f"{base_path}/convolved_instance_after_sigma_instance_{idx}_feature_{tridx}.csv", header=False)
        biases.append(trace['bias_b'])
        dilations.append(trace['dilation'])

    biases_path = f"{root_path}/biases.csv"
    if not os.path.exists(biases_path):
        pd.Series(biases).T.to_csv(biases_path, header=False)

    dilations_path = f"{root_path}/dilations.csv"
    if not os.path.exists(dilations_path):
        pd.Series(dilations).T.to_csv(dilations_path, header=False)

    metadata_dict = {'instance_prediction': int(explanation.explanation['instance_prediction']),
                     'instance_predicted_probability': float(explanation.explanation['instance_logit']),
                     'reference_predicted_probability': float(explanation.explanation['reference_logit']),
                     'reference_prediction': int(explanation.explanation['reference_prediction']),
                     'instance_label': int(explanation.explanation['instance_label'])
                    }
    with open(f"{base_path}/metadata_ref_policy_{reference_policy}_instance_{idx}.json", "w") as fp:
        json.dump(metadata_dict, fp)

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
        compute_p2p_explanations,
        output_path,
        change_class_propagation,
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
    print("compute_p2p_explanations:", compute_p2p_explanations)
    print("output_path:", output_path)
    print("change_class_propagation:", change_class_propagation)

    os.makedirs(output_path, exist_ok=True)
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

                    for reference_policy in studied_reference_policies:
                        for idx in range(start, end_dataset):
                            print(f'Instance {idx} out of {end_dataset}')
                            measures_for_instance = {}
                            explanations_for_instance = {}
                            x_target = X_test[idx]
                            y_target = y_test[idx] if label == 'training' else y_test_pred[idx]
                            for reference_policy in studied_reference_policies:
                                explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                                explanation, explanation_p2p, segmented_explanation = (
                                    compute_explanations(x_target, y_target, classifier, explainer, configuration,
                                                         reference_policy,
                                                         compute_p2p_explanations=False,
                                                         compute_segmented_explanations=False,
                                                         top_alpha=topk, top_alpha_that_change_class=change_class_propagation)
                                )
                                instance_output_path = output_path + f'/{dataset_name}/{mr_classifier_name}/{explainer_method}/{label}/{idx}'
                                os.makedirs(instance_output_path, exist_ok=True)
                                export(idx, explanation, classifier, instance_output_path, output_path + f'/{dataset_name}/{mr_classifier_name}')














