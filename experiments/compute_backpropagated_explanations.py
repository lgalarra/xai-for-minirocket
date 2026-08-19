#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Must be set before importing joblib/sklearn workers.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

TSHAP_REPO_PATHS = [
    os.environ.get("TSHAP_REPO_PATH"),
    str(REPO_ROOT.parent / "tshap"),
    "/home/lgalarraga/tshap",
]
for tshap_repo_path in TSHAP_REPO_PATHS:
    if tshap_repo_path and os.path.isdir(tshap_repo_path) and tshap_repo_path not in sys.path:
        sys.path.insert(0, tshap_repo_path)

import minirocket_multivariate_variable as mmv

importlib.reload(mmv)

import export_data as export_data_module
from classifier import MinirocketClassifier
from explainer import Explanation, MinirocketExplainer, get_classifier_explainer, get_feature_signature, print_dilated_triplet_array
from export_data import DataExporter
from exputils import to_sep_list
from reference import REFERENCE_POLICIES
from utils import (
    COGNITIVE_CIRCLES_CHANNELS,
    cognitive_circles_get_sorted_channels_from_df,
    get_abnormal_hearbeat_for_classification,
    get_cognitive_circles_data_for_classification,
    get_double_freq_test_for_classification,
    get_forda_for_classification,
    get_handoutlines_for_classification,
    get_starlightcurves_for_classification,
)


LRP_DATA_ROOT = EXPERIMENTS_DIR / "lrp-data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute point-to-point, back_propagate_attribution_2, and "
            "back_propagate_attribution_lrp explanations."
        )
    )
    parser.add_argument("--dump-data", "-d", type=str, default="yes",
                        choices=["yes", "no", "true", "false", "1", "0"])
    parser.add_argument("--datasets", "-D", type=lambda s: s.split(","), default=None)
    parser.add_argument("--labels", "-L", type=lambda s: s.split(","), default=None)
    parser.add_argument("--models", "-M", type=lambda s: s.split(","), default=None)
    parser.add_argument("--explainers", "-E", type=lambda s: s.split(","), default=None)
    parser.add_argument("--propagate_top_features", "-t", type=int, default=None)
    parser.add_argument("--reference_policy", "-r", type=lambda s: s.split(","), default=None)
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--end", "-e", type=int, default=sys.maxsize - 1)
    parser.add_argument("--job_id", type=int, default=random.randint(1_000_000, 9_999_999))
    parser.add_argument("--run_id", type=int, default=random.randint(1_000_000, 9_999_999))
    parser.add_argument("--p2p_explanations", "-p", type=str, default="yes",
                        choices=["yes", "no", "true", "false", "1", "0"])
    parser.add_argument("--metric", "-m", type=str, default="euclidean",
                        help="Distance metric used to calculate reference instances: euclidean, pca-mr")
    parser.add_argument("--lrp-epsilon", type=float, default=1e-6)
    parser.add_argument("--lrp-stabilizer", type=str, default="paper", choices=["paper", "signed"])
    parser.add_argument("--n-jobs", type=int, default=-1)

    args = parser.parse_args()
    return (
        args.dump_data.lower() in ("yes", "true", "1"),
        args.datasets,
        args.labels,
        args.models,
        args.explainers,
        args.propagate_top_features,
        args.reference_policy,
        args.start,
        args.end,
        args.metric,
        args.p2p_explanations.lower() in ("yes", "true", "1"),
        args.lrp_epsilon,
        args.lrp_stabilizer,
        args.n_jobs,
    )


MR_CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "MLPClassifier": MLPClassifier,
}

DATASET_FETCH_FUNCTIONS = {
    "ford-a": ("get_forda_for_classification()", [("C", "Noise intensity")]),
    "double-freq-test": ("get_double_freq_test_for_classification(n_samples=250)", [("X", "Frequency")]),
    "abnormal-heartbeat-c1": ("get_abnormal_hearbeat_for_classification('1')", [("A", "Amplitude Change")]),
    "starlight-c1": ("get_starlightcurves_for_classification('1')", [("B", "Brightness")]),
    "starlight-c2": ("get_starlightcurves_for_classification('2')", [("B", "Brightness")]),
    "starlight-c3": ("get_starlightcurves_for_classification('3')", [("B", "Brightness")]),
    "cognitive-circles": (
        "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
        [
            (x, COGNITIVE_CIRCLES_CHANNELS[x])
            for x in cognitive_circles_get_sorted_channels_from_df(data_dir="../data/cognitive-circles")
        ],
    ),
    "handoutlines": ("get_handoutlines_for_classification('1')", [("X", "X")]),
}


def build_map_of_already_trained_classifiers(datasets: list, classifiers):
    return {
        dataset: {
            classifier: f'pickle.load(open("data/{dataset}/{classifier}.pkl", "rb"))'
            for classifier in classifiers
        }
        for dataset in datasets
    }


MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT = build_map_of_already_trained_classifiers(
    [
        "starlight-c1",
        "starlight-c2",
        "starlight-c3",
        "abnormal-heartbeat-c1",
        "ford-a",
        "cognitive-circles",
        "handoutlines",
        "double-freq-test",
    ],
    ["LogisticRegression", "RandomForestClassifier", "MLPClassifier"],
)

MINIROCKET_PARAMS_DICT = {
    "ford-a": {"num_features": 5000},
    "starlight-c1": {"num_features": 5000},
    "starlight-c2": {"num_features": 1000},
    "starlight-c3": {"num_features": 5000},
    "handoutlines": {"num_features": 5000},
    "cognitive-circles": {"num_features": 500},
    "abnormal-heartbeat-c1": {"num_features": 10000},
    "double-freq-test": {"num_features": 5000},
}


LRP_METADATA_COLUMNS = (
    export_data_module.METADATA_COLUMNS
    + [f"beta_lrp_{i}_attributions" for i in range(len(REFERENCE_POLICIES))]
)


def get_lrp_output_folder_for_export(dataset_name: str, mr_classifier_name: str, explainer_method: str,
                                     label_type: str, metric: str) -> str:
    return str(LRP_DATA_ROOT / dataset_name / mr_classifier_name / explainer_method / label_type / metric)


class BackpropagatedDataExporter:
    METADATA_FILE = "metadata.csv"

    def __init__(self, dataset_name: str, mr_classifier_name: str, explainer_method: str, label_type: str, metric: str):
        self.output_path = get_lrp_output_folder_for_export(
            dataset_name, mr_classifier_name, explainer_method, label_type, metric
        )
        self.metadata_file = f"{self.output_path}/{BackpropagatedDataExporter.METADATA_FILE}"
        self.output_dataset_path = str(LRP_DATA_ROOT / dataset_name)
        self.dataset_name = dataset_name
        self.mr_classifier_name = mr_classifier_name
        self.explainer_method = explainer_method
        self.label_type = label_type
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def get_metadata_filename_for_reference_policy(reference_policy: str) -> str:
        base, ext = os.path.splitext(BackpropagatedDataExporter.METADATA_FILE)
        return f"{base}_ref_policy_{reference_policy}{ext}"

    def get_metadata_file_for_reference_policy(self, reference_policy: str) -> str:
        return f"{self.output_path}/{self.get_metadata_filename_for_reference_policy(reference_policy)}"

    def prepare_export(self, dataset_fetch_info: tuple, studied_reference_policies=REFERENCE_POLICIES):
        (_, features) = dataset_fetch_info
        for feature_name, _ in features:
            os.makedirs(f"{self.output_path}/{feature_name}", exist_ok=True)
            os.makedirs(f"{self.output_dataset_path}/{feature_name}", exist_ok=True)
        os.makedirs(f"{self.output_dataset_path}/{self.mr_classifier_name}/{self.explainer_method}", exist_ok=True)

        for reference_policy in export_data_module.get_reference_policy_list(studied_reference_policies):
            pd.DataFrame({col: [] for col in LRP_METADATA_COLUMNS}).to_csv(
                self.get_metadata_file_for_reference_policy(reference_policy),
                mode="w",
                index=False,
                header=True,
            )

    def export_instance_and_explanations(self, instance_id, y_i, features: list, explanations_dict: dict,
                                         studied_reference_policies=REFERENCE_POLICIES, topk=None):
        some_reference_policy = next(iter(explanations_dict))
        explanation_bp2 = explanations_dict[some_reference_policy][0]
        instance = explanation_bp2.get_instance()

        mr_filename = f"{self.output_dataset_path}/mr_embeddings_instance_{instance_id}.csv"
        if not os.path.exists(mr_filename):
            pd.Series(explanation_bp2.explanation["instance_transformed"]).to_csv(mr_filename, header=False)

        for reference_policy in export_data_module.get_reference_policy_list(studied_reference_policies):
            metadata_rows = []
            metadata_file = self.get_metadata_file_for_reference_policy(reference_policy)
            reference_policy_idx = REFERENCE_POLICIES.index(reference_policy)
            explanation_bp2, explanation_p2p, explanation_lrp = explanations_dict[reference_policy]
            betas_bp2 = explanation_bp2.get_attributions_in_original_dimensions()
            betas_lrp = explanation_lrp.get_attributions_in_original_dimensions()
            betas_p2p = (
                None if explanation_p2p is None
                else explanation_p2p.get_attributions_in_original_dimensions()
            )
            reference = explanation_bp2.explanation["reference"]

            for channel_idx, channel in enumerate(instance):
                row = {col: None for col in LRP_METADATA_COLUMNS}
                feature_name = features[channel_idx][0]
                reference_code = hashlib.md5(reference[channel_idx].data.tobytes()).hexdigest()
                reference_filename = f"{self.output_path}/{feature_name}/{feature_name}_reference_{reference_code}.csv"
                if not os.path.exists(reference_filename):
                    pd.Series(reference[channel_idx]).to_csv(reference_filename, header=False)

                if topk is None:
                    attr_bp2_filename = f"{self.output_path}/{feature_name}/beta_instance_{instance_id}_{reference_policy_idx}.csv"
                    attr_lrp_filename = f"{self.output_path}/{feature_name}/betalrp_instance_{instance_id}_{reference_policy_idx}.csv"
                else:
                    attr_bp2_filename = f"{self.output_path}/{feature_name}/beta_instance_{instance_id}_{reference_policy_idx}-topk-{topk}.csv"
                    attr_lrp_filename = f"{self.output_path}/{feature_name}/betalrp_instance_{instance_id}_{reference_policy_idx}-topk-{topk}.csv"

                pd.Series(betas_bp2[channel_idx]).to_csv(attr_bp2_filename, header=False)
                pd.Series(betas_lrp[channel_idx]).to_csv(attr_lrp_filename, header=False)

                attr_p2p_filename = None
                if betas_p2p is not None:
                    attr_p2p_filename = f"{self.output_path}/{feature_name}/betap2p_instance_{instance_id}_{reference_policy_idx}.csv"
                    pd.Series(betas_p2p[channel_idx]).to_csv(attr_p2p_filename, header=False)

                channel_filename = f"{self.output_dataset_path}/{feature_name}/{feature_name}_instance_{instance_id}.csv"
                if not os.path.exists(channel_filename):
                    pd.Series(channel).to_csv(channel_filename, header=False)

                row["instance_id"] = instance_id
                row["series"] = channel_filename
                row["label"] = y_i
                row["label_type"] = self.label_type
                row["label_probability"] = explanation_bp2.explanation["instance_logit"]
                row["channel"] = export_data_module.CHANNELS[self.dataset_name][channel_idx]
                row["group"] = export_data_module.get_group_id(self.dataset_name, instance_id)
                row["annotation"] = export_data_module.get_annotation(self.dataset_name, instance_id)
                row[f"reference_{reference_policy_idx}"] = reference_filename
                row[f"reference_{reference_policy_idx}_label"] = explanation_bp2.explanation["reference_prediction"]
                row[f"reference_{reference_policy_idx}_label_probability"] = explanation_bp2.explanation["reference_logit"]
                row[f"beta_{reference_policy_idx}_attributions"] = attr_bp2_filename
                row[f"beta_p2p_{reference_policy_idx}_attributions"] = attr_p2p_filename
                row[f"beta_lrp_{reference_policy_idx}_attributions"] = attr_lrp_filename
                metadata_rows.append(row)

            print(f"Flushing {len(metadata_rows)} rows in {metadata_file}")
            pd.DataFrame(metadata_rows, columns=LRP_METADATA_COLUMNS).to_csv(
                metadata_file,
                mode="a",
                index=False,
                header=False,
            )

        alphas = explanation_bp2.explanation["minirocket_coefficients"]
        alphas_file = (
            f"{self.output_dataset_path}/{self.mr_classifier_name}/"
            f"{self.explainer_method}/alphas_instance_{instance_id}.csv"
        )
        if not os.path.exists(alphas_file):
            pd.Series(alphas).to_csv(alphas_file, header=False)

    def export_metametadata(self):
        metametadata = {}
        for i in range(len(export_data_module.UNITS[self.dataset_name])):
            metametadata["units"] = export_data_module.DESCRIPTIONS[self.dataset_name][i]
            metametadata["references"] = export_data_module.REFERENCE_POLICIES_LABELS
            metametadata["classes"] = export_data_module.CLASSES[self.dataset_name]
        with open(f"{self.output_dataset_path}/metametadata.json", "w") as f:
            import json

            json.dump(metametadata, f)


def compare_explanations(left: Explanation, right: Explanation) -> float:
    res = kendalltau(left.get_attributions_as_single_vector(), right.get_attributions_as_single_vector())
    return 0.0 if np.isnan(res.statistic) else float(res.statistic)


def _select_alphas(alphas, reference_logit, instance_logit, top_alpha, minirocket_params):
    if top_alpha is None:
        return alphas, None, None

    selected_features = (-alphas).argsort()[:top_alpha] if reference_logit < instance_logit else alphas.argsort()[:top_alpha]
    if len(selected_features) > 0:
        print("Features:", print_dilated_triplet_array(get_feature_signature(selected_features[0], minirocket_params)))
    mask = np.zeros_like(alphas, dtype=np.float64)
    for mid in selected_features:
        mask[mid] = alphas[mid]
    return mask, mask, selected_features


def compute_backpropagated_explanations(
    x_target,
    y_target,
    classifier: MinirocketClassifier,
    explainer: MinirocketExplainer,
    configuration: tuple,
    reference_policy: str,
    *,
    compute_p2p_explanations=True,
    top_alpha=None,
    lrp_epsilon=1e-6,
    lrp_stabilizer="paper",
    n_jobs=-1,
):
    (dataset_name, _, explainer_method, _) = configuration

    setup_start = time.perf_counter()
    reference = explainer.get_reference(x_target, y_target, reference_policy, dataset_name=dataset_name)
    is_multichannel = x_target.shape[1] > 1
    out_x = mmv.transform_prime(x_target, parameters=explainer.minirocket_params)
    reference_mr = mmv.transform_prime(reference, parameters=explainer.minirocket_params)

    classifier_explainer_fn = get_classifier_explainer(
        explainer_method,
        lambda x: explainer.minirocket_classifier.predict_proba(x)[:, y_target],
        X_background=np.array([reference_mr["phi"][0]]),
        target=out_x["phi"][0],
    )
    y_ref_pred = explainer.minirocket_classifier.predict(reference_mr["phi"][0].reshape(1, -1))[0]
    y_pred = explainer.minirocket_classifier.predict(out_x["phi"][0].reshape(1, -1))[0]
    reference_logit = explainer.minirocket_classifier.predict_proba(reference_mr["phi"][0].reshape(1, -1))[0][y_pred]
    instance_logit = explainer.minirocket_classifier.predict_proba(out_x["phi"][0].reshape(1, -1))[0][y_pred]

    alphas = classifier_explainer_fn(out_x["phi"])
    if alphas.shape[0] == 1:
        alphas = alphas[0]
    alphas_to_backpropagate, mask, selected_features = _select_alphas(
        alphas, reference_logit, instance_logit, top_alpha, explainer.minirocket_params
    )
    setup_elapsed = time.perf_counter() - setup_start

    start_bp2 = time.perf_counter()
    beta_bp2 = mmv.back_propagate_attribution_2(
        alphas_to_backpropagate,
        out_x["traces"],
        x_target,
        reference,
        per_channel=is_multichannel,
        params=explainer.minirocket_params,
        n_jobs=n_jobs,
    )
    time_bp2 = setup_elapsed + (time.perf_counter() - start_bp2)
    print("Time elapsed (back_propagate_attribution_2): ", time_bp2)

    start_lrp = time.perf_counter()
    beta_lrp = mmv.back_propagate_attribution_lrp(
        alphas_to_backpropagate,
        out_x["traces"],
        x_target,
        epsilon=lrp_epsilon,
        stabilizer=lrp_stabilizer,
        per_channel=is_multichannel,
        n_jobs=n_jobs,
    )
    time_lrp = setup_elapsed + (time.perf_counter() - start_lrp)
    print("Time elapsed (back_propagate_attribution_lrp): ", time_lrp)

    base_payload = {
        "minirocket_coefficients": alphas,
        "instance": x_target,
        "instance_transformed": out_x["phi"][0],
        "reference_transformed": reference_mr["phi"][0],
        "traces": out_x["traces"],
        "reference": reference,
        "reference_traces": reference_mr["traces"],
        "instance_label": y_target,
        "reference_prediction": y_ref_pred,
        "instance_prediction": y_pred,
        "reference_logit": reference_logit,
        "instance_logit": instance_logit,
        "reference_policy": reference_policy,
        "backpropagated_features": mask,
        "selected_features": selected_features,
    }

    payload_bp2 = dict(base_payload, coefficients=beta_bp2, time_elapsed=time_bp2,
                       backpropagation_method="back_propagate_attribution_2")
    payload_lrp = dict(base_payload, coefficients=beta_lrp, time_elapsed=time_lrp,
                       backpropagation_method="back_propagate_attribution_lrp",
                       lrp_epsilon=lrp_epsilon, lrp_stabilizer=lrp_stabilizer)

    explanation_bp2 = Explanation(payload_bp2)
    explanation_lrp = Explanation(payload_lrp)
    if compute_p2p_explanations:
        explanation_p2p = classifier.explain_instances(
            x_target,
            reference,
            explainer=explainer_method,
            reference_policy=reference_policy,
        )
    else:
        explanation_p2p = None

    return explanation_bp2, explanation_p2p, explanation_lrp


def get_classifier(mr_classifier_name: str, dataset_name: str, X_train, y_train) -> MinirocketClassifier:
    mr_params = copy.deepcopy(MINIROCKET_PARAMS_DICT[dataset_name])
    mr_params["diff"] = mr_classifier_name == "LogisticRegression"
    model_path = DataExporter.get_classifier_path(mr_classifier_name, dataset_name)
    if os.path.exists(model_path):
        print(f"Loading existing classifier at {model_path}")
        classifier = eval(MR_ALREADY_TRAINED_CLASSIFIERS_FETCH_DICT[dataset_name][mr_classifier_name])
        mmv.MINIROCKET_PARAMETERS = classifier.minirocket_params
        classifier.ensure_training_transform_and_pca()
    else:
        print("Training new classifier...")
        mr_classifier = MR_CLASSIFIERS[mr_classifier_name]()
        classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
        classifier.fit(X_train, y_train, **mr_params)
        DataExporter.save_classifier(classifier, dataset_name)
    return classifier


def supports_explainer_for_classifier(explainer_method, classifier_name):
    return explainer_method != "gradients" or classifier_name == "LogisticRegression"


def _empty_measures():
    return {
        "kendall_bp2_p2p": [],
        "kendall_lrp_p2p": [],
        "kendall_bp2_lrp": [],
        "runtime_bp2": [],
        "runtime_lrp": [],
        "runtime_p2p": [],
        "complexity_bp2": [],
        "complexity_lrp": [],
        "complexity_p2p": [],
        "local_accuracy_bp2": 0,
        "local_accuracy_lrp": 0,
        "error_bp2": [],
        "error_lrp": [],
    }


def _update_measures(measures, explanation_bp2, explanation_p2p, explanation_lrp):
    measures["runtime_bp2"].append(explanation_bp2.get_runtime())
    measures["runtime_lrp"].append(explanation_lrp.get_runtime())
    measures["runtime_p2p"].append(-1.0 if explanation_p2p is None else explanation_p2p.get_runtime())
    measures["complexity_bp2"].append(np.count_nonzero(explanation_bp2.explanation["coefficients"]))
    measures["complexity_lrp"].append(np.count_nonzero(explanation_lrp.explanation["coefficients"]))
    measures["complexity_p2p"].append(
        -1.0 if explanation_p2p is None else np.count_nonzero(explanation_p2p.explanation["coefficients"])
    )
    measures["kendall_bp2_p2p"].append(0.0 if explanation_p2p is None else compare_explanations(explanation_bp2, explanation_p2p))
    measures["kendall_lrp_p2p"].append(0.0 if explanation_p2p is None else compare_explanations(explanation_lrp, explanation_p2p))
    measures["kendall_bp2_lrp"].append(compare_explanations(explanation_bp2, explanation_lrp))

    respects_bp2, delta_bp2 = explanation_bp2.check_explanation_local_accuracy_wrt_minirocket()
    respects_lrp, delta_lrp = explanation_lrp.check_explanation_local_accuracy_wrt_minirocket()
    measures["local_accuracy_bp2"] += 1 if respects_bp2 else 0
    measures["local_accuracy_lrp"] += 1 if respects_lrp else 0
    measures["error_bp2"].append(delta_bp2)
    measures["error_lrp"].append(delta_lrp)


def _add_measure_columns(row, prefix, values):
    row[f"{prefix}-seconds"] = to_sep_list(values)
    row[f"{prefix}-mean"] = np.mean(values)
    row[f"{prefix}-std"] = np.std(values)


def _add_complexity_columns(row, prefix, values):
    row[prefix] = to_sep_list(values)
    row[f"{prefix}-mean"] = np.mean(values)
    row[f"{prefix}-std"] = np.std(values)


def _result_row(configuration, reference_policy, measures):
    dataset_name, mr_classifier_name, explainer_method, label = configuration
    n = max(1, len(measures["runtime_bp2"]))
    row = {
        "timestamp": pd.Timestamp.now(),
        "base_explainer": explainer_method,
        "mr_classifier": mr_classifier_name,
        "reference_policy": reference_policy,
        "label": label,
        "dataset": dataset_name,
        "local_accuracy_bp2": measures["local_accuracy_bp2"] / n,
        "local_accuracy_lrp": measures["local_accuracy_lrp"] / n,
        "error_bp2": to_sep_list(measures["error_bp2"]),
        "error_lrp": to_sep_list(measures["error_lrp"]),
    }
    _add_measure_columns(row, "runtimes-bp2", measures["runtime_bp2"])
    _add_measure_columns(row, "runtimes-lrp", measures["runtime_lrp"])
    _add_measure_columns(row, "runtimes-p2p", measures["runtime_p2p"])
    _add_complexity_columns(row, "complexity-bp2", measures["complexity_bp2"])
    _add_complexity_columns(row, "complexity-lrp", measures["complexity_lrp"])
    _add_complexity_columns(row, "complexity-p2p", measures["complexity_p2p"])
    _add_complexity_columns(row, "kendall-bp2-p2p", measures["kendall_bp2_p2p"])
    _add_complexity_columns(row, "kendall-lrp-p2p", measures["kendall_lrp_p2p"])
    _add_complexity_columns(row, "kendall-bp2-lrp", measures["kendall_bp2_lrp"])
    return row


if __name__ == "__main__":
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
        lrp_epsilon,
        lrp_stabilizer,
        n_jobs,
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
    print("lrp_epsilon:", lrp_epsilon)
    print("lrp_stabilizer:", lrp_stabilizer)
    print("n_jobs:", n_jobs)

    LABELS = ["predicted", "training"]
    EXPLAINERS = ["extreme_feature_coalitions", "shap", "gradients", "stratoshap-k1"]

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

    results_dir = EXPERIMENTS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "backpropagated-explanation-results.csv"
    if datasets is not None:
        output_file = Path(str(output_file).replace(".csv", f"-{','.join(datasets)}.csv"))
    if labels is not None:
        output_file = Path(str(output_file).replace(".csv", f"-{','.join(labels)}.csv"))
    if models is not None:
        output_file = Path(str(output_file).replace(".csv", f"-{','.join(models)}.csv"))
    if reference_policy is not None:
        output_file = Path(str(output_file).replace(".csv", f"-{','.join(studied_reference_policies)}.csv"))
    if explainers is not None:
        output_file = Path(str(output_file).replace(".csv", f"-{','.join(explainers)}.csv"))
    if topk is not None:
        output_file = Path(str(output_file).replace(".csv", f"topk-{topk}.csv"))
        BackpropagatedDataExporter.METADATA_FILE = BackpropagatedDataExporter.METADATA_FILE.replace(".csv", f"-{topk}.csv")
    if end != sys.maxsize - 1:
        output_file = Path(str(output_file).replace(".csv", f"-{start}-{end}.csv"))
        BackpropagatedDataExporter.METADATA_FILE = BackpropagatedDataExporter.METADATA_FILE.replace(".csv", f"-{start}-{end}.csv")
    if not should_export_data:
        output_file = Path(str(output_file).replace(".csv", "-NOTDUMPED.csv"))
    output_file = Path(str(output_file).replace(".csv", f"metric-{metric}.csv"))

    result_columns = [
        "timestamp", "base_explainer", "mr_classifier", "reference_policy", "label", "dataset",
        "local_accuracy_bp2", "local_accuracy_lrp", "error_bp2", "error_lrp",
        "runtimes-bp2-seconds", "runtimes-bp2-mean", "runtimes-bp2-std",
        "runtimes-lrp-seconds", "runtimes-lrp-mean", "runtimes-lrp-std",
        "runtimes-p2p-seconds", "runtimes-p2p-mean", "runtimes-p2p-std",
        "complexity-bp2", "complexity-bp2-mean", "complexity-bp2-std",
        "complexity-lrp", "complexity-lrp-mean", "complexity-lrp-std",
        "complexity-p2p", "complexity-p2p-mean", "complexity-p2p-std",
        "kendall-bp2-p2p", "kendall-bp2-p2p-mean", "kendall-bp2-p2p-std",
        "kendall-lrp-p2p", "kendall-lrp-p2p-mean", "kendall-lrp-p2p-std",
        "kendall-bp2-lrp", "kendall-bp2-lrp-mean", "kendall-bp2-lrp-std",
    ]
    pd.DataFrame(columns=result_columns).to_csv(output_file, mode="w", index=False, header=True)

    exporters_dict = {}
    MinirocketExplainer.REFERENCE_DISTANCE = metric

    for dataset_name, (dataset_fetch_function, features) in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        end_dataset = min(len(X_test), end)
        for mr_classifier_name in models:
            classifier = get_classifier(mr_classifier_name, dataset_name, X_train, y_train)
            y_test_pred = classifier.predict(X_test)
            print(f"Accuracy on test set ({dataset_name}): {accuracy_score(y_test, y_test_pred)}")

            for explainer_method in explainers:
                if not supports_explainer_for_classifier(explainer_method, mr_classifier_name):
                    print(
                        f"Skipping {explainer_method} for {mr_classifier_name}: "
                        "gradients are only supported for LogisticRegression"
                    )
                    continue

                for label in LABELS:
                    configuration = (dataset_name, mr_classifier_name, explainer_method, label)
                    print(f"Evaluating configuration {configuration}")
                    if should_export_data:
                        exporter = BackpropagatedDataExporter(
                            dataset_name, mr_classifier_name, explainer_method, label, metric
                        )
                        exporter.prepare_export(
                            DATASET_FETCH_FUNCTIONS[dataset_name],
                            studied_reference_policies=studied_reference_policies,
                        )
                        exporters_dict[configuration] = exporter

                    measures_by_reference_policy = {
                        rp: _empty_measures()
                        for rp in studied_reference_policies
                    }

                    for idx in range(start, end_dataset):
                        print(f"Instance {idx} out of {end_dataset - start} (end={end_dataset})")
                        explanations_for_instance = {}
                        x_target = X_test[idx]
                        y_target = y_test[idx] if label == "training" else y_test_pred[idx]

                        for rp in studied_reference_policies:
                            explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                            explanation_bp2, explanation_p2p, explanation_lrp = compute_backpropagated_explanations(
                                x_target,
                                y_target,
                                classifier,
                                explainer,
                                configuration,
                                rp,
                                compute_p2p_explanations=(topk is None and compute_p2p_explanations),
                                top_alpha=topk,
                                lrp_epsilon=lrp_epsilon,
                                lrp_stabilizer=lrp_stabilizer,
                                n_jobs=n_jobs,
                            )
                            explanations_for_instance[rp] = (explanation_bp2, explanation_p2p, explanation_lrp)
                            _update_measures(
                                measures_by_reference_policy[rp],
                                explanation_bp2,
                                explanation_p2p,
                                explanation_lrp,
                            )

                        if should_export_data:
                            print(f"Exporting instance {idx} to {exporter.output_path} ({configuration})")
                            exporter.export_instance_and_explanations(
                                idx,
                                y_target,
                                features,
                                explanations_for_instance,
                                studied_reference_policies=studied_reference_policies,
                                topk=topk,
                            )

                    for rp, measures in measures_by_reference_policy.items():
                        row = _result_row(configuration, rp, measures)
                        print(pd.DataFrame([row], columns=result_columns))
                        pd.DataFrame([row], columns=result_columns).to_csv(
                            output_file,
                            mode="a",
                            index=False,
                            header=False,
                        )

    if should_export_data:
        for _, exporter in exporters_dict.items():
            exporter.export_metametadata()
