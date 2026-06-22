#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import time
import types
import warnings
from pathlib import Path

# Must be set before importing joblib/sklearn.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

TSHAP_REPO_PATH = REPO_ROOT.parent / "tshap"
if TSHAP_REPO_PATH.exists() and str(TSHAP_REPO_PATH) not in sys.path:
    sys.path.append(str(TSHAP_REPO_PATH))


FEATURE_COUNTS = (500, 1000, 2000, 5000, 10000)
CLASSIFIER_NAMES = ("RandomForestClassifier", "LogisticRegression", "MLPClassifier")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest, Logistic Regression, and MLP classifiers on "
            "MiniROCKET features with several feature counts and report test accuracies."
        )
    )
    parser.add_argument(
        "--datasets",
        "-D",
        type=lambda s: s.split(","),
        default=["ford-a"],
        help="Comma-separated datasets to evaluate. Default: ford-a.",
    )
    parser.add_argument(
        "--feature-counts",
        "-F",
        type=lambda s: [int(x) for x in s.split(",")],
        default=list(FEATURE_COUNTS),
        help="Comma-separated MiniROCKET feature counts.",
    )
    parser.add_argument(
        "--classifiers",
        "-C",
        type=lambda s: s.split(","),
        default=list(CLASSIFIER_NAMES),
        help="Comma-separated classifier names.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="minirocket-feature-accuracy-results.csv",
        help="CSV file where results will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sklearn classifiers and NumPy.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of jobs for classifiers that support it.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for Logistic Regression and MLP.",
    )
    return parser.parse_args()


def get_dataset_fetchers(utils_module):
    cognitive_path = REPO_ROOT / "data" / "cognitive-circles"
    return {
        "ford-a": utils_module.get_forda_for_classification,
        "double-freq-test": lambda: utils_module.get_double_freq_test_for_classification(n_samples=200),
        "abnormal-heartbeat-c1": lambda: utils_module.get_abnormal_hearbeat_for_classification("1"),
        "starlight-c1": lambda: utils_module.get_starlightcurves_for_classification("1"),
        "starlight-c2": lambda: utils_module.get_starlightcurves_for_classification("2"),
        "starlight-c3": lambda: utils_module.get_starlightcurves_for_classification("3"),
        "cognitive-circles": lambda: utils_module.get_cognitive_circles_data_for_classification(
            str(cognitive_path),
            target_col="RealDifficulty",
            as_numpy=True,
        ),
        "handoutlines": lambda: utils_module.get_handoutlines_for_classification("1"),
    }


def make_classifier(name, random_state, n_jobs, max_iter):
    if name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )
    if name == "MLPClassifier":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=max_iter,
                early_stopping=True,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown classifier: {name}")


def as_float32_3d(X):
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n, C, L) or (n, L), got {X.shape}")
    return X.astype(np.float32, copy=False)


def train_and_score(mmv, X_train, y_train, X_test, y_test, feature_count, classifier_name, args):
    started = time.perf_counter()
    params = mmv.fit_minirocket_parameters(
        X_train,
        num_features=feature_count,
        diff=(classifier_name == "LogisticRegression"),
    )
    X_train_phi = mmv._transform_batch(X_train, parameters=params)
    X_test_phi = mmv._transform_batch(X_test, parameters=params)

    classifier = make_classifier(
        classifier_name,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X_train_phi, y_train)

    y_pred = classifier.predict(X_test_phi)
    elapsed = time.perf_counter() - started
    return accuracy_score(y_test, y_pred), elapsed, X_train_phi.shape[1]


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    import minirocket_multivariate_variable as mmv
    if "numdifftools" not in sys.modules:
        try:
            import numdifftools  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["numdifftools"] = types.ModuleType("numdifftools")
    import utils

    dataset_fetchers = get_dataset_fetchers(utils)
    unknown_datasets = sorted(set(args.datasets) - set(dataset_fetchers))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown_datasets)}")

    unknown_classifiers = sorted(set(args.classifiers) - set(CLASSIFIER_NAMES))
    if unknown_classifiers:
        raise ValueError(f"Unknown classifier(s): {', '.join(unknown_classifiers)}")

    rows = []
    for dataset_name in args.datasets:
        print(f"\nLoading dataset: {dataset_name}")
        (X_train, y_train), (X_test, y_test) = dataset_fetchers[dataset_name]()
        X_train = as_float32_3d(X_train)
        X_test = as_float32_3d(X_test)

        for feature_count in args.feature_counts:
            for classifier_name in args.classifiers:
                print(
                    f"Training {classifier_name} on {dataset_name} "
                    f"with {feature_count} MiniROCKET features..."
                )
                accuracy, elapsed, actual_feature_count = train_and_score(
                    mmv,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    feature_count,
                    classifier_name,
                    args,
                )
                row = {
                    "dataset": dataset_name,
                    "classifier": classifier_name,
                    "requested_features": feature_count,
                    "actual_features": actual_feature_count,
                    "test_accuracy": accuracy,
                    "elapsed_seconds": elapsed,
                }
                rows.append(row)
                print(
                    f"{dataset_name} | {classifier_name} | "
                    f"features={feature_count} actual={actual_feature_count} | "
                    f"accuracy={accuracy:.4f} | elapsed={elapsed:.2f}s"
                )

    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False)

    print("\nAccuracy table:")
    table = results.pivot_table(
        index=["dataset", "requested_features"],
        columns="classifier",
        values="test_accuracy",
    )
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
