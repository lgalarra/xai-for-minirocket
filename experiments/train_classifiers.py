#!/usr/bin/env python
import argparse
import copy
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Must be set before importing sklearn/joblib through compute_explanations.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.chdir(EXPERIMENTS_DIR)

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from classifier import MinirocketClassifier
import compute_explanations as compute_config
from compute_explanations import DATASET_FETCH_FUNCTIONS, MINIROCKET_PARAMS_DICT, MR_CLASSIFIERS
from export_data import DataExporter


def parse_csv_arg(value, allowed, name):
    if value is None:
        return list(allowed)

    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}. Available values: {list(allowed)}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train MiniROCKET classifiers for the datasets/models configured in "
            "compute_explanations.py and report train/test accuracies."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets. Defaults to every dataset in DATASET_FETCH_FUNCTIONS.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models. Defaults to every model in MR_CLASSIFIERS.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/classifier-accuracies.csv",
        help="CSV report path, relative to experiments/ unless absolute.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Preferred random_state for sklearn estimators that accept it. Use -1 to leave unset.",
    )
    parser.add_argument(
        "--retry-random-states",
        type=str,
        default="42,0,1,2,3,4,5,6,7,8,9",
        help=(
            "Comma-separated random_state candidates for estimators that accept random_state. "
            "The first non-degenerate classifier is saved."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Set on sklearn estimators that accept n_jobs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not retrain classifiers whose pickle already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed dataset/model instead of recording the error and continuing.",
    )
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Save classifiers even when train predictions contain fewer than two classes.",
    )
    return parser.parse_args()


def random_state_candidates(random_state, retry_random_states):
    if random_state is None:
        return [None]

    candidates = [random_state]
    for value in retry_random_states.split(","):
        value = value.strip()
        if value:
            candidates.append(int(value))
    return list(dict.fromkeys(candidates))


def make_base_estimator(model_name, random_state, n_jobs):
    estimator_cls = MR_CLASSIFIERS[model_name]
    signature = inspect.signature(estimator_cls)
    kwargs = {}
    if random_state is not None and "random_state" in signature.parameters:
        kwargs["random_state"] = random_state
    if n_jobs is not None and "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = n_jobs
    return estimator_cls(**kwargs)


def counts_as_json(values):
    unique, counts = np.unique(values, return_counts=True)
    return json.dumps({str(label): int(count) for label, count in zip(unique, counts)}, sort_keys=True)


def fit_and_score(dataset_name, model_name, X_train, y_train, X_test, y_test, random_state, n_jobs):
    minirocket_params = copy.deepcopy(MINIROCKET_PARAMS_DICT[dataset_name])
    minirocket_params["diff"] = model_name == "LogisticRegression"

    base_estimator = make_base_estimator(model_name, random_state, n_jobs)
    classifier = MinirocketClassifier(minirocket_features_classifier=base_estimator)

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(X_train, y_train, **minirocket_params)
    fit_seconds = time.perf_counter() - started

    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_pred_classes = np.unique(y_train_pred)
    test_pred_classes = np.unique(y_test_pred)

    warnings_list = []
    if train_pred_classes.size < 2:
        warnings_list.append("single_train_predicted_class")
    if test_pred_classes.size < 2:
        warnings_list.append("single_test_predicted_class")

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "fit_seconds": fit_seconds,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_true_counts": counts_as_json(y_train),
        "train_pred_counts": counts_as_json(y_train_pred),
        "test_true_counts": counts_as_json(y_test),
        "test_pred_counts": counts_as_json(y_test_pred),
        "minirocket_params": json.dumps(minirocket_params, sort_keys=True),
        "estimator_params": json.dumps(base_estimator.get_params(), default=str, sort_keys=True),
        "random_state": random_state,
        "warning": ";".join(warnings_list),
    }
    return classifier, row


def train_one(dataset_name, model_name, random_states, n_jobs, skip_existing, allow_degenerate):
    model_path = DataExporter.get_classifier_path(model_name, dataset_name)
    if skip_existing and os.path.exists(model_path):
        return {
            "dataset": dataset_name,
            "model": model_name,
            "status": "skipped",
            "model_path": model_path,
        }

    dataset_fetch_expression, _ = DATASET_FETCH_FUNCTIONS[dataset_name]
    (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_expression, compute_config.__dict__)

    accepts_random_state = "random_state" in inspect.signature(MR_CLASSIFIERS[model_name]).parameters
    candidates = random_states if accepts_random_state else [None]
    attempts = []
    best_classifier = None
    best_row = None
    for candidate_random_state in candidates:
        classifier, row = fit_and_score(
            dataset_name,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            candidate_random_state,
            n_jobs,
        )
        attempts.append({
            "random_state": candidate_random_state,
            "train_accuracy": row["train_accuracy"],
            "test_accuracy": row["test_accuracy"],
            "train_pred_counts": row["train_pred_counts"],
            "test_pred_counts": row["test_pred_counts"],
            "warning": row["warning"],
        })
        if best_row is None or row["test_accuracy"] > best_row["test_accuracy"]:
            best_classifier = classifier
            best_row = row
        if "single_train_predicted_class" not in row["warning"]:
            best_classifier = classifier
            best_row = row
            break

    best_row["attempts"] = json.dumps(attempts, sort_keys=True)
    best_row["model_path"] = model_path
    if "single_train_predicted_class" in best_row["warning"] and not allow_degenerate:
        best_row["status"] = "failed"
        best_row["error"] = (
            "All attempted fits predicted a single class on X_train; not saving because "
            "opposite-class reference policies would fail."
        )
        return best_row

    DataExporter.save_classifier(best_classifier, dataset_name)
    best_row["status"] = "trained"
    return best_row


def main():
    args = parse_args()
    datasets = parse_csv_arg(args.datasets, DATASET_FETCH_FUNCTIONS.keys(), "datasets")
    models = parse_csv_arg(args.models, MR_CLASSIFIERS.keys(), "models")
    random_state = None if args.random_state == -1 else args.random_state
    random_states = random_state_candidates(random_state, args.retry_random_states)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = EXPERIMENTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in datasets:
        for model_name in models:
            print(f"Training {model_name} on {dataset_name}...")
            try:
                row = train_one(
                    dataset_name,
                    model_name,
                    random_states,
                    args.n_jobs,
                    args.skip_existing,
                    args.allow_degenerate,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

            if row["status"] == "trained":
                print(
                    f"  train_accuracy={row['train_accuracy']:.6f} "
                    f"test_accuracy={row['test_accuracy']:.6f} "
                    f"warning={row.get('warning', '')}"
                )
            elif row["status"] == "skipped":
                print(f"  skipped existing {row['model_path']}")
            else:
                print(f"  failed: {row['error']}")

            pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Wrote accuracy report to {output_path}")


if __name__ == "__main__":
    main()
