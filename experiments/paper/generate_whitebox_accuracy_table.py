#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
DATA_ROOT = EXPERIMENTS_DIR / "data"
RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("JOBLIB_START_METHOD", "threading")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xai-for-minirocket-mpl-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "xai-for-minirocket-numba-cache"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

for path in (REPO_ROOT / "code", EXPERIMENTS_DIR, REPO_ROOT.parent / "tshap"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import (  # noqa: E402
    get_abnormal_hearbeat_for_classification,
    get_cognitive_circles_data_for_classification,
    get_double_freq_test_for_classification,
    get_forda_for_classification,
    get_handoutlines_for_classification,
    get_starlightcurves_for_classification,
)


def classifier_training_size(classifier) -> int:
    for attr in ("_X_train", "X_train"):
        X_train = getattr(classifier, attr, None)
        if X_train is not None:
            return len(X_train)
    raise ValueError("The classifier does not store its training set.")


DATASET_LOADERS = {
    "ford-a": lambda classifier: get_forda_for_classification(),
    "double-freq-test": lambda classifier: get_double_freq_test_for_classification(
        n_samples=classifier_training_size(classifier)
    ),
    "starlight-c1": lambda classifier: get_starlightcurves_for_classification("1"),
    "starlight-c2": lambda classifier: get_starlightcurves_for_classification("2"),
    "starlight-c3": lambda classifier: get_starlightcurves_for_classification("3"),
    "abnormal-heartbeat-c1": lambda classifier: get_abnormal_hearbeat_for_classification("1"),
    "cognitive-circles": lambda classifier: get_cognitive_circles_data_for_classification(
        str(REPO_ROOT / "data" / "cognitive-circles"),
        target_col="RealDifficulty",
        as_numpy=True,
    ),
    "handoutlines": lambda classifier: get_handoutlines_for_classification("1"),
}


def latest_shapelet_accuracy(tsv_path: Path) -> float:
    runs = pd.read_csv(tsv_path, sep="\t")
    required_columns = {"run_started_at_utc", "shapelet_decision_tree_accuracy"}
    missing_columns = required_columns - set(runs.columns)
    if missing_columns:
        raise ValueError(f"{tsv_path} is missing required columns: {sorted(missing_columns)}")

    if "classifier_name" in runs.columns:
        rf_runs = runs[runs["classifier_name"] == RANDOM_FOREST_CLASSIFIER]
        if not rf_runs.empty:
            runs = rf_runs

    runs = runs.dropna(subset=["run_started_at_utc", "shapelet_decision_tree_accuracy"]).copy()
    if runs.empty:
        raise ValueError(f"{tsv_path} has no complete shapelet accuracy rows")

    runs["run_started_at_utc"] = pd.to_datetime(runs["run_started_at_utc"], utc=True)
    runs["shapelet_decision_tree_accuracy"] = pd.to_numeric(
        runs["shapelet_decision_tree_accuracy"],
        errors="raise",
    )
    latest = runs.sort_values("run_started_at_utc").iloc[-1]
    return float(latest["shapelet_decision_tree_accuracy"])


def random_forest_test_accuracy(dataset_name: str, classifier_path: Path) -> float:
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"No dataset loader configured for {dataset_name}")
    if not classifier_path.exists():
        return np.nan

    with classifier_path.open("rb") as classifier_file:
        classifier = pickle.load(classifier_file)

    _, (X_test, y_test) = DATASET_LOADERS[dataset_name](classifier)
    return float(accuracy_score(y_test, classifier.predict(X_test)))


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def build_latex_table(rows: list[dict[str, float | str]], digits: int) -> str:
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & Random Forest & Shapelet-based \\",
        r"\midrule",
    ]

    for row in rows:
        random_forest = row["random_forest"]
        random_forest_value = "--" if pd.isna(random_forest) else f"{random_forest:.{digits}f}"
        latex.append(
            " & ".join(
                [
                    latex_escape(str(row["dataset"])),
                    random_forest_value,
                    f"{row['shapelet_based']:.{digits}f}",
                ]
            )
            + r" \\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Test accuracy of Random Forest MiniROCKET classifiers and the latest shapelet-based classifiers.}",
            r"\label{tab:whitebox_accuracy}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table comparing Random Forest and latest shapelet-based accuracies."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing per-dataset experiment data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "whitebox_accuracy_table.tex",
        help="Path where the LaTeX table should be written.",
    )
    parser.add_argument("--digits", type=int, default=3, help="Number of decimals to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tsv_paths = sorted(args.data_root.glob("*/whitebox_classifier_runs.tsv"))
    if not tsv_paths:
        raise RuntimeError(f"No whitebox_classifier_runs.tsv files found under {args.data_root}")

    rows = []
    for tsv_path in tsv_paths:
        dataset_name = tsv_path.parent.name
        classifier_path = tsv_path.parent / f"{RANDOM_FOREST_CLASSIFIER}.pkl"
        rows.append(
            {
                "dataset": dataset_name,
                "random_forest": random_forest_test_accuracy(dataset_name, classifier_path),
                "shapelet_based": latest_shapelet_accuracy(tsv_path),
            }
        )

    latex = build_latex_table(rows, args.digits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex + "\n")
    print(latex)
    print(f"\nLaTeX table written to {args.output}")


if __name__ == "__main__":
    main()
