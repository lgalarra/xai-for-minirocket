import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
DEFAULT_SEQUENTIAL_DIR = EXPERIMENTS_DIR / "official-results"
DEFAULT_PARALLEL_DIR = EXPERIMENTS_DIR / "official-results-parallel"
DEFAULT_OUTPUT = SCRIPT_DIR / "backprop_e2e_speedup_official_parallel.tex"
BACKPROP_RUNTIME_COLUMN = "runtimes-seconds"
E2E_RUNTIME_COLUMN = "runtimes-p2p-seconds"
COGNITIVE_CIRCLES_RUNTIME_COLUMN = "runtimes-segmented-n20-seconds"

METHOD_LABELS = {
    "extreme_feature_coalitions": "EFC",
    "gradients": "Gradients",
    "shap": "KernelSHAP",
    "stratoshap-k1": "ST-SHAP",
}

DATASET_RENAMES = {
    r"^abnormal-heartbeat-c.*": "abnormal-heartbeat",
    r"^starlight-c.*": "starlight",
}

REQUIRED_COLUMNS = {
    "dataset",
    "base_explainer",
    "mr_classifier",
    "reference_policy",
    "label",
    BACKPROP_RUNTIME_COLUMN,
    E2E_RUNTIME_COLUMN,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table with the speed-up of backpropagated "
            "explanations relative to end-to-end explanations."
        )
    )
    parser.add_argument(
        "--sequential-dir",
        type=Path,
        default=DEFAULT_SEQUENTIAL_DIR,
        help=f"Directory with sequential approximation CSVs. Default: {DEFAULT_SEQUENTIAL_DIR}",
    )
    parser.add_argument(
        "--parallel-dir",
        type=Path,
        default=DEFAULT_PARALLEL_DIR,
        help=f"Directory with parallel approximation CSVs. Default: {DEFAULT_PARALLEL_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File where the LaTeX table is written. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def parse_runtime_list(value):
    if pd.isna(value):
        return np.array([], dtype=float)
    return np.array(
        [float(item) for item in str(value).split(";") if item != ""],
        dtype=float,
    )


def normalize_dataset_name(dataset):
    normalized = str(dataset)
    for pattern, replacement in DATASET_RENAMES.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def load_results(data_dir):
    files = sorted(data_dir.glob("approximation-results*"))
    if not files:
        raise FileNotFoundError(f"No approximation result files found in {data_dir}")

    frames = []
    for csv_file in files:
        df = pd.read_csv(csv_file, low_memory=False).dropna(axis=1, how="all")
        if df.empty or not REQUIRED_COLUMNS.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f"No usable approximation result rows found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    return df


def filter_results(df):
    df = df[df["reference_policy"] != "counterfactual"].copy()
    unsupported_gradients = (
        (df["base_explainer"] == "gradients")
        & (~df["mr_classifier"].isin(("LogisticRegression", "MLPClassifier")))
    )
    return df[~unsupported_gradients].copy()


def deduplicate_runtime_profiles(df):
    identity_cols = [
        "dataset",
        "base_explainer",
        "mr_classifier",
        "reference_policy",
        "label",
    ]
    runtime_cols = [
        column
        for column in (
            BACKPROP_RUNTIME_COLUMN,
            E2E_RUNTIME_COLUMN,
            COGNITIVE_CIRCLES_RUNTIME_COLUMN,
        )
        if column in df.columns
    ]
    return df.drop_duplicates(subset=identity_cols + runtime_cols).copy()


def comparison_runtime_column(dataset):
    if dataset == "cognitive-circles":
        return COGNITIVE_CIRCLES_RUNTIME_COLUMN
    return E2E_RUNTIME_COLUMN


def valid_speedups(backprop_runtimes, comparison_runtimes):
    n_values = min(len(backprop_runtimes), len(comparison_runtimes))
    if n_values == 0:
        return np.array([], dtype=float)

    backprop = backprop_runtimes[:n_values]
    comparison = comparison_runtimes[:n_values]
    valid = (
        (backprop > 0)
        & (comparison > 0)
        & np.isfinite(backprop)
        & np.isfinite(comparison)
    )
    if not valid.any():
        return np.array([], dtype=float)
    return comparison[valid] / backprop[valid]


def aggregate_speedups(data_dir, source_label):
    df = load_results(data_dir)
    df = filter_results(df)
    df = deduplicate_runtime_profiles(df)
    df["backprop_runtime_list"] = df[BACKPROP_RUNTIME_COLUMN].apply(parse_runtime_list)
    df["comparison_runtime_column"] = df["dataset"].apply(comparison_runtime_column)
    df["comparison_runtime_list"] = df.apply(
        lambda row: parse_runtime_list(row.get(row["comparison_runtime_column"])),
        axis=1,
    )
    df["speedup_runs"] = df.apply(
        lambda row: valid_speedups(
            row["backprop_runtime_list"],
            row["comparison_runtime_list"],
        ),
        axis=1,
    )

    rows = []
    for (dataset, base_explainer), group in df.groupby(["dataset", "base_explainer"]):
        speedups = [
            values
            for values in group["speedup_runs"]
            if isinstance(values, np.ndarray) and len(values) > 0
        ]
        if not speedups:
            continue

        all_speedups = np.concatenate(speedups)
        rows.append(
            {
                "dataset": dataset,
                "base_explainer": base_explainer,
                "explainer_label": METHOD_LABELS.get(base_explainer, base_explainer),
                "comparison_runtime_column": ",".join(
                    sorted(group["comparison_runtime_column"].unique())
                ),
                f"{source_label}_mean": float(np.mean(all_speedups)),
                f"{source_label}_std": float(np.std(all_speedups)),
                f"{source_label}_n": int(len(all_speedups)),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "base_explainer"])


def format_float(value):
    if value == 0:
        return "0.00"
    if abs(value) < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def format_speedup(row, source_label):
    mean = row.get(f"{source_label}_mean")
    std = row.get(f"{source_label}_std")
    if pd.isna(mean):
        return "-"
    return rf"{format_float(mean)} $\pm$ {format_float(std)}"


def latex_escape(value):
    return str(value).replace("_", r"\_")


def merge_speedups(sequential, parallel):
    return (
        sequential.merge(
            parallel,
            on=["dataset", "base_explainer", "explainer_label"],
            how="outer",
        )
        .sort_values(["dataset", "base_explainer"])
        .reset_index(drop=True)
    )


def build_latex_table(merged):
    latex_lines = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l c c}",
        r"\toprule",
        (
            r"\textbf{Dataset} & \textbf{Base explainer} & "
            r"\textbf{sequential} & \textbf{parallel} \\"
        ),
        r"\midrule",
    ]

    for dataset, group in merged.groupby("dataset", sort=True):
        group = group.sort_values("explainer_label")
        first = True
        for _, row in group.iterrows():
            dataset_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{{latex_escape(dataset)}}}"
                if first
                else ""
            )
            first = False
            latex_lines.append(
                rf"{dataset_cell} & {latex_escape(row['explainer_label'])} & "
                rf"{format_speedup(row, 'sequential')} & "
                rf"{format_speedup(row, 'parallel')} \\"
            )
        latex_lines.append(r"\midrule")

    if latex_lines[-1] == r"\midrule":
        latex_lines[-1] = r"\bottomrule"
    else:
        latex_lines.append(r"\bottomrule")

    latex_lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{Speed-up of the backpropagation strategy with respect "
                r"to end-to-end explanations, computed as end-to-end runtime "
                r"divided by backpropagated runtime, except for cognitive-circles "
                r"where the segmented $n=20$ runtime is used because end-to-end "
                r"runtimes are unavailable. Values are mean $\pm$ standard "
                r"deviation over deduplicated runtime runs.}"
            ),
            r"\label{tab:backprop-e2e-speedup-sequential-parallel}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex_lines)


def main():
    args = parse_args()
    sequential = aggregate_speedups(args.sequential_dir, "sequential")
    parallel = aggregate_speedups(args.parallel_dir, "parallel")
    latex_table = build_latex_table(merge_speedups(sequential, parallel))

    print(latex_table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_table + "\n")


if __name__ == "__main__":
    main()
