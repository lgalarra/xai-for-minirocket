import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "official-results"
COMPUTE_EXPLANATIONS = SCRIPT_DIR.parent / "compute_explanations.py"

METHOD_LABELS = {
    "extreme_feature_coalitions": "EFC",
    "shap": "KernelSHAP",
    "stratoshap-k1": "ST-SHAP",
    "gradients": "Gradients",
}

DATASET_RENAMES = {
    r"^starlight-c.*": "starlight",
    r"^abnormal-heartbeat-c.*": "abnormal-heartbeat",
}

SEGMENTED_CONFIGS = (10, 20, 50, 100)
TSHAP_CONFIGS = (
    (10, 5),
    (10, 20),
    (15, 5),
    (15, 20),
    (20, 5),
    (20, 20),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX runtime-ratio table from official approximation result CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory with approximation-results-* CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file where the LaTeX table is written. The table is always printed.",
    )
    return parser.parse_args()


def parse_list(value):
    if pd.isna(value):
        return np.array([], dtype=float)
    return np.array(
        [float(v) for v in str(value).split(";") if v != ""],
        dtype=float,
    )


def load_minirocket_features():
    tree = ast.parse(COMPUTE_EXPLANATIONS.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MINIROCKET_PARAMS_DICT":
                    params = ast.literal_eval(node.value)
                    return {
                        dataset: int(values["num_features"])
                        for dataset, values in params.items()
                    }
    raise ValueError(f"MINIROCKET_PARAMS_DICT not found in {COMPUTE_EXPLANATIONS}")


def runtime_prefix_for_segmented(num_segments):
    return "runtimes-segmented" if num_segments == 10 else f"runtimes-segmented-n{num_segments}"


def runtime_prefix_for_tshap(window_size_percent, stride):
    return f"runtimes-tshap-w{window_size_percent}_s{stride}"


def variant_specs():
    specs = [("p2p", "P2P", "runtimes-p2p")]
    specs.extend(
        (
            f"seg{num_segments}",
            f"Seg. {num_segments}",
            runtime_prefix_for_segmented(num_segments),
        )
        for num_segments in SEGMENTED_CONFIGS
    )
    specs.extend(
        (
            f"tshap_w{window_size_percent}_s{stride}",
            rf"t-SHAP {window_size_percent}/{stride}",
            runtime_prefix_for_tshap(window_size_percent, stride),
        )
        for window_size_percent, stride in TSHAP_CONFIGS
    )
    return specs


def load_results(data_dir):
    files = sorted(data_dir.glob("approximation-results-*"))
    if not files:
        raise FileNotFoundError(f"No approximation-results-* files found in {data_dir}")

    frames = [
        pd.read_csv(csv_file, low_memory=False).dropna(axis=1, how="all")
        for csv_file in files
    ]
    df = pd.concat(frames, ignore_index=True)
    for pattern, replacement in DATASET_RENAMES.items():
        df["dataset"] = df["dataset"].str.replace(pattern, replacement, regex=True)
    return df


def infer_observation_counts(df):
    complexity_cols = [
        col
        for col in df.columns
        if col.startswith("complexity") and col.endswith("-mean")
    ]
    for col in complexity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.groupby("dataset")[complexity_cols]
        .max()
        .max(axis=1)
        .round()
        .astype("Int64")
        .astype(object)
        .to_dict()
    )


def deduplicate_runtime_profiles(df, specs):
    runtime_cols = [
        f"{prefix}-seconds"
        for _, _, prefix in specs
        if f"{prefix}-seconds" in df.columns
    ]
    identity_cols = [
        col
        for col in ("dataset", "base_explainer", "mr_classifier", "reference_policy", "label")
        if col in df.columns
    ]
    return df.drop_duplicates(subset=identity_cols + runtime_cols).copy()


def valid_ratio(base_runtimes, variant_runtimes):
    n_values = min(len(base_runtimes), len(variant_runtimes))
    if n_values == 0:
        return np.array([], dtype=float)

    base = base_runtimes[:n_values]
    variant = variant_runtimes[:n_values]
    valid = (base > 0) & (variant > 0) & np.isfinite(base) & np.isfinite(variant)
    if not valid.any():
        return np.array([], dtype=float)
    return variant[valid] / base[valid]


def compute_ratios(df, specs):
    df["base_runtime_list"] = df["runtimes-seconds"].apply(parse_list)

    for key, _, prefix in specs:
        seconds_col = f"{prefix}-seconds"
        if seconds_col not in df.columns:
            df[f"{key}_ratio_runs"] = [np.array([], dtype=float) for _ in range(len(df))]
            continue
        df[f"{key}_runtime_list"] = df[seconds_col].apply(parse_list)
        df[f"{key}_ratio_runs"] = df.apply(
            lambda row: valid_ratio(
                row["base_runtime_list"],
                row[f"{key}_runtime_list"],
            ),
            axis=1,
        )

    rows = []
    for (dataset, base_explainer), group in df.groupby(["dataset", "base_explainer"]):
        row = {"dataset": dataset, "base_explainer": base_explainer}
        for key, _, _ in specs:
            ratios = [
                values
                for values in group[f"{key}_ratio_runs"]
                if isinstance(values, np.ndarray) and len(values) > 0
            ]
            if ratios:
                all_ratios = np.concatenate(ratios)
                row[f"{key}_mean"] = float(np.mean(all_ratios))
                row[f"{key}_std"] = float(np.std(all_ratios))
                row[f"{key}_min"] = float(np.min(all_ratios))
                row[f"{key}_max"] = float(np.max(all_ratios))
            else:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_std"] = np.nan
                row[f"{key}_min"] = np.nan
                row[f"{key}_max"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "base_explainer"])


def format_metadata_ratio(observations, features):
    if observations is None or features is None or pd.isna(observations) or pd.isna(features):
        return "-"
    if features == 0:
        return "-"
    return format_float(float(observations) / float(features))


def format_ratio(row, key):
    mean = row[f"{key}_mean"]
    std = row[f"{key}_std"]
    if pd.isna(mean):
        return "-"
    return rf"{format_float(mean)} $\pm$ {format_float(std)}"


def format_float(value):
    if value == 0:
        return "0.00"
    if abs(value) < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def latex_escape(value):
    return str(value).replace("_", r"\_")


def build_latex_table(agg, specs, observation_counts, minirocket_features):
    latex_lines = []
    latex_lines.append(r"\begin{table}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\scriptsize")
    latex_lines.append(
        rf"\begin{{tabular}}{{l l r {' '.join('c' for _ in specs)}}}"
    )
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"\textbf{Dataset} & \textbf{Method} & "
        r"\textbf{Obs./MR feat.} & "
        + " & ".join(rf"\textbf{{{label}}}" for _, label, _ in specs)
        + r" \\"
    )
    latex_lines.append(r"\midrule")

    for dataset, group in agg.groupby("dataset", sort=True):
        first = True
        group = group.sort_values("base_explainer")
        for _, row in group.iterrows():
            method = METHOD_LABELS.get(row["base_explainer"], latex_escape(row["base_explainer"]))
            dataset_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{{latex_escape(dataset)}}}"
                if first
                else ""
            )
            metadata_ratio_cell = (
                format_metadata_ratio(
                    observation_counts.get(dataset),
                    minirocket_features.get(dataset),
                )
                if first
                else ""
            )
            ratio_values = " & ".join(format_ratio(row, key) for key, _, _ in specs)
            latex_lines.append(
                rf"{dataset_cell} & {method} & {metadata_ratio_cell} & {ratio_values} \\"
            )
            first = False
        latex_lines.append(r"\midrule")

    if latex_lines[-1] == r"\midrule":
        latex_lines[-1] = r"\bottomrule"
    else:
        latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(
        r"\caption{Runtime ratio relative to the backpropagated runtime. "
        r"Values are mean $\pm$ standard deviation of each scheme runtime divided by the "
        r"backpropagated runtime "
        r"over deduplicated runtime runs.}"
    )
    latex_lines.append(r"\label{tab:runtime-ratio}")
    latex_lines.append(r"\end{table}")
    return "\n".join(latex_lines)


def main():
    args = parse_args()
    specs = variant_specs()
    df = load_results(args.data_dir)
    observation_counts = infer_observation_counts(df)
    minirocket_features = load_minirocket_features()
    runtime_df = deduplicate_runtime_profiles(df, specs)
    agg = compute_ratios(runtime_df, specs)
    latex_table = build_latex_table(agg, specs, observation_counts, minirocket_features)

    print(latex_table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex_table + "\n")


if __name__ == "__main__":
    main()
