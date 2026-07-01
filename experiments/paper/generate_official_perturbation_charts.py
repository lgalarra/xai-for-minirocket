import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "official-results"
DEFAULT_OUT_DIR = SCRIPT_DIR / "official-perturbation-figures"

EVOLUTION_FACTORS_LABELS = {
    "percentile_cut": "top-k% observations perturbed",
    "interpolation": "perturbation scale (rho)",
    "sigma": "perturbation scale (rho)",
    "perturbation_policy": "perturbation policy",
}

DATASET_RENAMES = {
    r"^starlight-c.*": "starlight",
    r"^abnormal-heartbeat-c.*": "abnormal-heartbeat",
}

CORE_STEMS = ("f_minus_f0", "p2p_f_minus_f0")
SEGMENTED_RE = re.compile(r"^segmented(?:_n(?P<n>\d+))?_f_minus_f0$")
TSHAP_RE = re.compile(r"^tshap_w(?P<w>\d+)_s(?P<s>\d+)_f_minus_f0$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate perturbation charts from official perturbation result CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory with perturbation-results-* CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory where figures are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--evolution-factor",
        choices=("percentile_cut", "interpolation", "sigma", "perturbation_policy"),
        default="percentile_cut",
        help="Column used as the x axis.",
    )
    parser.add_argument(
        "--metric-kind",
        choices=("probability", "probability-norm", "change-ratio"),
        default="probability",
        help="Metric family to plot.",
    )
    parser.add_argument(
        "--perturbation-policy",
        default="gaussian",
        help="Perturbation policy filter. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--percentile-cut",
        type=float,
        default=90.0,
        help="percentile_cut filter when it is not the evolution factor.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="sigma filter when it is not the evolution factor.",
    )
    parser.add_argument(
        "--base-explainer",
        default="shap",
        help="base_explainer filter. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="mr_classifier filter. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--reference-policy",
        default="all",
        help="reference_policy filter. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--invert-percentile-cut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot 100 - percentile_cut when percentile_cut is the x axis.",
    )
    return parser.parse_args()


def is_metric_for_kind(column, metric_kind):
    if metric_kind == "probability":
        return column.endswith("-mean") and not column.endswith("_norm-mean")
    if metric_kind == "probability-norm":
        return column.endswith("_norm-mean")
    if metric_kind == "change-ratio":
        return column.endswith("-change_ratio")
    raise ValueError(f"Unknown metric kind: {metric_kind}")


def metric_stem(metric):
    for suffix in ("_norm-mean", "-mean", "-change_ratio"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def metric_family(metric):
    stem = metric_stem(metric)
    if stem in CORE_STEMS:
        return "core"
    if SEGMENTED_RE.match(stem):
        return "segmented"
    if TSHAP_RE.match(stem):
        return "tshap"
    return "other"


def metric_sort_key(metric):
    stem = metric_stem(metric)
    if stem == "f_minus_f0":
        return (0, 0, 0)
    if stem == "p2p_f_minus_f0":
        return (0, 1, 0)

    segmented_match = SEGMENTED_RE.match(stem)
    if segmented_match:
        n = segmented_match.group("n")
        return (1, int(n) if n else 0, 0)

    tshap_match = TSHAP_RE.match(stem)
    if tshap_match:
        return (2, int(tshap_match.group("w")), int(tshap_match.group("s")))

    return (3, stem, 0)


def metric_label(metric):
    stem = metric_stem(metric)
    if stem == "f_minus_f0":
        return "backprop"
    if stem == "p2p_f_minus_f0":
        return "e2e"

    segmented_match = SEGMENTED_RE.match(stem)
    if segmented_match:
        n = segmented_match.group("n")
        return "segmented" if n is None else f"segmented n={n}"

    tshap_match = TSHAP_RE.match(stem)
    if tshap_match:
        return f"t-shap w={tshap_match.group('w')} s={tshap_match.group('s')}"

    return stem.replace("_", " ")


def parse_args_column(df):
    args = df["args"].apply(ast.literal_eval)
    df["percentile_cut"] = args.apply(lambda d: as_float(d.get("percentile_cut")))
    df["budget"] = args.apply(lambda d: as_float(d.get("budget")))
    df["interpolation"] = args.apply(lambda d: as_float(d.get("interpolation"), default=0.0))
    df["sigma"] = args.apply(lambda d: as_float(d.get("sigma")))
    return df


def as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_metric_columns(df):
    metric_like = [
        col
        for col in df.columns
        if col.endswith(("-mean", "-std", "-change_ratio"))
        or col.endswith(("_norm-mean", "_norm-std"))
    ]
    for col in metric_like:
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(
                r"^\[([-+]?[0-9]*\.?[0-9]+)\]$",
                r"\1",
                regex=True,
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data(data_dir):
    files = sorted(data_dir.glob("perturbation-results-*"))
    if not files:
        raise FileNotFoundError(f"No perturbation-results-* files found in {data_dir}")

    dfs = []
    for csv_file in files:
        df = pd.read_csv(csv_file, low_memory=False)
        df = parse_args_column(df)
        df = coerce_metric_columns(df)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), files


def normalize_dataset_names(data):
    data = data.copy()
    for pattern, replacement in DATASET_RENAMES.items():
        data["dataset"] = data["dataset"].str.replace(pattern, replacement, regex=True)
    return data


def should_filter_sigma(args):
    if args.evolution_factor == "sigma":
        return False
    if args.perturbation_policy == "all":
        return False
    return args.perturbation_policy.startswith("gaussian")


def filter_data(data, args):
    data = data.copy()

    if args.evolution_factor != "perturbation_policy" and args.perturbation_policy != "all":
        data = data[data["perturbation_policy"] == args.perturbation_policy]

    if args.evolution_factor != "percentile_cut":
        data = data[np.isclose(data["percentile_cut"], args.percentile_cut, equal_nan=False)]

    if should_filter_sigma(args):
        data = data[np.isclose(data["sigma"], args.sigma, equal_nan=False)]

    if args.base_explainer != "all":
        data = data[data["base_explainer"] == args.base_explainer]

    if args.model != "all":
        data = data[data["mr_classifier"] == args.model]

    if args.reference_policy != "all":
        data = data[data["reference_policy"] == args.reference_policy]

    data = data[
        ~(
            (data["mr_classifier"] != "LogisticRegression")
            & (data["base_explainer"] == "gradients")
        )
    ]

    return normalize_dataset_names(data)


def discover_metrics(data, args):
    metrics = []
    for col in data.columns:
        if not is_metric_for_kind(col, args.metric_kind):
            continue
        stem = metric_stem(col)
        if stem in CORE_STEMS or SEGMENTED_RE.match(stem) or TSHAP_RE.match(stem):
            metrics.append(col)

    return sorted(metrics, key=metric_sort_key)


def aggregate(data, evolution_factor):
    grouping = ["dataset", "base_explainer", evolution_factor]
    if evolution_factor != "perturbation_policy":
        data = data[pd.notna(data[evolution_factor])]

    return (
        data.groupby(grouping, as_index=False)
        .mean(numeric_only=True)
        .sort_values(grouping)
    )


def best_metric_for_family(g_ds, metrics, family):
    candidates = [m for m in metrics if metric_family(m) == family]
    if not candidates:
        return None
    scores = g_ds[candidates].mean(numeric_only=True).sort_values(ascending=False)
    if scores.empty:
        return None
    return scores.index[0]


def best_metric_subset(g_ds, metrics):
    selected = [m for m in metrics if metric_family(m) == "core"]
    for family in ("segmented", "tshap"):
        best_metric = best_metric_for_family(g_ds, metrics, family)
        if best_metric is not None:
            selected.append(best_metric)
    return sorted(dict.fromkeys(selected), key=metric_sort_key)


def style_maps(metrics, explainers):
    cmap = plt.get_cmap("tab20")
    color_map = {
        metric: cmap(i % cmap.N)
        for i, metric in enumerate(metrics)
    }
    linestyles = ["-", "--", ":", "-."]
    linestyle_map = {
        explainer: linestyles[i % len(linestyles)]
        for i, explainer in enumerate(explainers)
    }
    return color_map, linestyle_map


def plot_dataset(g_ds, dataset, metrics, args, out_file):
    if args.evolution_factor == "percentile_cut" and args.invert_percentile_cut:
        g_ds = g_ds.copy()
        g_ds[args.evolution_factor] = 100 - g_ds[args.evolution_factor]

    g_ds = g_ds.sort_values(args.evolution_factor)
    explainers = sorted(g_ds["base_explainer"].unique())
    color_map, linestyle_map = style_maps(metrics, explainers)

    fig_width = max(7, min(14, 0.9 * len(metrics) + 5))
    plt.figure(figsize=(fig_width, 5))

    for explainer, g_exp in g_ds.groupby("base_explainer"):
        for metric in metrics:
            if metric not in g_exp:
                continue

            x = g_exp[args.evolution_factor]
            y = g_exp[metric]
            label = metric_label(metric)
            if len(explainers) > 1:
                label = f"{explainer} | {label}"

            plt.plot(
                x,
                y,
                linestyle=linestyle_map[explainer],
                linewidth=2.5,
                marker=".",
                color=color_map[metric],
                label=label,
            )

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(EVOLUTION_FACTORS_LABELS[args.evolution_factor], fontsize=20)
    plt.ylabel(y_axis_label(args.metric_kind), fontsize=20)
    plt.title(dataset, fontsize=20)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    if args.evolution_factor != "perturbation_policy":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncols = 1 if len(labels) <= 5 else 2
        plt.legend(
            handles,
            labels,
            fontsize=12,
            ncol=ncols,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
        )

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def y_axis_label(metric_kind):
    if metric_kind == "probability":
        return "Delta f - probability drop"
    if metric_kind == "probability-norm":
        return "Normalized probability drop"
    if metric_kind == "change-ratio":
        return "Change class ratio"
    return "mean value"


def output_name(dataset, args):
    parts = [
        dataset,
        args.evolution_factor,
        args.metric_kind,
        args.perturbation_policy if args.evolution_factor != "perturbation_policy" else "all-policies",
        args.model,
    ]
    return "_".join(str(part).replace("/", "-") for part in parts) + ".png"


def main():
    args = parse_args()
    data, files = load_data(args.data_dir)
    data = filter_data(data, args)
    if data.empty:
        raise ValueError("No rows left after filtering. Adjust the CLI filters.")

    metrics = discover_metrics(data, args)
    if not metrics:
        raise ValueError(f"No metrics discovered for metric kind {args.metric_kind}.")

    agg = aggregate(data, args.evolution_factor)
    print(f"Read {len(files)} files from {args.data_dir}")
    print(f"Plotting {len(metrics)} metrics: {', '.join(metric_label(m) for m in metrics)}")

    selected = {}
    for dataset, g_ds in agg.groupby("dataset"):
        filename = output_name(dataset, args)
        plot_dataset(g_ds, dataset, metrics, args, args.out_dir / "all_methods" / filename)

        best_metrics = best_metric_subset(g_ds, metrics)
        selected[dataset] = [metric_label(m) for m in best_metrics]
        plot_dataset(g_ds, dataset, best_metrics, args, args.out_dir / "best_methods" / filename)

    print(f"Wrote figures under {args.out_dir}")
    print("Best-method plots used:")
    for dataset, labels in selected.items():
        print(f"  {dataset}: {', '.join(labels)}")


if __name__ == "__main__":
    main()
