import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


SCRIPT_DIR = Path(__file__).resolve().parent
TEX_DIR = SCRIPT_DIR / "tex"
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "official-results"
DEFAULT_OUT_DIR = TEX_DIR / "bar_charts_reference_policy"
DEFAULT_OUTPUT_FORMAT = "tikz"

BASE_METRIC = "f_minus_f0"
METRIC = BASE_METRIC + "-mean"
LABEL = "predicted"
PERTURBATION_POLICIES = ("gaussian", "instance_to_reference")
GRADIENT_CLASSIFIERS = ("LogisticRegression", "MLPClassifier")
MODEL_ORDER = ("LR", "MLP", "RF")
MODEL_COLORS = dict(
    zip(MODEL_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"])
)
GRADIENT_PERTURBATION_POLICY_REPLACEMENTS = {
    "gaussian": "gradient_gaussian_opposite",
}
METRICS_LABELS = {"f_minus_f0-mean": "avg. Delta f - Probability drop"}
MODEL_RENAMES = {
    "LogisticRegression": "LR",
    "RandomForestClassifier": "RF",
    "MLPClassifier": "MLP",
}
REFERENCE_POLICY_RENAMES = {
    "global_centroid": "centroid",
    "global_medoid": "medoid",
    "opposite_class_centroid": "enemy centroid",
    "opposite_class_medoid": "enemy medoid",
    "opposite_class_farthest_instance": "farthest enemy",
    "opposite_class_closest_instance": "closest enemy",
}
REQUIRED_COLUMNS = {
    "args",
    "base_explainer",
    BASE_METRIC,
    "dataset",
    "label",
    "mr_classifier",
    "perturbation_policy",
    "reference_policy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reference-policy boxplots from official results."
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
        help=f"Directory where charts are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--format",
        choices=("tikz", "png"),
        default=DEFAULT_OUTPUT_FORMAT,
        dest="output_format",
        help=(
            "Write one PGFPlots/TikZ .tex file per chart or retain the legacy "
            f"PNG output. Default: {DEFAULT_OUTPUT_FORMAT}."
        ),
    )
    return parser.parse_args()


def as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_data(data_dir):
    dfs = []
    for csv_file in data_dir.glob("perturbation-results-*"):
        df = pd.read_csv(
            csv_file,
            usecols=lambda column: column in REQUIRED_COLUMNS,
            low_memory=False,
        )
        parsed_args = df["args"].apply(ast.literal_eval)
        df["percentile_cut"] = parsed_args.apply(
            lambda values: as_float(values.get("percentile_cut"))
        )
        df["budget"] = parsed_args.apply(
            lambda values: as_float(values.get("budget"))
        )
        df["interpolation"] = parsed_args.apply(
            lambda values: as_float(values.get("interpolation"), default=0.0)
        )
        df["sigma"] = parsed_args.apply(lambda values: as_float(values.get("sigma")))
        dfs.append(df.dropna(axis=1, how="all"))

    if not dfs:
        raise FileNotFoundError(f"No perturbation-results-* files found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def preprocess_data(data):
    data = data[
        (data["base_explainer"] != "gradients")
        | data["mr_classifier"].isin(GRADIENT_CLASSIFIERS)
    ].copy()
    data = data[data["reference_policy"] != "counterfactual"].copy()
    data["dataset"] = data["dataset"].str.replace(
        r"^starlight-c.*", "starlight", regex=True
    )
    data["dataset"] = data["dataset"].str.replace(
        r"^abnormal-heartbeat-c.*", "abnormal-heartbeat", regex=True
    )
    data["mr_classifier"] = data["mr_classifier"].replace(MODEL_RENAMES)
    data["reference_policy"] = data["reference_policy"].replace(
        REFERENCE_POLICY_RENAMES
    )
    return data


def parse_semicolon_list(value):
    if pd.isna(value):
        return []
    return [float(item) for item in str(value).split(";") if item != ""]


def expand_metric_values(data):
    data = data.copy()
    list_column = f"{BASE_METRIC}_list"
    value_column = f"{BASE_METRIC}_value"
    data[list_column] = data[BASE_METRIC].apply(parse_semicolon_list)
    data = data.explode(list_column).rename(columns={list_column: value_column})
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    return data.dropna(subset=[value_column])


def safe_filename_part(value):
    return str(value).replace("/", "-")


def perturbation_policy_for_explainer(explainer, perturbation_policy):
    if explainer == "gradients":
        return GRADIENT_PERTURBATION_POLICY_REPLACEMENTS.get(
            perturbation_policy, perturbation_policy
        )
    return perturbation_policy


def filter_for_plot(data, explainer, perturbation_policy):
    resolved_policy = perturbation_policy_for_explainer(
        explainer, perturbation_policy
    )
    filtered = data[
        (data["base_explainer"] == explainer)
        & (data["label"] == LABEL)
        & (data["perturbation_policy"] == resolved_policy)
        & np.isclose(data["percentile_cut"], 90.0)
    ]
    if resolved_policy.startswith("gradient_gaussian") or resolved_policy == "gaussian":
        return filtered[np.isclose(filtered["sigma"], 3.0)]
    return filtered[
        np.isclose(filtered["budget"], 1.0)
        & np.isclose(filtered["interpolation"], 1.0)
    ]


def tikz_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def tikz_color(color):
    red, green, blue = to_rgb(color)
    return f"{{rgb,1:red,{red:.4f};green,{green:.4f};blue,{blue:.4f}}}"


def boxplot_statistics(values):
    values = np.sort(np.asarray(values, dtype=float))
    lower_quartile, median, upper_quartile = np.percentile(values, [25, 50, 75])
    iqr = upper_quartile - lower_quartile
    lower_limit = lower_quartile - 1.5 * iqr
    upper_limit = upper_quartile + 1.5 * iqr
    inliers = values[(values >= lower_limit) & (values <= upper_limit)]
    outliers = values[(values < lower_limit) | (values > upper_limit)]
    return {
        "lower whisker": inliers.min(),
        "lower quartile": lower_quartile,
        "median": median,
        "upper quartile": upper_quartile,
        "upper whisker": inliers.max(),
        "outliers": outliers,
    }


def write_tikz_boxplot(g_ds, order, model_order, out_file):
    if len(model_order) == 2:
        offsets = (-0.16, 0.16)
        box_extend = 0.24
    else:
        offsets = (
            np.linspace(-0.25, 0.25, len(model_order))
            if len(model_order) > 1
            else [0]
        )
        box_extend = {1: 0.55}.get(len(model_order), 0.22)
    positions = dict(zip(model_order, offsets))
    tick_positions = ",".join(str(index) for index in range(1, len(order) + 1))
    tick_labels = ",".join(f"{{{tikz_escape(label)}}}" for label in order)
    lines = [
        "% Generated by generate_bars_per_reference_policy.py.",
        "% Requires \\usepackage{pgfplots}, \\usepgfplotslibrary{statistics},",
        "% and \\pgfplotsset{compat=1.18}.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.84\linewidth,",
        r"  height=0.58\linewidth,",
        r"  ylabel={Average $\Delta f$ -- probability drop},",
        f"  xtick={{{tick_positions}}},",
        f"  xticklabels={{{tick_labels}}},",
        r"  x tick label style={rotate=25,anchor=east,font=\scriptsize,yshift=-0.3em},",
        r"  tick label style={font=\small},",
        r"  label style={font=\small},",
        r"  ylabel style={xshift=0.4em},",
        r"  grid=major,",
        r"  boxplot/draw direction=y,",
        f"  boxplot/box extend={box_extend},",
        r"  legend style={at={(0.5,-0.34)},anchor=north,legend columns=-1,",
        r"    font=\small,draw=none},",
        r"]",
    ]

    value_column = f"{BASE_METRIC}_value"
    for reference_index, reference_policy in enumerate(order, start=1):
        reference_data = g_ds[g_ds["reference_policy"] == reference_policy]
        for model in model_order:
            values = reference_data.loc[
                reference_data["mr_classifier"] == model, value_column
            ].dropna()
            if values.empty:
                continue
            stats = boxplot_statistics(values)
            position = reference_index + positions[model]
            color = tikz_color(MODEL_COLORS[model])
            prepared = ",".join(
                f"{key}={float(stats[key]):.12g}"
                for key in (
                    "lower whisker",
                    "lower quartile",
                    "median",
                    "upper quartile",
                    "upper whisker",
                )
            )
            outliers = " ".join(
                f"({position:.6g},{float(value):.12g})"
                for value in stats["outliers"]
            )
            lines.extend(
                [
                    rf"\addplot+[boxplot prepared={{{prepared}}},",
                    rf"  boxplot prepared={{draw position={position:.6g}}},",
                    rf"  fill={color},draw={color},mark size=0.8pt,forget plot]",
                    rf"  coordinates {{{outliers}}};",
                ]
            )

    for model in model_order:
        color = tikz_color(MODEL_COLORS[model])
        lines.extend(
            [
                (
                    r"\addlegendimage{only marks,mark=square*,mark size=3pt,"
                    rf"color={color}}}"
                ),
                rf"\addlegendentry{{{tikz_escape(model)}}}",
            ]
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", ""])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")


def write_png_boxplot(g_ds, order, model_order, out_file):
    try:
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "PNG output requires seaborn; install the project requirements first."
        ) from error

    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=g_ds,
        x="reference_policy",
        y=f"{BASE_METRIC}_value",
        hue="mr_classifier",
        order=order,
        hue_order=model_order,
        width=0.7,
        fliersize=2,
        linewidth=1.1,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(METRICS_LABELS[METRIC], fontsize=18)
    ax.tick_params(axis="x", labelrotation=20, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(
        title="Model", loc="best", frameon=True, fontsize=14, title_fontsize=15
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def output_name(dataset, explainer, perturbation_policy, output_format):
    extension = ".tex" if output_format == "tikz" else ".png"
    return (
        f"{safe_filename_part(dataset)}_{safe_filename_part(explainer)}_"
        f"{safe_filename_part(BASE_METRIC)}_"
        f"{safe_filename_part(perturbation_policy)}_reference_policy_boxplot"
        f"{extension}"
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = preprocess_data(load_data(args.data_dir))

    written_files = []
    explainers = sorted(data["base_explainer"].dropna().unique())
    for perturbation_policy in PERTURBATION_POLICIES:
        for explainer in explainers:
            plot_data = expand_metric_values(
                filter_for_plot(data, explainer, perturbation_policy)
            )
            if plot_data.empty:
                continue

            for dataset, g_ds in plot_data.groupby("dataset"):
                order = sorted(g_ds["reference_policy"].unique())
                model_order = [
                    model
                    for model in MODEL_ORDER
                    if model in g_ds["mr_classifier"].unique()
                ]
                out_file = args.out_dir / output_name(
                    dataset, explainer, perturbation_policy, args.output_format
                )
                if args.output_format == "tikz":
                    write_tikz_boxplot(g_ds, order, model_order, out_file)
                else:
                    write_png_boxplot(g_ds, order, model_order, out_file)
                written_files.append(out_file)

    if not written_files:
        raise ValueError("No charts were generated. Adjust the script configuration.")
    print(f"Wrote {len(written_files)} {args.output_format} charts under {args.out_dir}")


if __name__ == "__main__":
    main()
