import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "official-results"
OUT_DIR = SCRIPT_DIR / "bar_charts_reference_policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_METRIC = "f_minus_f0"
METRIC = BASE_METRIC + "-mean"
LABEL = "predicted"
PERTURBATION_POLICIES = ("gaussian", "instance_to_reference")
#MODEL_NAME = "RandomForestClassifier"


def as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =========================
# Load and preprocess data
# =========================
dfs = []

for csv_file in DATA_DIR.glob("perturbation-results-*"):
    df = pd.read_csv(csv_file, low_memory=False)

    # Parse args dictionary
    args = df["args"].apply(ast.literal_eval)
    df["percentile_cut"] = args.apply(lambda d: as_float(d.get("percentile_cut")))
    df["budget"] = args.apply(lambda d: as_float(d.get("budget")))
    df["interpolation"] = args.apply(lambda d: as_float(d.get("interpolation"), default=0.0))
    df["sigma"] = args.apply(lambda d: as_float(d.get("sigma")))

    dfs.append(df)

if not dfs:
    raise FileNotFoundError(f"No perturbation-results-* files found in {DATA_DIR}")

data = pd.concat(dfs, ignore_index=True)
data = data.drop(data[(data["mr_classifier"] != "LogisticRegression")
& (data["base_explainer"] == "gradients")].index)

# Normalize dataset names
data["dataset"] = data["dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)

data["mr_classifier"] = data["mr_classifier"].replace({'LogisticRegression': 'LR', 'RandomForestClassifier': 'RF', 'MLPClassifier': 'MLP'})
data["reference_policy"] = data["reference_policy"].replace(
    {'global_centroid': 'centroid', 'global_medoid': 'medoid',
     'opposite_class_centroid': 'enemy centroid', 'opposite_class_medoid': 'enemy medoid',
     'opposite_class_farthest_instance': 'farthest enemy', 'opposite_class_closest_instance': 'closest enemy'})

data["dataset"] = data["dataset"].str.replace(
    r"^abnormal-heartbeat-c.*",
    "abnormal-heartbeat",
    regex=True,
)

# =========================
# Expand f_minus_f0 column (semicolon-separated list)
# =========================

def parse_semicolon_list(x):
    if pd.isna(x):
        return []
    return [float(v) for v in str(x).split(";") if v != ""]


def expand_metric_values(data):
    data = data.copy()
    data[f"{BASE_METRIC}_list"] = data[f"{BASE_METRIC}"].apply(parse_semicolon_list)
    data = data.explode(f"{BASE_METRIC}_list")
    data = data.rename(columns={f"{BASE_METRIC}_list": f"{BASE_METRIC}_value"})
    data[f"{BASE_METRIC}_value"] = pd.to_numeric(data[f"{BASE_METRIC}_value"], errors="coerce")
    return data.dropna(subset=[f"{BASE_METRIC}_value"])


# -------------------------
# Aggregate across runs
# -------------------------
#agg = (
#    data
#    .groupby(["dataset", "reference_policy"])[METRIC]
#    .agg(["mean", "std"])
#    .reset_index()
#)
METRICS_LABELS = {"f_minus_f0-mean": "avg. ∆f - Probability drop"}

EXPLAINER_LABELS = {'shap': 'SHAP', 'stratoshap-k1': 'ST-SHAP',
                    'extreme_feature_coalitions': 'EFC', 'gradients': 'Gradients'}


sns.set(style="whitegrid", context="paper")


def safe_filename_part(value):
    return str(value).replace("/", "-")


def filter_for_plot(data, explainer, perturbation_policy):
    filtered = data[
        (data["base_explainer"] == explainer)
        & (data["label"] == LABEL)
        & (data["perturbation_policy"] == perturbation_policy)
        & (np.isclose(data["percentile_cut"], 90.0))
    ]

    if perturbation_policy == "gaussian":
        return filtered[np.isclose(filtered["sigma"], 3.0)]

    return filtered[
        (np.isclose(filtered["budget"], 1.0))
        & (np.isclose(filtered["interpolation"], 1.0))
    ]


written_files = []
explainers = sorted(data["base_explainer"].dropna().unique())
for perturbation_policy in PERTURBATION_POLICIES:
    for explainer in explainers:
        plot_data = filter_for_plot(data, explainer, perturbation_policy)
        if plot_data.empty:
            continue
        plot_data = expand_metric_values(plot_data)
        if plot_data.empty:
            continue

        for dataset, g_ds in plot_data.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(8, 5))

            order = sorted(g_ds["reference_policy"].unique())
            model_order = sorted(g_ds["mr_classifier"].unique())

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
            #ax.set_xlabel("Reference policy")
            #ax.set_title(f"{dataset} — {EXPLAINER_LABELS.get(explainer, explainer)}")

            ax.tick_params(axis="x", labelrotation=20, labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.legend(
                title="Model",
                loc="best",
                frameon=True,
            )
            fig.tight_layout()

            out_file = OUT_DIR / (
                f"{safe_filename_part(dataset)}_{safe_filename_part(explainer)}_"
                f"{safe_filename_part(BASE_METRIC)}_{safe_filename_part(perturbation_policy)}_"
                "reference_policy_boxplot.png"
            )
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written_files.append(out_file)

if not written_files:
    raise ValueError("No charts were generated. Adjust the script configuration.")

print(f"Wrote {len(written_files)} figures under {OUT_DIR}")
