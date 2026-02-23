import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# =========================
# Configuration
# =========================
DATA_DIR = Path("perturbation-results")
OUT_DIR = Path("./bar_charts_reference_policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPLAINER = "gradients"
#BASE_METRIC = "p2p_f_minus_f0"
BASE_METRIC = "f_minus_f0"
METRIC = BASE_METRIC + "-mean"
LABEL = "predicted"
PERTURBATION_POLICY = "instance_to_reference"
#MODEL_NAME = "RandomForestClassifier"


# =========================
# Load and preprocess data
# =========================
dfs = []



for csv_file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)

    # Parse args dictionary
    args = df["args"].apply(ast.literal_eval)
    df["percentile_cut"] = args.apply(lambda d: d.get("percentile_cut"))
    df["budget"] = args.apply(lambda d: d.get("budget"))
    
    if PERTURBATION_POLICY != "gaussian":    
        df["interpolation"] = args.apply(lambda d: d.get("interpolation"))
    else:
        df["sigma"] = args.apply(lambda d: d.get("sigma"))

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.drop(data[(data["mr_classifier"] != "LogisticRegression")
& (data["base_explainer"] == "gradients")].index)

# -------------------------
# Filters
# -------------------------
if PERTURBATION_POLICY != "gaussian":
    data = data[
        (data["budget"] == 1)
        & (data["percentile_cut"] == 90)
        & (data["interpolation"] == 1.0)
        & (data["base_explainer"] == EXPLAINER) 
	    & (data["label"] == LABEL) 
        & (data["perturbation_policy"] == PERTURBATION_POLICY)
    ]
else:
    data = data[
        (data["percentile_cut"] == 90)
        & (data["sigma"] == 3.0)
        & (data["base_explainer"] == EXPLAINER) 
	    & (data["label"] == LABEL) 
        & (data["perturbation_policy"] == PERTURBATION_POLICY)
    ]

# Normalize dataset names
data["dataset"] = data["dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)
REFERENCE_POLICIES = {'global_centroid': 'centroid', 'global_medoid': 'medoid'}

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
    r"^starlight-c.*",
    "starlight",
    regex=True,
)

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

data[f"{BASE_METRIC}_list"] = data[F"{BASE_METRIC}"].apply(parse_semicolon_list)

# Explode into long format
data = data.explode(f"{BASE_METRIC}_list")

# Rename for clarity
data = data.rename(columns={f"{BASE_METRIC}_list": f"{BASE_METRIC}_value"})


# -------------------------
# Aggregate across runs
# -------------------------
#agg = (
#    data
#    .groupby(["dataset", "reference_policy"])[METRIC]
#    .agg(["mean", "std"])
#    .reset_index()
#)
METRICS_LABELS = {"f_minus_f0-mean": "avg. Δf", "p2p_f_minus_f0-mean": "avg. Δf",
                  "segmented_f_minus_f0-mean": "avg. Δf", "f_minus_f0-change_ratio": "avg. Δf",
                  "p2p_f_minus_f0-change_ratio": "avg. Δf", "segmented_f_minus_f0-change_ratio": "avg. Δf"
                  }

EXPLAINER_LABELS = {'shap': 'SHAP', 'stratoshap-k1': 'ST-SHAP',
                    'extreme_feature_coalitions': 'EFC', 'gradients': 'Gradients'}


# Use a clean style suitable for papers
sns.set(style="whitegrid", context="paper")

# =========================
# Plot: one violin chart per dataset
# =========================
import seaborn as sns

sns.set(style="whitegrid", context="paper")

# =========================
# Plot: one boxplot per dataset
# =========================
# =========================
# Plot: one boxplot per dataset
# Each reference policy contains one box per MODEL
# =========================

for dataset, g_ds in data.groupby("dataset"):

    plt.figure(figsize=(8, 5))

    order = sorted(g_ds["reference_policy"].unique())
    model_order = sorted(g_ds["mr_classifier"].unique())

    ax = sns.boxplot(
        data=g_ds,
        x="reference_policy",
        y=f"{BASE_METRIC}_value",
        hue="mr_classifier",
        order=order,
        hue_order=model_order,
        width=0.7,
        fliersize=2,
        linewidth=1.1,
    )

    ax.set_ylabel(METRICS_LABELS[METRIC])
    ax.set_xlabel("Reference policy")
    ax.set_title(f"{dataset} — {EXPLAINER_LABELS[EXPLAINER]}")

    plt.xticks(rotation=20)
    #plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.legend(
        title="Model",
        loc="best",
        frameon=True
    )
    plt.tight_layout()

    out_file = OUT_DIR / f"{dataset}_{EXPLAINER}_{BASE_METRIC}_{PERTURBATION_POLICY}_reference_policy_boxplot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

