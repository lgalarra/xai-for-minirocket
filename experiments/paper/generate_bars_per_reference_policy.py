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

EXPLAINER = "stratoshap-k1"
METRIC = "f_minus_f0-mean"
#METRIC = "p2p_f_minus_f0-mean"
LABEL = "predicted"
PERTURBATION_POLICY = "instance_to_reference"


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
data = data.drop(data[(data["mr_classifier"] == "RandomForestClassifier") 
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
        & (data["mr_classifier"] == "LogisticRegression")
	    
    ]
else:
    data = data[
        (data["percentile_cut"] == 90)
        & (data["sigma"] == 3.0)
        & (data["base_explainer"] == EXPLAINER) 
	    & (data["label"] == LABEL) 
        & (data["perturbation_policy"] == PERTURBATION_POLICY)
        & (data["mr_classifier"] == "LogisticRegression")
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

# -------------------------
# Aggregate across runs
# -------------------------
#agg = (
#    data
#    .groupby(["dataset", "reference_policy"])[METRIC]
#    .agg(["mean", "std"])
#    .reset_index()
#)



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
for dataset, g_ds in data.groupby("dataset"):

    plt.figure(figsize=(7, 5))

    order = sorted(g_ds["reference_policy"].unique())

    ax = sns.boxplot(
        data=g_ds,
        x="reference_policy",
        y=METRIC,
        order=order,
        width=0.6,
        fliersize=3,        # size of outlier markers
        linewidth=1.2,
    )

    # Optional: overlay individual runs (recommended for small N)
    sns.stripplot(
        data=g_ds,
        x="reference_policy",
        y=METRIC,
        order=order,
        color="black",
        size=3,
        alpha=0.4,
        jitter=0.15,
    )

    ax.set_ylabel(METRIC)
    ax.set_xlabel("Reference policy")
    ax.set_title(f"{dataset} — {EXPLAINER}")

    plt.xticks(rotation=20)
    plt.tight_layout()

    out_file = OUT_DIR / f"{dataset}_{EXPLAINER}_reference_policy_boxplot.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
