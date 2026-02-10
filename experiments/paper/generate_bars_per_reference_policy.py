import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
DATA_DIR = Path("perturbation-results")
OUT_DIR = Path("./bar_charts_reference_policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPLAINER = "gradients"
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

# -------------------------
# Aggregate across runs
# -------------------------
agg = (
    data
    .groupby(["dataset", "reference_policy"])[METRIC]
    .agg(["mean", "std"])
    .reset_index()
)


# =========================
# Plot: one chart per dataset
# =========================
for dataset, g_ds in agg.groupby("dataset"):
    g_ds = g_ds.sort_values("reference_policy")

    x = np.arange(len(g_ds))
    means = g_ds["mean"].values
    stds = g_ds["std"].values

    plt.figure(figsize=(7, 5))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    plt.xticks(x, g_ds["reference_policy"], rotation=20)
    plt.ylabel(METRIC)
    plt.title(f"{dataset} — {EXPLAINER}")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_file = OUT_DIR / f"{dataset}_{EXPLAINER}_reference_policy.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

