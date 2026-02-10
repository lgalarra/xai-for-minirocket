import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
DATA_DIR = Path("perturbation-results")
OUT_DIR = Path("./bar_charts_by_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "f_minus_f0-mean",
    "p2p_f_minus_f0-mean",
    "segmented_f_minus_f0-mean",
]

METRICS_FOR_LARGE_DATASETS = [
    "f_minus_f0-mean",
    "segmented_f_minus_f0-mean",
]

PERTURBATION_POLICY = "gaussian"


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

# -------------------------
# Filters
# -------------------------
data = data[
    ~(
        (data["mr_classifier"] == "RandomForestClassifier")
        & (data["base_explainer"] == "gradients")
    )
]

# Normalize dataset names
data["dataset"] = data["dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)

best_policies = {'starlight': 'opposite_class_medoid', 'abnormal-heartbeat-c1': 'opposite_class_farthest_instance', 'cognitive-circles': 'global_medoid', 'ford-a': 'opposite_class_medoid' }
data = data[
    data["perturbation_policy"] == PERTURBATION_POLICY
]

datas = []
for dataset, ref_policy in best_policies.items():
    datas.append(data[
        (data["dataset"] == dataset) & (data['reference_policy'] == ref_policy)
    ])    

data = pd.concat(datas, ignore_index=True)

# -------------------------
# Aggregate across runs
# -------------------------
agg = (
    data
    .groupby(["dataset", "base_explainer"], as_index=False)
    .mean(numeric_only=True)
)


# =========================
# Plot: one bar chart per dataset
# =========================
for dataset, g_ds in agg.groupby("dataset"):
    explainers = sorted(g_ds["base_explainer"].unique())
    THE_METRICS = METRICS if dataset not in ['cognitive-circles', 'abnormal-heartbeat-c1'] else METRICS_FOR_LARGE_DATASETS
    x = np.arange(len(THE_METRICS))
    width = 0.8 / len(explainers)

    plt.figure(figsize=(7, 5))

    for i, explainer in enumerate(explainers):
        g_exp = g_ds[g_ds["base_explainer"] == explainer]
        values = [g_exp[m].values[0] for m in THE_METRICS]

        plt.bar(
            x + i * width,
            values,
            width=width,
            label=explainer,
        )

    plt.xticks(x + width * (len(explainers) - 1) / 2, THE_METRICS, rotation=15)
    plt.ylabel("mean value")
    plt.title(f"{dataset} — budget = 1")
    plt.grid(axis="y", alpha=0.3)

    plt.legend(frameon=False)
    plt.tight_layout()

    out_file = OUT_DIR / f"{dataset}_budget1_barchart.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

