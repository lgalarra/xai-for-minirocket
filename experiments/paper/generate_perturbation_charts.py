import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================
DATA_DIR = Path("perturbation-results")      # directory with the 864 CSV files
OUT_DIR = Path("./figures")       # output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "f_minus_f0-mean",
    "p2p_f_minus_f0-mean",
    "segmented_f_minus_f0-mean",
]

OUTPUT_FIG = OUT_DIR / "budget1_all_explainers.png"


# =========================
# Load and preprocess data
# =========================
dfs = []

for csv_file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)

    # Parse args dict
    args = df["args"].apply(ast.literal_eval)
    df["percentile_cut"] = args.apply(lambda d: float(d.get("percentile_cut")))
    df["interpolation"] = args.apply(lambda d: float(d.get("interpolation")) if "interpolation" in d else 0.0)
    df["budget"] = args.apply(lambda d: int(d.get("budget")))

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# -------------------------
# Filter: interpolation = 1.0 only
# -------------------------
data = data[data["perturbation_policy"] == 'instance_to_reference']
#data = data[data["interpolation"] == 1.0]
data = data[data["percentile_cut"] == 90]

data["dataset"] = data["dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)

print(data["dataset"].unique())

# exclude (mr_classifier == RF) AND (base_explainer == gradients)
data = data[
    ~(
        (data["mr_classifier"] == "RandomForestClassifier")
        & (data["base_explainer"] == "gradients")
    )
]

# -------------------------
# Aggregate across runs
# -------------------------
agg = (
    data
    .groupby(
        ["dataset", "base_explainer", "interpolation"],
        as_index=False
    )
    .mean(numeric_only=True)
)


# =========================
# Plot: one chart per dataset
# =========================

LINESTYLES = {
    "f_minus_f0-mean": "-",
    "p2p_f_minus_f0-mean": "--",
    "segmented_f_minus_f0-mean": ":",
}

for dataset, g_ds in agg.groupby("dataset"):
    g_ds = g_ds.sort_values("interpolation")

    plt.figure(figsize=(7, 5))

    # Assign one color per explainer
    explainers = sorted(g_ds["base_explainer"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        explainer: color_cycle[i % len(color_cycle)]
        for i, explainer in enumerate(explainers)
    }

    for explainer, g_exp in g_ds.groupby("base_explainer"):
        color = color_map[explainer]

        for metric in METRICS:
            plt.plot(
                g_exp["interpolation"],
                g_exp[metric],
                linestyle=LINESTYLES[metric],
                marker="o",
                color=color,
                label=f"{explainer} | {metric}",
            )

    plt.xlabel("interpolation")
    plt.ylabel("mean value")
    plt.title(f"{dataset} — budget = 1")
    plt.grid(True, alpha=0.3)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(
            handles,
            labels,
            fontsize=8,
            ncol=2,
            frameon=False,
        )

    plt.tight_layout()
    out_file = OUT_DIR / f"{dataset}_budget1.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

