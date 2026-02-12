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
PERTURBATION_POLICY = 'gaussian'
EVOLUTION_FACTOR = 'percentile_cut'
EXPLANATION_METHOD = 'shap'

PROBABILITY_METRICS = [
    "f_minus_f0-mean",
    "p2p_f_minus_f0-mean",
    "segmented_f_minus_f0-mean",
]
CHANGE_RATIO_METRICS = [
    "f_minus_f0-change_ratio",
    "p2p_f_minus_f0-change_ratio",
    "segmented_f_minus_f0-change_ratio",
]

METRICS_LABELS = {"f_minus_f0-mean": "backpropagated", "p2p_f_minus_f0-mean": "p2p",
                  "segmented_f_minus_f0-mean": "segmented", "f_minus_f0-change_ratio": "backpropagated",
                  "p2p_f_minus_f0-change_ratio": "p2p", "segmented_f_minus_f0-change_ratio": "segmented"
                  }

METRICS = PROBABILITY_METRICS

METRICS_LARGE_DATASETS = [METRICS[0], METRICS[2]]

if METRICS == PROBABILITY_METRICS:
    Y_AXIS_LABEL = "Δf"
else:
    Y_AXIS_LABEL = "Change class ratio"

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
    df["budget"] = args.apply(lambda d: int(d.get("budget")))
    df["interpolation"] = args.apply(lambda d: float(d.get("interpolation")) if "interpolation" in d else 0.0)
    df["sigma"] = args.apply(lambda d: float(d.get("sigma")) if "sigma" in d else None)


    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# -------------------------
# Filter: interpolation = 1.0 only
# -------------------------
if EVOLUTION_FACTOR != "perturbation_policy":
    data = data[data["perturbation_policy"] == PERTURBATION_POLICY]
#data = data[data["interpolation"] == 1.0]

if EVOLUTION_FACTOR != "percentile_cut":
    data = data[data["percentile_cut"] == 90]

if EVOLUTION_FACTOR != "sigma":
    data = data[data["sigma"] == 3.0]

if EXPLANATION_METHOD is not None:
    data = data[data["base_explainer"] == EXPLANATION_METHOD]


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

data["p2p_f_minus_f0-change_ratio"] = pd.to_numeric(data["p2p_f_minus_f0-change_ratio"].str.replace(
    r"^\[-1.0\]$",
    "-1.0",
    regex=True,
))

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
        ["dataset", "base_explainer", EVOLUTION_FACTOR],
        as_index=False
    )
    .mean(numeric_only=True)
)



# =========================
# Plot: one chart per dataset
# =========================

LINESTYLES = {
    METRICS[0]: "-",
    METRICS[1]: "--",
    METRICS[2]: ":",
}


# for dataset, g_ds in agg.groupby("dataset"):
#     g_ds = g_ds.sort_values(EVOLUTION_FACTOR)
#
#     plt.figure(figsize=(7, 5))
#
#     # Assign one color per explainer
#     explainers = sorted(g_ds["base_explainer"].unique())
#     color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     color_map = {
#         explainer: color_cycle[i % len(color_cycle)]
#         for i, explainer in enumerate(explainers)
#     }
#
#     for explainer, g_exp in g_ds.groupby("base_explainer"):
#         color = color_map[explainer]
#
#         for metric in (METRICS if dataset not in ('cognitive-circles', 'abnormal-heartbeat') else METRICS_LARGE_DATASETS):
#             plt.plot(
#                 g_exp[EVOLUTION_FACTOR],
#                 g_exp[metric],
#                 linestyle=LINESTYLES[metric],
#                 marker="o",
#                 color=color,
#                 label=f"{explainer} | {metric}",
#             )
#
#     plt.xlabel(EVOLUTION_FACTOR)
#     plt.ylabel("mean value")
#     plt.title(f"{dataset}")
#     plt.grid(True, alpha=0.3)
#
#     # Legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     if handles:
#         plt.legend(
#             handles,
#             labels,
#             fontsize=8,
#             ncol=2,
#             frameon=False,
#         )
#
#     plt.tight_layout()
#     out_file = OUT_DIR / f"{dataset}_{EVOLUTION_FACTOR}.png"
#     plt.savefig(out_file, dpi=300, bbox_inches="tight")
#     plt.close()
#

EVOLUTION_FACTORS_LABELS = {"percentile_cut": "top-k% observations perturbed",
                            "interpolation" :  "perturbation scale (ρ)",
                            "sigma" : "perturbation scale (ρ)",
                            }



for dataset, g_ds in agg.groupby("dataset"):
    g_ds = g_ds.sort_values(EVOLUTION_FACTOR)

    plt.figure(figsize=(7, 5))

    explainers = sorted(g_ds["base_explainer"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        explainer: color_cycle[i % len(color_cycle)]
        for i, explainer in enumerate(explainers)
    }

    for explainer, g_exp in g_ds.groupby("base_explainer"):
        color = color_map[explainer]

        metrics_to_use = (
            METRICS
            if dataset not in ('cognitive-circles', 'abnormal-heartbeat')
            else METRICS_LARGE_DATASETS
        )

        for metric in metrics_to_use:
            std_col = metric.replace("mean", "std")
            x = g_exp[EVOLUTION_FACTOR]
            y = g_exp[metric]
            # y_std = g_exp[std_col]

            # Plot mean curve
            plt.plot(
                x,
                y,
                linestyle=LINESTYLES[metric],
                marker=".",
                color=color,
                label=f"{explainer} | {METRICS_LABELS[metric]}" if EXPLANATION_METHOD is None else f"{METRICS_LABELS[metric]}",
            )

            # # Add variance band (mean ± std)
            # plt.fill_between(
            #     x,
            #     y - y_std/2,
            #     y + y_std/2,
            #     color=color,
            #     alpha=0.2,
            # )

    plt.xlabel(EVOLUTION_FACTORS_LABELS[EVOLUTION_FACTOR])
    plt.ylabel(Y_AXIS_LABEL)
    #plt.title(f"{dataset}")
    plt.grid(True, alpha=0.3)
    if dataset == 'ford-a':
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(
             handles,
             labels,
             fontsize=14,
             ncol=1,
             frameon=False, loc="center right", handlelength=3.0, handleheight=1.5,
            )

    plt.tight_layout()
    out_file = OUT_DIR / f"{dataset}_{explainer}_{EVOLUTION_FACTOR}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
