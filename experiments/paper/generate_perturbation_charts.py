import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, MaxNLocator

# =========================
# Configuration
# =========================
DATA_DIR = Path("perturbation-results")      # directory with the 864 CSV files
OUT_DIR = Path("./figures")       # output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)
PERTURBATION_POLICY = 'gaussian'
EVOLUTION_FACTOR = 'percentile_cut'
EXPLANATION_METHOD = 'gradients'
MODEL = None
#MODEL = "MLPClassifier"

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

METRICS_LABELS = {"f_minus_f0-mean": "backprop", "p2p_f_minus_f0-mean": "e2e",
                  "segmented_f_minus_f0-mean": "leftist", "f_minus_f0-change_ratio": "backprop",
                  "p2p_f_minus_f0-change_ratio": "e2e", "segmented_f_minus_f0-change_ratio": "leftist",
                  }

METRICS = PROBABILITY_METRICS

METRICS_LARGE_DATASETS = [METRICS[0], METRICS[2]]

if METRICS == PROBABILITY_METRICS:
    Y_AXIS_LABEL = "Δf - Probability drop"
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

if MODEL is not None:
    data = data[data["mr_classifier"] == MODEL]


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
        (data["mr_classifier"] != "LogisticRegression")
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

min_y = np.min([data[m].min() for m in METRICS_LARGE_DATASETS ])
max_y = np.min([data[m].max() for m in METRICS_LARGE_DATASETS ])

min_x = data[EVOLUTION_FACTOR].min()
max_x = data[EVOLUTION_FACTOR].max()

for dataset, g_ds in agg.groupby("dataset"):
    if EVOLUTION_FACTOR == "percentile_cut":
        g_ds['percentile_cut'] = 100 - g_ds['percentile_cut']

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
            y_std = g_exp[std_col]

            # Plot mean curve
            plt.plot(
                x,
                y,
                linestyle=LINESTYLES[metric],
                linewidth=3,
                marker=".",
                color=color,
                label=f"{explainer} | {METRICS_LABELS[metric]}" if EXPLANATION_METHOD is None else f"{METRICS_LABELS[metric]}",
            )

            # # Add variance band (mean ± std)
            #plt.fill_between(
            #    x,
            #     y - y_std,
            #     y + y_std,
            #     color=color,
            #     alpha=0.2,
            #)

    #plt.xlabel(EVOLUTION_FACTORS_LABELS[EVOLUTION_FACTOR], fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    #plt.title(f"{dataset}")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    if EXPLANATION_METHOD != 'gradients':
        ax.set_ylim(min_y, max_y)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(
            handles,
            labels,
            fontsize=24,
            ncol=2,
            frameon=False, loc="upper left" if dataset != 'handoutlines' else "center left",
            handlelength=2.0, handleheight=1.5,
            columnspacing=1.0
        )

    if dataset == 'ford-a':
        plt.ylabel(Y_AXIS_LABEL, fontsize=24)
    else:
        # Keep grid
        ax.grid(True, axis="y")

        # Remove tick marks and tick labels
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)

    plt.tight_layout()
    out_file = OUT_DIR / f"{dataset}_{explainer}_{EVOLUTION_FACTOR}_{MODEL}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
