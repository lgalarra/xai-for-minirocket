import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------
INPUT_DIR = Path("runtime-results-full")
OUTPUT_TEX = "complexity_table.tex"
METHOD_LABELS = {'shap': "KernelSHAP", "stratoshap-k1": "ST-SHAP", "gradients": "Gradients",
                 "extreme_feature_coalitions": "EFC"}
CLASSIFIER_LABELS = {"RandomForestClassifier": "Random Forest", "MLPClassifier": "Multilayer Perceptron",
                     "LogisticRegression": "Logistic Regression"}

VALUE_COLUMNS = {
    "BP": "complexity-mean",
    "e2e": "complexity-p2p-mean",
    "LFT": "complexity-segmented-mean",
}

METHOD = "extreme_feature_coalitions"


FLOAT_FMT = "{:.0f}"

# -------------------------------------------------
# Load and concatenate all CSV files
# -------------------------------------------------
csv_files = sorted(INPUT_DIR.glob("**/*.csv"))
if not csv_files:
    raise RuntimeError(f"No CSV files found in {INPUT_DIR}")

dfs = []
for csv in csv_files:
    df = pd.read_csv(csv)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df[df["base_explainer"] == METHOD]

df["dataset"] = df["dataset"].str.replace(
    r"^abnormal-heartbeat-c.*",
    "abnormal-heartbeat",
    regex=True,
)


# -------------------------------------------------
# Keep relevant columns only
# -------------------------------------------------
required_cols = ["dataset", "mr_classifier"] + list(VALUE_COLUMNS.values())
missing = set(required_cols) - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns in CSVs: {missing}")

df = df[required_cols]

# -------------------------------------------------
# Aggregate across runs
# -------------------------------------------------
agg = (
    df
    .groupby(["dataset", "mr_classifier"], as_index=False)
    .mean()
)

datasets = sorted(agg["dataset"].unique())
classifiers = sorted(agg["mr_classifier"].unique())

# -------------------------------------------------
# Build LaTeX table
# -------------------------------------------------
latex = []

n_subcols = len(VALUE_COLUMNS)
col_spec = "l" + "c" * (len(classifiers) * n_subcols)

latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\begin{tabular}{" + col_spec + r"}")
latex.append(r"\toprule")

# Header row 1: classifier groups
header_1 = ["Dataset"]
for clf in classifiers:
    header_1.append(
        r"\multicolumn{" + str(n_subcols) + r"}{c}{" + CLASSIFIER_LABELS[clf] + "}"
    )
latex.append(" & ".join(header_1) + r" \\")

latex.append(
    r"\cmidrule(lr){2-" +
    str(1 + len(classifiers) * n_subcols) +
    r"}"
)

# Header row 2: method names
header_2 = [""]
for _ in classifiers:
    header_2.extend(VALUE_COLUMNS.keys())
latex.append(" & ".join(header_2) + r" \\")
latex.append(r"\midrule")

# Data rows
for dataset in datasets:
    row = [dataset]
    for clf in classifiers:
        subset = agg[
            (agg["dataset"] == dataset) &
            (agg["mr_classifier"] == clf)
        ]
        if subset.empty:
            row.extend(["--"] * n_subcols)
        else:
            for col in VALUE_COLUMNS.values():
                if subset.iloc[0][col] != -1:
                    row.append(FLOAT_FMT.format(subset.iloc[0][col]))
                else:
                    row.append("-")

    latex.append(" & ".join(row) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\caption{Average number of non-zero attributions per dataset and classifier for " + METHOD_LABELS[METHOD]
             + ". BP stands for back-propagated, e2e means end-to-end and LFT means LEFTIST}")
latex.append(r"\label{tab:complexity_" + METHOD + "}")
latex.append(r"\end{table}")

# -------------------------------------------------
# Write output
# -------------------------------------------------
with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(latex))

print(f"LaTeX table written to {OUTPUT_TEX}")