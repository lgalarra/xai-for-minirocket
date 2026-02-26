import os
import pandas as pd
import pathlib

import numpy as np

def parse_list(col):
    return col.astype(str).apply(
        lambda x: np.array([float(v) for v in x.split(";") if v != ""])
    )


methods_labels = {'extreme_feature_coalitions': 'EFC', 'shap': 'KernelSHAP', 'stratoshap-k1': 'ST-SHAP', 'gradients': 'Gradients'}

INPUT_DIR = "runtime-results-full"
desktop = pathlib.Path(INPUT_DIR)

# Collect all CSVs
records = []
for fname in desktop.glob('**/*.csv'):
    df = pd.read_csv(fname)
    records.append(df)

df = pd.concat(records, ignore_index=True)
df["runtimes_list"] = parse_list(df["runtimes-seconds"])
df["runtimes_p2p_list"] = parse_list(df["runtimes-p2p-seconds"])
df["runtimes_segmented_list"] = parse_list(df["runtimes-segmented-seconds"])

df["speedup_p2p_runs"] = df.apply(
    lambda r: r["runtimes_p2p_list"] / r["runtimes_list"], axis=1
)

df["speedup_segmented_runs"] = df.apply(
    lambda r: r["runtimes_segmented_list"] / r["runtimes_list"], axis=1
)


# Keep only rows where required columns are present
required_cols = [
    "dataset",
    "base_explainer",
    "runtimes-mean",
    "runtimes-p2p-mean",
    "runtimes-segmented-mean",
]

df = df.dropna(subset=required_cols)
df["dataset"] = df["dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)

df["dataset"] = df["dataset"].str.replace(
    r"^abnormal-heartbeat-c.*",
    "abnormal-heartbeat",
    regex=True,
)

print(df["dataset"].unique())

# Ensure numeric
for c in ["runtimes-mean", "runtimes-p2p-mean", "runtimes-segmented-mean"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[
    "runtimes-mean",
    "runtimes-p2p-mean",
    "runtimes-segmented-mean",
])

# Compute speed-ups
df["speedup_p2p"] = df["runtimes-p2p-mean"] / df["runtimes-mean"]
df["speedup_segmented"] = df["runtimes-segmented-mean"] / df["runtimes-mean"]


agg = (
    df
    .groupby(["dataset", "base_explainer"])
    .agg(
        p2p_mean=("speedup_p2p", "mean"),
        p2p_std=("speedup_p2p_runs", lambda x: np.std(np.concatenate(x.values))),
        p2p_min=("speedup_p2p_runs", lambda x: np.min(np.concatenate(x.values))),
        p2p_max=("speedup_p2p_runs", lambda x: np.max(np.concatenate(x.values))),

        seg_mean=("speedup_segmented", "mean"),
        seg_std=("speedup_segmented_runs", lambda x: np.std(np.concatenate(x.values))),
        seg_min=("speedup_segmented_runs", lambda x: np.min(np.concatenate(x.values))),
        seg_max=("speedup_segmented_runs", lambda x: np.max(np.concatenate(x.values))),
    )
    .reset_index()
)

assert df["speedup_p2p_runs"].apply(lambda x: isinstance(x, np.ndarray)).all()
assert df["speedup_segmented_runs"].apply(lambda x: isinstance(x, np.ndarray)).all()


# ---- Generate LaTeX table ----
latex_lines = []
latex_lines.append(r"\begin{table}")
latex_lines.append(r"\begin{tabular}{l l cccc cccc}")
latex_lines.append(r"\toprule")
latex_lines.append(
    r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Method}} & "
    r"\multicolumn{4}{c}{\textbf{End-to-end}} & "
    r"\multicolumn{4}{c}{\textbf{LEFTIST}} \\" + "\cmidrule(lr){3-6}  \cmidrule(lr){7-10}"
)
latex_lines.append(
    r" &  & "
    r"\textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} & "
    r"\textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\"
)
latex_lines.append(r"\midrule")



for dataset, group in agg.groupby('dataset'):
    first = True
    for j, row in group.iterrows():
        method = methods_labels[row["base_explainer"]]
        if dataset in ('cognitive-circles', 'abnormal-heartbeat'):
            values = (
                f"- & - & "
                f"- & - & "
                f"{row['seg_mean']:.2f} & {row['seg_std']:.2f} & "
                f"{row['seg_min']:.2f} & {row['seg_max']:.2f}"
            )        
        else:
            values = (
                f"{row['p2p_mean']:.2f} & {row['p2p_std']:.2f} & "
                f"{row['p2p_min']:.2f} & {row['p2p_max']:.2f} & "
                f"{row['seg_mean']:.2f} & {row['seg_std']:.2f} & "
                f"{row['seg_min']:.2f} & {row['seg_max']:.2f}"
            )

        if first:
            latex_lines.append(
                rf"\multirow{{{len(group)}}}{{*}}{{{dataset}}} & {method} & {values} \\"
            )
            first = False
        else:
            latex_lines.append(
                rf" & {method} & {values} \\"
            )
        if j % 4 == 3:
            latex_lines.append(r"\midrule")



latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\caption{Runtime comparison across datasets.}")
latex_lines.append(r"\label{tab:runtime}")
latex_lines.append(r"\end{table}")

print("\n".join(latex_lines))

