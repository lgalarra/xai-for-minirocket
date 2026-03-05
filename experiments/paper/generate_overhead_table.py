import pandas as pd

# Load the TSV file
df = pd.read_csv("traces_overhead.tsv", sep="\t")

df["Dataset"] = df["Dataset"].str.replace(
    r"^starlight-c.*",
    "starlight",
    regex=True,
)
df["Dataset"] = df["Dataset"].str.replace(
    r"^abnormal-heartbeat-c.*",
    "abnormal-heartbeat",
    regex=True,
)
df["Overhead"] = df["Time RT"] / df["Time NRT"]

# Compute mean and standard deviation of Overhead per dataset
stats = (
    df.groupby("Dataset")["Overhead"]
      .agg(["mean", "std"])
      .reset_index()
)


# Build LaTeX table components
datasets = stats["Dataset"].tolist()
values = [
    f"{row['mean']:.2f} ({row['std']:.2f})"
    for _, row in stats.iterrows()
]

# Generate LaTeX table
latex_table = []
latex_table.append("\\begin{table}[t]")
latex_table.append("\\centering")
latex_table.append(
    "\\begin{tabular}{" + "c" * len(datasets) + "}"
)
latex_table.append("\\toprule")
latex_table.append(" & ".join(datasets) + " \\\\")
latex_table.append("\\midrule")
latex_table.append(" & ".join(values) + " \\\\")
latex_table.append("\\bottomrule")
latex_table.append("\\end{tabular}")
latex_table.append("\\caption{Average runtime overhead at prediction time incurred by back-propagation (standard deviation in parentheses).}")
latex_table.append("\\label{tab:overhead}")
latex_table.append("\\end{table}")

# Print LaTeX table
print("\n".join(latex_table))