import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------
# Configuration
# ---------------------------------------
ROOT_DIR = Path("data")
OUTPUT_TEX = "nonzero_alpha_ratio.tex"
FLOAT_FMT = "{:.3f}"

# ---------------------------------------
# Collect ratios per method
# ---------------------------------------
ratios_per_method = defaultdict(list)

csv_files = ROOT_DIR.glob("**/alphas_instance_*.csv")

for csv_file in csv_files:
    # method is the parent directory name
    method = csv_file.parent.name

    # load csv (no header, second column is value)
    df = pd.read_csv(csv_file, header=None)

    values = df.iloc[:, 1]

    ratio_nonzero = (values != 0).mean()
    ratios_per_method[method].append(ratio_nonzero)

if not ratios_per_method:
    raise RuntimeError("No alphas_instance_*.csv files found.")

# ---------------------------------------
# Aggregate (mean per method)
# ---------------------------------------
methods = sorted(ratios_per_method.keys())
mean_ratios = {
    m: sum(ratios_per_method[m]) / len(ratios_per_method[m])
    for m in methods
}

# ---------------------------------------
# Generate LaTeX table
# ---------------------------------------
latex = []

col_spec = "c" * len(methods)

latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\begin{tabular}{" + col_spec + r"}")
latex.append(r"\toprule")

# Header
latex.append(" & ".join(methods) + r" \\")
latex.append(r"\midrule")

# Single data row
latex.append(
    " & ".join(FLOAT_FMT.format(mean_ratios[m]) for m in methods) + r" \\"
)

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\caption{Average ratio of non-zero alpha coefficients per method}")
latex.append(r"\label{tab:alpha-nonzero-ratio}")
latex.append(r"\end{table}")

# ---------------------------------------
# Write output
# ---------------------------------------
with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(latex))

print(f"LaTeX table written to {OUTPUT_TEX}")