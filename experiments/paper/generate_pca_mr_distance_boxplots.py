from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


DEFAULT_PCA_MR_DIR = Path("/home/luis/Desktop/abnormal-heartbeat-newdistance")
DEFAULT_NON_PCA_MR_DIR = Path("/home/luis/Desktop/abnormal-heartbeat-olddistance")
DEFAULT_OUT_DIR = Path("experiments/paper/pca_mr_distance_boxplots")

CLASSIFIER_LABELS = {
    "LogisticRegression": "LR",
    "RandomForestClassifier": "RF",
    "MLPClassifier": "MLP",
}

REFERENCE_POLICY_LABELS = {
    "global_centroid": "centroid",
    "global_medoid": "medoid",
    "opposite_class_centroid": "enemy centroid",
    "opposite_class_medoid": "enemy medoid",
    "opposite_class_farthest_instance": "farthest enemy",
    "opposite_class_closest_instance": "closest enemy",
}

DISTANCE_ORDER = ["non pca-mr", "pca-mr"]
DEFAULT_EXCLUDED_REFERENCE_POLICIES = [
    "global_centroid",
    "opposite_class_centroid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand semicolon-separated f_minus_f0 values and plot comparative "
            "box plots for PCA-MR vs non-PCA-MR results per classifier."
        )
    )
    parser.add_argument("--pca-mr-dir", type=Path, default=DEFAULT_PCA_MR_DIR)
    parser.add_argument("--non-pca-mr-dir", type=Path, default=DEFAULT_NON_PCA_MR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--glob", default="*.csv", help="CSV file glob to read from each directory.")
    parser.add_argument("--base-explainer", default=None, help="Optional base_explainer filter.")
    parser.add_argument("--label", default=None, help="Optional label filter, e.g. predicted.")
    parser.add_argument("--perturbation-policy", default=None, help="Optional perturbation_policy filter.")
    parser.add_argument(
        "--exclude-reference-policy",
        action="append",
        default=DEFAULT_EXCLUDED_REFERENCE_POLICIES,
        help=(
            "Reference policy to exclude. Can be passed multiple times. "
            "Defaults to excluding global_centroid and opposite_class_centroid."
        ),
    )
    return parser.parse_args()


def infer_distance_type(df: pd.DataFrame, csv_file: Path, fallback: str) -> str:
    for column in ("metric", "distance"):
        if column in df.columns:
            values = df[column].dropna().astype(str).unique()
            if any(value == "pca-mr" for value in values):
                return "pca-mr"

    if csv_file.stem.endswith("_pca-mr"):
        return "pca-mr"

    return fallback


def parse_semicolon_values(value: object) -> list[float]:
    if pd.isna(value):
        return []

    values = []
    for raw in str(value).split(";"):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    return values


def load_results(directory: Path, fallback_distance_type: str, glob_pattern: str) -> pd.DataFrame:
    frames = []
    csv_files = sorted(directory.glob(glob_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {glob_pattern!r} in {directory}")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        required = {"f_minus_f0", "mr_classifier", "reference_policy"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["distance_type"] = infer_distance_type(df, csv_file, fallback_distance_type)
        df["source_file"] = csv_file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def apply_filters(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = data
    filters = {
        "base_explainer": args.base_explainer,
        "label": args.label,
        "perturbation_policy": args.perturbation_policy,
    }
    for column, value in filters.items():
        if value is not None:
            if column not in filtered.columns:
                raise ValueError(f"Cannot filter by {column!r}: column not found.")
            filtered = filtered[filtered[column] == value]

    if args.exclude_reference_policy:
        filtered = filtered[
            ~filtered["reference_policy"].isin(args.exclude_reference_policy)
        ]

    return filtered


def expand_f_minus_f0(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["f_minus_f0_value"] = data["f_minus_f0"].apply(parse_semicolon_values)
    data = data.explode("f_minus_f0_value", ignore_index=True)
    data["f_minus_f0_value"] = pd.to_numeric(data["f_minus_f0_value"], errors="coerce")
    return data.dropna(subset=["f_minus_f0_value"])


def normalize_labels(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["mr_classifier_label"] = data["mr_classifier"].replace(CLASSIFIER_LABELS)
    data["reference_policy_label"] = data["reference_policy"].replace(REFERENCE_POLICY_LABELS)
    return data


def save_classifier_boxplots(data: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    written = []
    for classifier, group in data.groupby("mr_classifier_label", sort=True):
        if group.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        reference_order = sorted(group["reference_policy_label"].unique())
        distance_order = [item for item in DISTANCE_ORDER if item in set(group["distance_type"])]
        colors = {"non pca-mr": "#4c78a8", "pca-mr": "#f58518"}

        box_width = 0.32
        offsets = {
            "non pca-mr": -box_width / 1.8,
            "pca-mr": box_width / 1.8,
        }

        for distance_type in distance_order:
            values = []
            positions = []
            for index, reference_policy in enumerate(reference_order, start=1):
                subset = group[
                    (group["reference_policy_label"] == reference_policy)
                    & (group["distance_type"] == distance_type)
                ]["f_minus_f0_value"].to_numpy()
                if len(subset) == 0:
                    continue
                values.append(subset)
                positions.append(index + offsets[distance_type])

            if values:
                box = ax.boxplot(
                    values,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={
                        "marker": ".",
                        "markersize": 1.5,
                        "markeredgecolor": colors[distance_type],
                        "alpha": 0.35,
                    },
                    medianprops={"color": "black", "linewidth": 1.0},
                    boxprops={"linewidth": 1.0},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(colors[distance_type])
                    patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(reference_order) + 1))
        ax.set_xticklabels(reference_order, rotation=20, ha="right")
        ax.set_xlabel("Reference policy")
        ax.set_ylabel("f_minus_f0")
        ax.set_title(str(classifier))
        ax.legend(
            handles=[
                Patch(facecolor=colors[item], alpha=0.75, label=item)
                for item in distance_order
            ],
            title="Distance",
            frameon=True,
        )
        fig.tight_layout()

        safe_classifier = str(classifier).replace("/", "_").replace(" ", "_")
        out_file = out_dir / f"{safe_classifier}_f_minus_f0_reference_policy_pca_mr_boxplot.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(out_file)

    return written


def main() -> None:
    args = parse_args()

    data = pd.concat(
        [
            load_results(args.non_pca_mr_dir, "non pca-mr", args.glob),
            load_results(args.pca_mr_dir, "pca-mr", args.glob),
        ],
        ignore_index=True,
    )
    data = apply_filters(data, args)
    if data.empty:
        raise ValueError("No rows left after applying filters.")

    data = expand_f_minus_f0(data)
    data = normalize_labels(data)
    written = save_classifier_boxplots(data, args.out_dir)

    print(f"Read rows: {len(data):,}")
    print(f"Wrote {len(written)} plot(s):")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
