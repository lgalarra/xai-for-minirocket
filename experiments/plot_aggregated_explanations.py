"""Plot pointwise aggregates of precomputed reference-policy explanations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = {"max": np.max, "avg": np.mean, "sum": np.sum, "min": np.min}


def aggregate_explanations(explanations, aggregation):
    """Reduce equal-shaped, finite attribution arrays across policies only."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    arrays = [np.asarray(values, dtype=float) for values in explanations]
    if not arrays or any(values.size == 0 for values in arrays):
        raise ValueError("At least one nonempty explanation is required.")
    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("Explanation shapes must match across reference policies.")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Explanations must contain only finite values.")
    result = AGGREGATIONS[aggregation](np.stack(arrays), axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Aggregation produced nonfinite values.")
    return result


def read_series(filename, export_root):
    path = Path(filename)
    if not path.is_absolute():
        path = export_root / path
    frame = pd.read_csv(path, header=None, index_col=0)
    if frame.shape[1] != 1 or frame.empty or frame.index.has_duplicates:
        raise ValueError(f"Expected an indexed, single-value-column series: {path}")
    values = frame.iloc[:, 0].astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values in {path}")
    return values


def load_policy_metadata(directory, policy, attribution):
    paths = sorted(directory.glob(f"metadata-*-*_ref_policy_{policy}.csv"))
    policy_path = directory / f"metadata_ref_policy_{policy}.csv"
    if policy_path.exists():
        paths.insert(0, policy_path)
    if not paths:
        paths = [directory / "metadata.csv"]
    metadata = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"instance_id", "channel", "series", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {required - set(metadata.columns)}")
    reference_columns = [
        column for column in metadata.columns
        if column.startswith("reference_") and column.removeprefix("reference_").isdigit()
    ]
    if paths == [directory / "metadata.csv"]:
        policies = (
            "opposite_class_closest_instance", "opposite_class_medoid",
            "opposite_class_centroid", "global_medoid", "global_centroid",
            "opposite_class_farthest_instance", "counterfactual", "counterfactual_dtw",
        )
        if policy not in policies:
            raise ValueError(f"Unknown legacy reference policy: {policy}")
        reference_columns = [f"reference_{policies.index(policy)}"]
    else:
        reference_columns = [column for column in reference_columns if metadata[column].notna().any()]
    if len(reference_columns) != 1:
        raise ValueError(f"Cannot identify a unique reference column for {policy}")
    policy_index = reference_columns[0].removeprefix("reference_")
    column = f"{attribution}_{policy_index}_attributions"
    if column not in metadata:
        raise ValueError(f"Missing {column} for {policy}")
    metadata = metadata.dropna(subset=[column])
    metadata = metadata[["instance_id", "channel", "series", "label", column]].drop_duplicates()
    if metadata.duplicated(["instance_id", "channel"]).any():
        raise ValueError(f"Conflicting metadata rows for {policy}")
    if metadata.empty:
        raise ValueError(f"No precomputed {attribution} explanations for {policy}")
    return metadata.rename(columns={column: "attribution"}).set_index(["instance_id", "channel"])


def plot_instance(channels, output, title, dpi=200):
    """Overlay a signed heatmap behind each channel's time series."""
    limit = max(float(np.max(np.abs(values))) for _, _, values in channels) or 1.0
    figure, axes = plt.subplots(
        len(channels), 1, figsize=(12, 2.8 * len(channels)),
        squeeze=False, layout="constrained",
    )
    try:
        for axis, (name, series, values) in zip(axes[:, 0], channels):
            lower, upper = float(np.min(series)), float(np.max(series))
            padding = (upper - lower) * 0.08 or max(abs(lower) * 0.08, 0.5)
            heatmap = axis.imshow(
                values[np.newaxis, :], aspect="auto", interpolation="nearest",
                extent=(-0.5, len(series) - 0.5, lower - padding, upper + padding),
                cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=0.7,
            )
            axis.plot(np.arange(len(series)), series, color="black", linewidth=1)
            axis.set_ylabel(str(name))
            axis.set_xlim(-0.5, len(series) - 0.5)
        axes[-1, 0].set_xlabel("Time step")
        figure.colorbar(heatmap, ax=axes[:, 0].tolist(), label="Aggregated attribution")
        figure.suptitle(title)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--explainer", required=True)
    parser.add_argument("--reference-policies", required=True, help="Comma-separated policy names")
    parser.add_argument("--aggregation", choices=AGGREGATIONS, required=True)
    parser.add_argument("--label", choices=("predicted", "training"), default="predicted")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--attribution", default="beta", help="Column prefix, e.g. beta, beta_p2p, beta_lrp, beta_segmented, beta_tshap_w10_s5")
    parser.add_argument("--instances", type=lambda value: [int(item) for item in value.split(",")], help="Comma-separated instance IDs; default: all")
    parser.add_argument("--export-root", type=Path, default=REPOSITORY_ROOT / "experiments", help="Directory containing exported data/ and resolving metadata paths")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "reports" / "aggregated_explanations")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    policies = [policy.strip() for policy in args.reference_policies.split(",")]
    if not all(policies) or len(set(policies)) != len(policies):
        parser.error("Reference policies must be nonempty and unique.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    directory = args.export_root / "data" / args.dataset / args.classifier / args.explainer / args.label
    if (directory / args.metric).exists() or args.metric != "euclidean":
        directory = directory / args.metric
    metadata = [load_policy_metadata(directory, policy, args.attribution) for policy in policies]
    if args.instances is not None:
        metadata = [frame.loc[frame.index.get_level_values("instance_id").isin(args.instances)] for frame in metadata]
    keys = set(metadata[0].index)
    if not keys or any(set(frame.index) != keys for frame in metadata):
        raise ValueError("Selected policies must have matching, nonempty instance/channel coverage.")
    instance_ids = sorted({key[0] for key in keys})
    if args.instances is not None and set(args.instances) != set(instance_ids):
        raise ValueError("Some requested instance IDs are missing from the exports.")
    for instance_id in instance_ids:
        channels = []
        for key in sorted(key for key in keys if key[0] == instance_id):
            rows = [frame.loc[key] for frame in metadata]
            series = read_series(rows[0]["series"], args.export_root)
            explanations = []
            for policy, row in zip(policies, rows):
                other_series = read_series(row["series"], args.export_root)
                values = read_series(row["attribution"], args.export_root)
                if row["label"] != rows[0]["label"] or not series.equals(other_series):
                    raise ValueError(f"Mismatched series or target label for {key}, {policy}")
                if not series.index.equals(values.index):
                    raise ValueError(f"Mismatched attribution time indices for {key}, {policy}")
                explanations.append(values.to_numpy())
            channels.append((key[1], series.to_numpy(), aggregate_explanations(explanations, args.aggregation)))
        output = args.output_dir / f"{args.dataset}_{args.classifier}_{args.explainer}_{args.aggregation}_instance_{instance_id}.png"
        title = f"{args.dataset} · {args.classifier} · {args.explainer} · instance {instance_id}\n{args.aggregation}: {', '.join(policies)}"
        plot_instance(channels, output, title, args.dpi)
        print(output)


if __name__ == "__main__":
    main()
