import argparse
import math
import os
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


DATASET_NAMES = (
    "ford-a",
    "double-freq-test",
    "abnormal-heartbeat-c1",
    "starlight-c1",
    "starlight-c2",
    "starlight-c3",
    "cognitive-circles",
    "handoutlines",
)


def _default_dataset_output_path(dataset_name: str, filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "experiments" / "data" / dataset_name / filename


def _default_explanation_output_path(dataset_name: str, test_instance_id: int) -> Path:
    return _default_dataset_output_path(
        dataset_name,
        f"shapelet_tree_instance_{test_instance_id}_explanation.png",
    )


def _euclidean_window_matches(
    channel_values: np.ndarray,
    shapelet_values: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    shapelet_values = np.asarray(shapelet_values, dtype=np.float64).reshape(-1)
    channel_values = np.asarray(channel_values, dtype=np.float64).reshape(-1)
    shapelet_length = shapelet_values.shape[0]
    if shapelet_length == 0 or channel_values.shape[0] < shapelet_length:
        return []

    windows = np.lib.stride_tricks.sliding_window_view(channel_values, shapelet_length)
    distances = np.linalg.norm(windows - shapelet_values, axis=1)
    starts = np.flatnonzero(distances <= threshold)
    return [
        {
            "start": int(start),
            "end": int(start + shapelet_length),
            "distance": float(distances[start]),
            "threshold": float(threshold),
        }
        for start in starts
    ]


def _ensure_tshap_utils_on_path() -> None:
    tshap_repo_path = Path(__file__).resolve().parents[2] / "tshap"
    if tshap_repo_path.is_dir():
        os.environ.setdefault("TSHAP_REPO_PATH", str(tshap_repo_path))
        if str(tshap_repo_path) not in sys.path:
            sys.path.insert(0, str(tshap_repo_path))

    code_path = Path(__file__).resolve().parents[1] / "code"
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def _load_double_freq_test(n_samples: int | None) -> tuple:
    _ensure_tshap_utils_on_path()
    from utils import get_double_freq_test_for_classification

    if n_samples is None:
        n_samples = 250
    return get_double_freq_test_for_classification(n_samples=n_samples)


def load_dataset(dataset_name: str, n_samples: int | None) -> tuple:
    _ensure_tshap_utils_on_path()

    from utils import (
        get_abnormal_hearbeat_for_classification,
        get_cognitive_circles_data_for_classification,
        get_forda_for_classification,
        get_handoutlines_for_classification,
        get_starlightcurves_for_classification,
    )

    if dataset_name == "double-freq-test":
        return _load_double_freq_test(n_samples=n_samples)
    if dataset_name == "ford-a":
        return get_forda_for_classification()
    if dataset_name == "abnormal-heartbeat-c1":
        return get_abnormal_hearbeat_for_classification("1")
    if dataset_name == "starlight-c1":
        return get_starlightcurves_for_classification("1")
    if dataset_name == "starlight-c2":
        return get_starlightcurves_for_classification("2")
    if dataset_name == "starlight-c3":
        return get_starlightcurves_for_classification("3")
    if dataset_name == "cognitive-circles":
        return get_cognitive_circles_data_for_classification(
            str(Path(__file__).resolve().parents[1] / "data" / "cognitive-circles"),
            target_col="RealDifficulty",
            as_numpy=True,
        )
    if dataset_name == "handoutlines":
        return get_handoutlines_for_classification("1")
    raise ValueError(
        f"Dataset '{dataset_name}' is not supported by this script. "
        f"Supported datasets: {', '.join(DATASET_NAMES)}."
    )


def load_shapelet_tree_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as input_file:
        payload = pickle.load(input_file)

    if isinstance(payload, dict) and "classifier" in payload:
        return payload

    return {
        "classifier": payload,
        "shapelets": None,
        "detector_threshold": None,
        "detector_threshold_scale": 0.1,
        "feature_names": None,
    }


def shapelet_counts_and_matches_for_instance(
    instance: np.ndarray,
    shapelets: list[dict[str, Any]],
    threshold: float | None = None,
    threshold_scale: float = 0.1,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    counts = np.zeros(len(shapelets), dtype=np.int64)
    matches = []

    for shapelet_index, shapelet in enumerate(shapelets):
        channel = int(shapelet["channel"])
        if channel >= instance.shape[0]:
            locations = []
        else:
            channel_values = instance[channel]
            match_threshold = threshold
            if match_threshold is None:
                match_threshold = threshold_scale * float(np.std(channel_values))
            locations = _euclidean_window_matches(
                channel_values,
                shapelet["values"],
                match_threshold,
            )

        counts[shapelet_index] = len(locations)
        matches.append(
            {
                "shapelet_index": int(shapelet_index),
                "class": shapelet["class"],
                "cluster": shapelet["cluster"],
                "channel": channel,
                "count": len(locations),
                "locations": locations,
            }
        )

    return counts, matches


def decision_path_features(tree: Any, shapelet_counts: np.ndarray) -> list[dict[str, Any]]:
    node_indicator = tree.decision_path(shapelet_counts.reshape(1, -1))
    leaf_id = int(tree.apply(shapelet_counts.reshape(1, -1))[0])
    triggered_features = []

    for node_id in node_indicator.indices:
        if int(node_id) == leaf_id:
            continue

        feature_index = int(tree.tree_.feature[node_id])
        if feature_index < 0:
            continue

        threshold = float(tree.tree_.threshold[node_id])
        threshold_floor = math.floor(threshold)
        value = int(shapelet_counts[feature_index])
        triggered_features.append(
            {
                "node": int(node_id),
                "shapelet_index": feature_index,
                "count": value,
                "threshold": threshold,
                "threshold_floor": threshold_floor,
                "operator": "<=" if value <= threshold else ">",
                "rule": (
                    f"shapelet_{feature_index} <= {threshold_floor}"
                    if value <= threshold
                    else f"shapelet_{feature_index} > {threshold_floor}"
                ),
            }
        )

    return triggered_features


def format_prediction_rule(
    triggered_features: list[dict[str, Any]],
    predicted_label: Any,
) -> str:
    if not triggered_features:
        return f"predict {predicted_label}"

    conditions = [feature["rule"] for feature in triggered_features]
    return " and ".join(conditions) + f" -> predict {predicted_label}"


def plot_instance_with_shapelet_matches(
    instance: np.ndarray,
    triggered_features: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    shapelets: list[dict[str, Any]],
    prediction_rule: str,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_channels = instance.shape[0]
    n_shapelet_panels = max(1, len(triggered_features))
    max_shapelet_columns = 4
    n_shapelet_columns = min(max_shapelet_columns, n_shapelet_panels)
    n_shapelet_rows = math.ceil(n_shapelet_panels / n_shapelet_columns)
    fig = plt.figure(
        figsize=(max(20, 5.1 * n_shapelet_columns), max(10.5, 3.3 * n_channels + 2.4 * n_shapelet_rows + 1.4)),
        constrained_layout=True,
    )
    grid = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[max(3.5, 3.2 * n_channels), 1.25 + 2.2 * n_shapelet_rows],
        hspace=0.16,
    )
    series_grid = grid[0].subgridspec(n_channels, 1, hspace=0.12)
    shapelet_grid = grid[1].subgridspec(
        2 + n_shapelet_rows,
        n_shapelet_columns,
        height_ratios=[0.4, 0.18] + [1.0] * n_shapelet_rows,
        hspace=0.34,
        wspace=0.28,
    )
    axes = []
    for channel in range(n_channels):
        share_axis = axes[0] if axes else None
        axes.append(fig.add_subplot(series_grid[channel, 0], sharex=share_axis))
    time_axis = np.arange(instance.shape[-1])
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    triggered_indexes = [feature["shapelet_index"] for feature in triggered_features]
    matches_by_shapelet = {match["shapelet_index"]: match for match in matches}

    for channel, ax in enumerate(axes):
        ax.plot(time_axis, instance[channel], color="black", linewidth=1.2)
        ax.set_ylabel(f"channel {channel}")
        ax.grid(True, alpha=0.25)
        if channel < n_channels - 1:
            ax.tick_params(labelbottom=False)

        for rank, shapelet_index in enumerate(triggered_indexes):
            match = matches_by_shapelet.get(shapelet_index)
            if match is None or match["channel"] != channel:
                continue

            color = color_cycle[rank % len(color_cycle)]
            label = f"shapelet {shapelet_index} (count={match['count']})"
            for location_index, location in enumerate(match["locations"]):
                ax.axvspan(
                    location["start"],
                    location["end"],
                    color=color,
                    alpha=0.25,
                    label=label if location_index == 0 else None,
                )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    axes[-1].set_xlabel("time")
    fig.suptitle(title, fontsize=13)

    rule_ax = fig.add_subplot(shapelet_grid[0, :])
    rule_ax.axis("off")
    wrapped_rule = textwrap.fill(f"Rule: {prediction_rule}", width=140)
    rule_ax.text(
        0.0,
        0.92,
        wrapped_rule,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
        transform=rule_ax.transAxes,
    )

    shapelet_title_ax = fig.add_subplot(shapelet_grid[1, :])
    shapelet_title_ax.axis("off")
    shapelet_title_ax.text(
        0.0,
        0.15,
        "Triggered shapelets",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        transform=shapelet_title_ax.transAxes,
    )

    for rank, feature in enumerate(triggered_features):
        shapelet_index = int(feature["shapelet_index"])
        if shapelet_index < 0 or shapelet_index >= len(shapelets):
            continue

        shapelet = shapelets[shapelet_index]
        values = np.asarray(shapelet["values"], dtype=np.float64).reshape(-1)
        color = color_cycle[rank % len(color_cycle)]

        row = 2 + rank // n_shapelet_columns
        column = rank % n_shapelet_columns
        mini_ax = fig.add_subplot(shapelet_grid[row, column])
        mini_ax.plot(np.arange(values.shape[0]), values, color=color, linewidth=2.1)
        mini_ax.set_box_aspect(0.42)
        mini_ax.margins(x=0.06, y=0.28)
        mini_ax.grid(True, alpha=0.2)
        mini_ax.tick_params(axis="both", labelsize=10)
        mini_ax.set_title(
            f"s{shapelet_index}: {feature['rule']}",
            fontsize=12,
            loc="left",
            pad=6,
        )
        for spine in mini_ax.spines.values():
            spine.set_alpha(0.35)

    for rank in range(len(triggered_features), n_shapelet_rows * n_shapelet_columns):
        row = 2 + rank // n_shapelet_columns
        column = rank % n_shapelet_columns
        fig.add_subplot(shapelet_grid[row, column]).axis("off")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-pickle-path", type=Path, default=None)
    parser.add_argument("--dataset-name", default="double-freq-test", choices=DATASET_NAMES)
    parser.add_argument("--dataset-n-samples", type=int, default=None)
    parser.add_argument("--test-instance-id", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    if args.tree_pickle_path is None:
        args.tree_pickle_path = _default_dataset_output_path(args.dataset_name, "shapelet_decision_tree.pkl")
    if args.output_path is None:
        args.output_path = _default_explanation_output_path(args.dataset_name, args.test_instance_id)

    bundle = load_shapelet_tree_bundle(args.tree_pickle_path)
    tree = bundle["classifier"]
    shapelets = bundle.get("shapelets")
    if not shapelets:
        raise ValueError(
            "The tree pickle does not contain shapelets. Re-run whitebox_classifier.py "
            "so the tree is saved as a bundle with detector shapelets."
        )

    metadata = bundle.get("metadata", {})
    dataset_n_samples = args.dataset_n_samples
    if dataset_n_samples is None:
        dataset_n_samples = int(metadata.get("dataset_n_samples", 250))

    (_, _), (X_test, y_test) = load_dataset(args.dataset_name, n_samples=dataset_n_samples)
    if args.test_instance_id < 0 or args.test_instance_id >= len(X_test):
        raise ValueError(
            f"test-instance-id must be in [0, {len(X_test) - 1}], got {args.test_instance_id}."
        )

    instance = np.asarray(X_test[args.test_instance_id], dtype=np.float64)
    counts, matches = shapelet_counts_and_matches_for_instance(
        instance,
        shapelets,
        threshold=bundle.get("detector_threshold"),
        threshold_scale=float(bundle.get("detector_threshold_scale", 0.1)),
    )
    predicted_label = tree.predict(counts.reshape(1, -1))[0]
    triggered_features = decision_path_features(tree, counts)
    prediction_rule = format_prediction_rule(triggered_features, predicted_label)

    plot_instance_with_shapelet_matches(
        instance,
        triggered_features,
        matches,
        shapelets,
        prediction_rule,
        args.output_path,
        title=(
            f"{args.dataset_name} test[{args.test_instance_id}] "
            f"true={int(y_test[args.test_instance_id])} predicted={predicted_label}"
        ),
    )

    print(f"tree_pickle_path={args.tree_pickle_path}")
    print(f"dataset_name={args.dataset_name}")
    print(f"dataset_n_samples={dataset_n_samples}")
    print(f"test_instance_id={args.test_instance_id}")
    print(f"true_label={int(y_test[args.test_instance_id])}")
    print(f"predicted_label={predicted_label}")
    print(f"prediction_rule={prediction_rule}")
    print(f"shapelet_counts={counts.tolist()}")
    print(f"triggered_features={triggered_features}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
