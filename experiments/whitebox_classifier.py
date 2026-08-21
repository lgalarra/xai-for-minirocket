import argparse
import csv
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

import minirocket_multivariate_variable as mmv
from explainer import get_classifier_explainer


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "data"

RUN_SUMMARY_TSV_FIELDS = [
    "run_started_at_utc",
    "run_finished_at_utc",
    "run_id",
    "job_id",
    "classifier_name",
    "classifier_path",
    "tree_pickle_path",
    "tree_png_path",
    "dataset_name",
    "classifier_explainer",
    "reference_policy",
    "score_mode",
    "sample_size",
    "computed_alpha_attributions",
    "random_state",
    "random_state_grid",
    "sample_random_state",
    "n_jobs_requested",
    "n_jobs_resolved",
    "validation_size",
    "train_size",
    "test_size",
    "validation_train_size",
    "validation_size_count",
    "top_k_grid",
    "n_clusters_grid",
    "threshold_scale_grid",
    "max_depth_grid",
    "grid_candidates",
    "minirocket_train_accuracy",
    "minirocket_test_accuracy",
    "best_top_k",
    "best_n_clusters",
    "best_threshold_scale",
    "best_max_depth",
    "best_validation_accuracy",
    "best_test_accuracy_before_refit",
    "shapelet_decision_tree_accuracy",
    "n_shapelets",
    "shapelet_count_train_rows",
    "shapelet_count_train_columns",
    "shapelet_count_test_rows",
    "shapelet_count_test_columns",
    "decision_tree_depth",
    "decision_tree_leaves",
    "runtime_load_classifier_seconds",
    "runtime_compute_alpha_attributions_seconds",
    "runtime_load_data_and_split_validation_seconds",
    "runtime_grid_search_total_seconds",
    "runtime_refit_best_decision_tree_seconds",
    "runtime_save_decision_tree_seconds",
    "runtime_total_seconds",
    "runtime_breakdown_seconds",
]


def _training_data(classifier: Any) -> np.ndarray:
    for attr in ("_X_train", "X_train"):
        X_train = getattr(classifier, attr, None)
        if X_train is not None:
            return X_train
    raise ValueError("The MinirocketClassifier does not store its training set.")


def _training_labels(classifier: Any) -> np.ndarray:
    for attr in ("_y_train", "y_train"):
        y_train = getattr(classifier, attr, None)
        if y_train is not None:
            return np.asarray(y_train)
    raise ValueError("The MinirocketClassifier does not store its training labels.")


def _load_dataset_for_classifier(classifier: Any, dataset_name: str) -> tuple:
    tshap_repo_path = Path(__file__).resolve().parents[2] / "tshap"
    if tshap_repo_path.is_dir():
        os.environ.setdefault("TSHAP_REPO_PATH", str(tshap_repo_path))
        if str(tshap_repo_path) not in sys.path:
            sys.path.insert(0, str(tshap_repo_path))

    from utils import (
        get_abnormal_hearbeat_for_classification,
        get_cognitive_circles_data_for_classification,
        get_double_freq_test_for_classification,
        get_forda_for_classification,
        get_handoutlines_for_classification,
        get_starlightcurves_for_classification,
    )

    if dataset_name == "double-freq-test":
        return get_double_freq_test_for_classification(n_samples=len(_training_data(classifier)))
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
            "../data/cognitive-circles",
            target_col="RealDifficulty",
            as_numpy=True,
        )
    if dataset_name == "handoutlines":
        return get_handoutlines_for_classification("1")
    raise ValueError(f"Dataset loading is not configured for '{dataset_name}'.")


def _record_runtime(runtimes: dict[str, float], stage: str, start: float) -> None:
    runtime = time.perf_counter() - start
    runtimes[stage] = runtime
    print(f"runtime_{stage}_seconds={runtime:.6f}", flush=True)


def _json_dumps_for_tsv(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _default_dataset_output_path(dataset_name: str, filename: str) -> Path:
    return DEFAULT_DATA_ROOT / dataset_name / filename


def _default_classifier_path(dataset_name: str, classifier_name: str) -> Path:
    classifier_filename = f"{Path(classifier_name).stem}.pkl"
    return _default_dataset_output_path(dataset_name, classifier_filename)


def _write_run_summary_tsv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    serialized_row = {}
    for field in RUN_SUMMARY_TSV_FIELDS:
        value = row.get(field, "")
        if isinstance(value, float):
            serialized_row[field] = f"{value:.12g}"
        elif isinstance(value, (list, tuple, dict)):
            serialized_row[field] = _json_dumps_for_tsv(value)
        elif value is None:
            serialized_row[field] = ""
        else:
            serialized_row[field] = str(value)

    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=RUN_SUMMARY_TSV_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(serialized_row)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer or -1.")
    return n_jobs


def _parse_int_grid(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return values


def _parse_optional_int_grid(value: str) -> list[int | None]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() == "none":
            values.append(None)
        else:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer or None value.")
    return values


def _parse_float_grid(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _dedupe_preserving_order(values: list[Any]) -> list[Any]:
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values


def _balanced_sample_indexes(
    labels: np.ndarray,
    sample_size: int | None,
    random_state: int | None,
) -> np.ndarray:
    if sample_size is None:
        return np.arange(labels.shape[0])
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    classes, counts = np.unique(labels, return_counts=True)
    if sample_size < classes.shape[0]:
        raise ValueError(
            f"sample_size must be at least the number of predicted classes ({classes.shape[0]})."
        )

    per_class = min(sample_size // classes.shape[0], counts.min())
    rng = np.random.default_rng(random_state)
    sampled_indexes = []
    for predicted_class in classes:
        class_indexes = np.flatnonzero(labels == predicted_class)
        sampled_indexes.extend(
            rng.choice(class_indexes, size=per_class, replace=False).tolist()
        )
    return np.asarray(sampled_indexes, dtype=int)


def _as_hashable_label(label: Any) -> Hashable:
    return label.item() if isinstance(label, np.generic) else label


def _positive_attribution_segments(
    instance: np.ndarray,
    attributions: np.ndarray,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    instance = np.asarray(instance)
    attributions = np.asarray(attributions)
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)
    if attributions.ndim == 1:
        attributions = attributions.reshape(1, -1)
    if attributions.shape != instance.shape:
        if attributions.T.shape == instance.shape:
            attributions = attributions.T
        else:
            raise ValueError(
                f"Attribution shape {attributions.shape} is incompatible with instance shape {instance.shape}."
            )

    segments = []
    for channel in range(attributions.shape[0]):
        positive = attributions[channel] > threshold
        starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
        ends = np.flatnonzero(positive & np.r_[~positive[1:], True]) + 1
        for start, end in zip(starts, ends):
            segments.append(
                {
                    "channel": int(channel),
                    "start": int(start),
                    "end": int(end),
                    "values": instance[channel, start:end].copy(),
                    "attributions": attributions[channel, start:end].copy(),
                }
            )
    return segments


def _alpha_attributions_for_instance(
    classifier: Any,
    explainer: Any,
    x_target: np.ndarray,
    predicted_class: Hashable,
    classifier_explainer: str | Callable,
    reference_policy: str,
    dataset_name: str | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reference = explainer.get_reference(
        x_target,
        predicted_class,
        reference_policy,
        dataset_name=dataset_name,
    )
    out_x = mmv.transform_prime(x_target, parameters=classifier.minirocket_params)
    reference_mr = mmv.transform_prime(reference, parameters=classifier.minirocket_params)

    classifier_explainer_fn = get_classifier_explainer(
        classifier_explainer,
        lambda x: classifier.classifier.predict_proba(x)[:, predicted_class],
        X_background=np.array([reference_mr["phi"][0]]),
        target=out_x["phi"][0],
    )
    attributions = classifier_explainer_fn(out_x["phi"])
    if attributions.shape[0] == 1:
        attributions = attributions[0]
    return np.asarray(attributions, dtype=float).reshape(-1), reference, out_x


def compute_alpha_attributions_for_balanced_sample(
    classifier: Any,
    explainer: Any,
    classifier_explainer: str | Callable = "shap",
    reference_policy: str = "global_centroid",
    dataset_name: str | None = None,
    sample_size: int | None = None,
    random_state: int | None = 0,
    n_jobs: int | None = 1,
) -> list[dict[str, Any]]:
    """
    Compute alpha attributions once for a balanced training sample.

    Alpha attribution for each sampled instance is independent, so callers can
    set ``n_jobs`` above 1 to compute several instances concurrently. The
    implementation uses threads instead of processes because the trained
    MiniROCKET classifier and explainer can be large and expensive to pickle;
    each thread still builds its own classifier-level explainer for the target
    instance.

    Parameters
    ----------
    classifier:
        Trained MiniRocketClassifier that stores its training data.

    explainer:
        MiniROCKET explainer used to obtain the reference time series.

    classifier_explainer:
        Feature attribution method passed to ``get_classifier_explainer``.

    reference_policy:
        Strategy used by ``explainer.get_reference``.

    dataset_name:
        Optional dataset name forwarded to reference policies that need it.

    sample_size:
        Optional total sample size, balanced across predicted classes.

    random_state:
        Seed used by the balanced sampler.

    n_jobs:
        Number of worker threads. Use ``1`` for sequential execution, or ``-1``
        to use all available CPU cores. Results are returned in the same order
        as the balanced sample indexes, independently of completion order.
    """
    X_train = _training_data(classifier)
    predicted_classes = np.asarray(classifier.predict(X_train))
    sampled_indexes = _balanced_sample_indexes(predicted_classes, sample_size, random_state)
    n_workers = min(_resolve_n_jobs(n_jobs), len(sampled_indexes))

    def compute_record(sampled_index: int) -> dict[str, Any]:
        x_target = X_train[sampled_index]
        predicted_class = _as_hashable_label(predicted_classes[sampled_index])
        alphas, reference, out_x = _alpha_attributions_for_instance(
            classifier,
            explainer,
            x_target,
            predicted_class,
            classifier_explainer,
            reference_policy,
            dataset_name,
        )
        return {
            "sample_index": int(sampled_index),
            "predicted_class": predicted_class,
            "instance": x_target,
            "reference": reference,
            "out_x": out_x,
            "alphas": alphas,
        }

    if n_workers == 1:
        alpha_records = []
        for sampled_index in sampled_indexes:
            alpha_records.append(compute_record(int(sampled_index)))
        return alpha_records

    alpha_records: list[dict[str, Any] | None] = [None] * len(sampled_indexes)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(compute_record, int(sampled_index)): position
            for position, sampled_index in enumerate(sampled_indexes)
        }
        for future in as_completed(futures):
            alpha_records[futures[future]] = future.result()

    return [record for record in alpha_records if record is not None]


def rank_top_minirocket_features_by_predicted_class_from_alphas(
    alpha_attributions: list[dict[str, Any]],
    top_k: int = 10,
    score_mode: str = "positive",
) -> dict[Any, list[int]]:
    """
    Aggregate precomputed alpha attributions and return top-k features per class.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    class_scores: dict[Hashable, np.ndarray] = {}
    for alpha_record in alpha_attributions:
        predicted_class = _as_hashable_label(alpha_record["predicted_class"])
        attributions = np.asarray(alpha_record["alphas"], dtype=float).reshape(-1)
        if score_mode == "positive":
            scores = np.maximum(attributions, 0.0)
        elif score_mode == "absolute":
            scores = np.abs(attributions)
        elif score_mode == "raw":
            scores = attributions
        else:
            raise ValueError("score_mode must be one of: positive, absolute, raw.")

        if predicted_class not in class_scores:
            class_scores[predicted_class] = np.zeros_like(scores)
        class_scores[predicted_class] += scores

    return {
        predicted_class: np.argsort(-scores)[:top_k].astype(int).tolist()
        for predicted_class, scores in class_scores.items()
    }


def rank_top_minirocket_features_by_predicted_class(
    classifier: Any,
    explainer: Any,
    top_k: int = 10,
    classifier_explainer: str | Callable = "shap",
    reference_policy: str = "global_centroid",
    dataset_name: str | None = None,
    score_mode: str = "positive",
    sample_size: int | None = None,
    random_state: int | None = 0,
) -> dict[Any, list[int]]:
    """
    Compute alpha attributions over a balanced sample and rank top-k features.
    """
    alpha_attributions = compute_alpha_attributions_for_balanced_sample(
        classifier,
        explainer,
        classifier_explainer=classifier_explainer,
        reference_policy=reference_policy,
        dataset_name=dataset_name,
        sample_size=sample_size,
        random_state=random_state,
    )
    return rank_top_minirocket_features_by_predicted_class_from_alphas(
        alpha_attributions,
        top_k=top_k,
        score_mode=score_mode,
    )


def backpropagate_selected_features_positive_segments(
    classifier: Any,
    selected_feature_indexes_by_class: dict[Any, list[int]],
    alpha_attributions: list[dict[str, Any]],
    threshold: float = 0.0,
    n_jobs: int = 1,
) -> dict[Hashable, list[dict[str, Any]]]:
    """
    Backpropagate selected MiniROCKET features from precomputed alpha records.

    Use alpha_attributions returned by
    compute_alpha_attributions_for_balanced_sample so the selected features are
    backpropagated over the same observations used to compute the rankings.
    """
    selected_by_class = {
        _as_hashable_label(predicted_class): np.asarray(feature_indexes, dtype=int).reshape(-1)
        for predicted_class, feature_indexes in selected_feature_indexes_by_class.items()
    }
    positive_segments_by_class: dict[Hashable, list[dict[str, Any]]] = {
        predicted_class: [] for predicted_class in selected_by_class
    }

    for alpha_record in alpha_attributions:
        predicted_class = _as_hashable_label(alpha_record["predicted_class"])
        selected_features = selected_by_class.get(predicted_class)
        if selected_features is None or selected_features.size == 0:
            continue

        x_target = alpha_record["instance"]
        reference = alpha_record["reference"]
        out_x = alpha_record["out_x"]
        alphas = np.asarray(alpha_record["alphas"], dtype=float).reshape(-1)
        selected_features = selected_features[
            (0 <= selected_features) & (selected_features < alphas.shape[0])
        ]
        if selected_features.size == 0:
            continue

        selected_alphas = np.zeros_like(alphas, dtype=np.float64)
        selected_alphas[selected_features] = alphas[selected_features]
        beta = mmv.back_propagate_attribution_2(
            selected_alphas,
            out_x["traces"],
            x_target,
            reference,
            per_channel=True,
            params=classifier.minirocket_params,
            n_jobs=n_jobs,
        )
        segments = _positive_attribution_segments(x_target, beta, threshold)
        if not segments:
            continue
        positive_segments_by_class.setdefault(predicted_class, []).append(
            {
                "sample_index": int(alpha_record["sample_index"]),
                "selected_features": selected_features.astype(int).tolist(),
                "segments": segments,
            }
        )

    return positive_segments_by_class


def _dtw_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("inf")

    previous = np.full(y.shape[0] + 1, np.inf, dtype=np.float64)
    current = np.full(y.shape[0] + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for x_value in x:
        current[0] = np.inf
        for j, y_value in enumerate(y, start=1):
            local_cost = (x_value - y_value) ** 2
            current[j] = local_cost + min(
                previous[j],
                current[j - 1],
                previous[j - 1],
            )
        previous, current = current, previous
    return float(np.sqrt(previous[-1]))


def _pairwise_dtw_distances(series: list[np.ndarray]) -> np.ndarray:
    distance_matrix = np.zeros((len(series), len(series)), dtype=np.float64)
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            distance = _dtw_distance_1d(series[i], series[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def _k_medoids_from_distance_matrix(
    distance_matrix: np.ndarray,
    n_clusters: int,
    random_state: int | None,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = distance_matrix.shape[0]
    effective_n_clusters = min(n_clusters, n_samples)
    if effective_n_clusters == n_samples:
        return np.arange(n_samples, dtype=int), np.arange(n_samples, dtype=int)

    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n_samples, size=effective_n_clusters, replace=False)

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        updated_medoids = medoids.copy()

        for cluster_id in range(effective_n_clusters):
            members = np.flatnonzero(labels == cluster_id)
            if members.size == 0:
                nearest_medoid_distance = np.min(distance_matrix[:, medoids], axis=1)
                non_medoids = np.setdiff1d(np.arange(n_samples), updated_medoids)
                if non_medoids.size == 0:
                    continue
                updated_medoids[cluster_id] = non_medoids[
                    np.argmax(nearest_medoid_distance[non_medoids])
                ]
                continue

            within_cluster_distances = distance_matrix[np.ix_(members, members)]
            updated_medoids[cluster_id] = members[
                np.argmin(within_cluster_distances.sum(axis=1))
            ]

        if np.array_equal(medoids, updated_medoids):
            break
        medoids = updated_medoids

    labels = np.argmin(distance_matrix[:, medoids], axis=1)
    return medoids.astype(int), labels.astype(int)


def cluster_positive_segments_by_class_with_dtw_kmeans(
    segments_by_class: dict[Any, list[dict[str, Any]]],
    min_length: int = 5,
    n_clusters: int = 3,
    random_state: int | None = 0,
    max_iter: int = 100,
) -> dict[Hashable, dict[str, Any]]:
    """
    Z-normalize and cluster positive-attribution segments per class with DTW.

    Parameters
    ----------
    segments_by_class:
        Output of ``backpropagate_selected_features_positive_segments``. The
        expected shape is ``{class_label: observation_entries}``, where each
        observation entry contains a ``"segments"`` list. Each segment must have
        at least ``"channel"``, ``"start"``, ``"end"``, and ``"values"``.

    min_length:
        Minimum number of time points required for a segment to be clustered.
        Segments shorter than this threshold are discarded. The default is 5,
        which avoids clustering tiny runs that are usually too short to form a
        meaningful time-series motif.

    n_clusters:
        Requested number of clusters per class. If a class has fewer retained
        segments than ``n_clusters``, the effective number of clusters for that
        class is reduced to the number of retained segments.

    random_state:
        Random seed used to initialize medoids.

    max_iter:
        Maximum number of k-medoids update iterations.

    Returns
    -------
    dict
        A dictionary keyed by class label. For each class, the value contains:

        ``"normalization"``
            Per-channel z-normalization statistics used for that class, stored
            as ``{channel: {"mean": float, "std": float}}``.

        ``"dtw_distance_matrix"``
            Pairwise DTW distances between retained normalized segments.

        ``"cluster_medoids"``
            Mapping ``cluster_id -> segment_index``. A medoid is a real segment,
            which can be used directly as a time-domain shapelet.

        ``"segments"``
            One entry per retained segment with original metadata, normalized
            values, and cluster label.

        ``"clusters"``
            Mapping ``cluster_id -> segment_indexes`` into the returned
            ``"segments"`` list.

    Design decisions
    ----------------
    - Z-normalization is done independently per predicted class and per channel,
      using all retained segment values available for that class/channel. The
      function only receives extracted segments, not the full original training
      observations, so these are "global" statistics over the provided segment
      observations.
    - Clustering is done per class across all channels. Channel identity is kept
      in each segment record, while channel-scale differences are handled by the
      per-class/per-channel z-normalization.
    - DTW works with variable-length segments, so no resampling or padding is
      applied to the raw segments.
    - Standard arithmetic k-means is not well-defined for variable-length series
      under DTW without a barycenter algorithm. To keep the implementation local
      and dependency-light, this function uses k-medoids with DTW distances. The
      medoid is the real segment with the smallest total DTW distance to the
      other members of its cluster.
    - The local DTW cost is squared Euclidean distance between scalar time
      points, and the final DTW distance is the square root of the accumulated
      optimal warping cost.
    """
    if min_length <= 0:
        raise ValueError("min_length must be positive.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    clustered_by_class = {}
    for predicted_class, observation_entries in segments_by_class.items():
        predicted_class = _as_hashable_label(predicted_class)
        flat_segments = []
        values_by_channel: dict[int, list[np.ndarray]] = {}

        for observation_entry in observation_entries:
            for segment in observation_entry.get("segments", []):
                values = np.asarray(segment["values"], dtype=np.float64).reshape(-1)
                if values.shape[0] < min_length:
                    continue

                channel = int(segment["channel"])
                record = {
                    "sample_index": int(observation_entry["sample_index"]),
                    "selected_features": list(observation_entry.get("selected_features", [])),
                    "channel": channel,
                    "start": int(segment["start"]),
                    "end": int(segment["end"]),
                    "values": values.copy(),
                    "attributions": np.asarray(segment["attributions"], dtype=np.float64).copy(),
                }
                flat_segments.append(record)
                values_by_channel.setdefault(channel, []).append(values)

        if not flat_segments:
            clustered_by_class[predicted_class] = {
                "normalization": {},
                "dtw_distance_matrix": np.empty((0, 0), dtype=np.float64),
                "cluster_medoids": {},
                "segments": [],
                "clusters": {},
            }
            continue

        normalization = {}
        for channel, channel_values in values_by_channel.items():
            pooled_values = np.concatenate(channel_values)
            mean = float(np.mean(pooled_values))
            std = float(np.std(pooled_values))
            if std == 0.0:
                std = 1.0
            normalization[channel] = {"mean": mean, "std": std}

        enriched_segments = []
        for segment in flat_segments:
            stats = normalization[segment["channel"]]
            normalized_values = (segment["values"] - stats["mean"]) / stats["std"]
            enriched_segments.append(
                {
                    **segment,
                    "normalized_values": normalized_values,
                }
            )

        distance_matrix = _pairwise_dtw_distances(
            [segment["normalized_values"] for segment in enriched_segments]
        )
        medoids, labels = _k_medoids_from_distance_matrix(
            distance_matrix,
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
        )

        clusters = {cluster_id: [] for cluster_id in range(medoids.shape[0])}
        for segment_index, (segment, label) in enumerate(zip(enriched_segments, labels)):
            label = int(label)
            segment["cluster"] = label
            clusters[label].append(segment_index)

        clustered_by_class[predicted_class] = {
            "normalization": normalization,
            "dtw_distance_matrix": distance_matrix,
            "cluster_medoids": {
                int(cluster_id): int(medoid_index)
                for cluster_id, medoid_index in enumerate(medoids)
            },
            "segments": enriched_segments,
            "clusters": clusters,
        }

    return clustered_by_class


def _as_timeseries_batch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X.reshape(1, 1, -1)
    if X.ndim == 2:
        return X.reshape(1, *X.shape)
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected X with shape (L,), (C, L), or (n, C, L); got {X.shape}.")


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


def build_shapelet_detector_from_cluster_centers(
    clustered_segments_by_class: dict[Any, dict[str, Any]],
    threshold: float | None = None,
    threshold_scale: float = 0.1,
) -> Callable[[np.ndarray], list[dict[str, Any]]]:
    """
    Build a Euclidean shapelet detector from clustered positive segments.

    ``cluster_positive_segments_by_class_with_dwt_kmeans`` now clusters segments
    with DTW k-medoids. Each cluster representative is therefore already a real
    retained time-domain segment. This function uses those medoids as shapelets.

    The returned detector accepts a single time series with shape ``(L,)`` or
    ``(C, L)``, or a batch with shape ``(n, C, L)``. For each shapelet, it scans
    the matching channel with a sliding window and returns:

    - ``count``: number of windows whose Euclidean distance to the shapelet is
      at most the threshold.
    - ``locations``: all matching ``start``/``end`` windows, with distance and
      threshold.

    If ``threshold`` is ``None``, each match uses
    ``threshold_scale * std(target_channel)``. The default therefore matches
    windows within ``0.1 * std(channel)``.
    """
    if threshold is not None and threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if threshold_scale < 0.0:
        raise ValueError("threshold_scale must be non-negative.")

    shapelets = []
    for predicted_class, class_clusters in clustered_segments_by_class.items():
        predicted_class = _as_hashable_label(predicted_class)
        segments = class_clusters.get("segments", [])
        cluster_medoids = class_clusters.get("cluster_medoids", {})
        if not cluster_medoids or not segments:
            continue

        for cluster_id, medoid_index in cluster_medoids.items():
            medoid_index = int(medoid_index)
            if medoid_index < 0 or medoid_index >= len(segments):
                continue

            best_segment = segments[medoid_index]
            shapelets.append(
                {
                    "class": predicted_class,
                    "cluster": int(cluster_id),
                    "channel": int(best_segment["channel"]),
                    "values": np.asarray(best_segment["values"], dtype=np.float64).copy(),
                    "source_segment_index": int(medoid_index),
                    "source_sample_index": int(best_segment["sample_index"]),
                    "source_start": int(best_segment["start"]),
                    "source_end": int(best_segment["end"]),
                }
            )

    def detector(X: np.ndarray) -> list[dict[str, Any]]:
        X_batch = _as_timeseries_batch(X)
        detection_results = []
        for instance_index, instance in enumerate(X_batch):
            instance_matches = []
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
                instance_matches.append(
                    {
                        "class": shapelet["class"],
                        "cluster": shapelet["cluster"],
                        "shapelet_index": int(shapelet_index),
                        "channel": channel,
                        "count": len(locations),
                        "locations": locations,
                    }
                )

            detection_results.append(
                {
                    "instance_index": int(instance_index),
                    "matches": instance_matches,
                }
            )
        return detection_results

    detector.shapelets = shapelets
    return detector


def _summarize_detector_results(
    detector_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for instance_result in detector_results:
        matched_shapelets = [
            {
                "class": match["class"],
                "cluster": match["cluster"],
                "shapelet_index": match["shapelet_index"],
                "channel": match["channel"],
                "count": match["count"],
                "locations": match["locations"],
            }
            for match in instance_result["matches"]
            if match["count"] > 0
        ]
        summary.append(
            {
                "instance_index": instance_result["instance_index"],
                "matched_shapelets": matched_shapelets,
                "total_matches": int(sum(match["count"] for match in matched_shapelets)),
            }
        )
    return summary


def detector_results_to_shapelet_count_matrix(
    detector_results: list[dict[str, Any]],
    n_shapelets: int | None = None,
) -> np.ndarray:
    """
    Convert detector output into a tabular shapelet-count representation.

    Each row represents one time series. Each column represents one shapelet.
    Cell ``X[i, j]`` is the number of times shapelet ``j`` matched time series
    ``i`` according to the detector's threshold rule.

    Parameters
    ----------
    detector_results:
        Output returned by the callable produced by
        ``build_shapelet_detector_from_cluster_centers``.

    n_shapelets:
        Optional total number of shapelets. If omitted, it is inferred from the
        maximum ``shapelet_index`` present in the detector results. Passing it is
        safer when some shapelets have zero matches in every time series, because
        those zero-count columns are still retained.

    Returns
    -------
    np.ndarray
        Matrix with shape ``(n_time_series, n_shapelets)`` and integer counts.
    """
    if n_shapelets is None:
        max_shapelet_index = -1
        for instance_result in detector_results:
            for match in instance_result.get("matches", []):
                max_shapelet_index = max(max_shapelet_index, int(match["shapelet_index"]))
        n_shapelets = max_shapelet_index + 1

    if n_shapelets < 0:
        raise ValueError("n_shapelets must be non-negative.")

    count_matrix = np.zeros((len(detector_results), n_shapelets), dtype=np.int64)
    for row_index, instance_result in enumerate(detector_results):
        for match in instance_result.get("matches", []):
            shapelet_index = int(match["shapelet_index"])
            if 0 <= shapelet_index < n_shapelets:
                count_matrix[row_index, shapelet_index] = int(match["count"])
    return count_matrix


def train_decision_tree_on_shapelet_counts(
    train_detector_results: list[dict[str, Any]],
    y_train: np.ndarray,
    test_detector_results: list[dict[str, Any]],
    y_test: np.ndarray,
    n_shapelets: int | None = None,
    random_state: int | None = 0,
    **decision_tree_kwargs,
) -> dict[str, Any]:
    """
    Train and evaluate a decision tree over shapelet-count features.

    The detector results are converted to a tabular representation where column
    ``j`` is the count of matches for shapelet ``j``. The decision tree is fit
    on the training count matrix and evaluated on the test count matrix.
    """
    X_train_counts = detector_results_to_shapelet_count_matrix(
        train_detector_results,
        n_shapelets=n_shapelets,
    )
    X_test_counts = detector_results_to_shapelet_count_matrix(
        test_detector_results,
        n_shapelets=X_train_counts.shape[1],
    )
    if X_train_counts.shape[1] == 0:
        raise ValueError("Cannot train a decision tree because no shapelets were detected.")
    if X_train_counts.shape[0] != len(y_train):
        raise ValueError(
            f"Training count rows ({X_train_counts.shape[0]}) do not match labels ({len(y_train)})."
        )
    if X_test_counts.shape[0] != len(y_test):
        raise ValueError(
            f"Test count rows ({X_test_counts.shape[0]}) do not match labels ({len(y_test)})."
        )

    tree = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=decision_tree_kwargs.pop("max_depth", 3),
        **decision_tree_kwargs,
    )
    tree.fit(X_train_counts, y_train)
    y_pred = tree.predict(X_test_counts)

    return {
        "classifier": tree,
        "X_train_counts": X_train_counts,
        "X_test_counts": X_test_counts,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def _train_validation_indexes(
    y_train: np.ndarray,
    validation_size: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    indexes = np.arange(len(y_train))
    classes, counts = np.unique(y_train, return_counts=True)
    stratify = y_train if classes.shape[0] > 1 and counts.min() >= 2 else None
    train_indexes, validation_indexes = train_test_split(
        indexes,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )
    return np.asarray(train_indexes, dtype=int), np.asarray(validation_indexes, dtype=int)


def save_decision_tree_classifier(
    decision_tree: DecisionTreeClassifier,
    pickle_path: str | Path,
    png_path: str | Path,
    shapelets: list[dict[str, Any]] | None = None,
    detector_threshold: float | None = None,
    detector_threshold_scale: float = 0.1,
    metadata: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (24.0, 12.0),
    dpi: int = 180,
) -> None:
    """
    Persist a learned decision tree as both a pickle file and a PNG image.

    Parameters
    ----------
    decision_tree:
        Fitted ``DecisionTreeClassifier`` to save.

    pickle_path:
        Output path for the pickled sklearn estimator.

    png_path:
        Output path for a static visualization of the tree.

    shapelets:
        Optional list of shapelet metadata from the detector. When provided, the
        pickle stores a bundle with both the classifier and the shapelets so
        downstream explanation scripts can map tree features back to shapelets.

    detector_threshold:
        Absolute detector threshold used to create the shapelet-count features,
        or ``None`` when the detector used a channel-standard-deviation-scaled
        threshold.

    detector_threshold_scale:
        Scale used by the detector when ``detector_threshold`` is ``None``.

    metadata:
        Optional extra metadata stored in the pickle bundle.

    feature_names:
        Optional names for shapelet-count features. If omitted, sklearn will
        display generic feature names.

    class_names:
        Optional class names for the tree leaves. If omitted, class labels are
        inferred from ``decision_tree.classes_``.

    figsize:
        Matplotlib figure size used for the PNG.

    dpi:
        Resolution used for the PNG file.
    """
    if not hasattr(decision_tree, "tree_"):
        raise ValueError("decision_tree must be fitted before it can be saved.")

    pickle_path = Path(pickle_path)
    png_path = Path(png_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    pickle_payload = {
        "classifier": decision_tree,
        "shapelets": shapelets,
        "detector_threshold": detector_threshold,
        "detector_threshold_scale": detector_threshold_scale,
        "feature_names": feature_names,
        "metadata": metadata or {},
    }
    with pickle_path.open("wb") as output_file:
        pickle.dump(pickle_payload, output_file, pickle.HIGHEST_PROTOCOL)

    if class_names is None:
        class_names = [str(class_label) for class_label in decision_tree.classes_]

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    total_start = time.perf_counter()
    run_started_at = datetime.now(timezone.utc).isoformat()
    runtimes = {}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-name",
        default=None,
        help=(
            "Classifier pickle basename to load from experiments/data/<dataset-name>. "
            "The .pkl suffix is optional. Default: RandomForestClassifier when "
            "--classifier-path is omitted."
        ),
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit classifier pickle path. If omitted, the script loads "
            "experiments/data/<dataset-name>/<classifier-name>.pkl."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=_parse_int_grid,
        default=[5, 10],
        help="Comma-separated top-k values for the feature-ranking grid. Default: 5,10.",
    )
    parser.add_argument("--classifier-explainer", default="shap")
    parser.add_argument("--reference-policy", default="global_centroid")
    parser.add_argument("--dataset-name", default="double-freq-test")
    parser.add_argument("--run_id", type=int, default=None)
    parser.add_argument("--job_id", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Optional total sample size, balanced across predicted classes.",
    )
    parser.add_argument(
        "--random-state",
        type=_parse_int_grid,
        default=[0],
        help=(
            "Comma-separated random_state values for the optimization grid. "
            "The first value is also used for alpha sampling and the validation split. Default: 0."
        ),
    )
    parser.add_argument(
        "--n-clusters",
        type=_parse_int_grid,
        default=[10, 20],
        help="Comma-separated cluster counts for the grid. Default: 10,20.",
    )
    parser.add_argument(
        "--threshold-scale",
        type=_parse_float_grid,
        default=[0.05, 0.1],
        help="Comma-separated detector threshold scales for the grid. Default: 0.05,0.1.",
    )
    parser.add_argument(
        "--max-depth",
        type=_parse_optional_int_grid,
        default=[2, 3, 4, 5],
        help=(
            "Comma-separated decision-tree max_depth values for the grid. "
            "Use None for unlimited depth. Default: 2,3,4,5."
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the training set used to select grid-search hyperparameters. Default: 0.2.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of worker threads for alpha attribution. Use -1 for all CPU cores.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("positive", "absolute", "raw"),
        default="positive",
    )
    parser.add_argument(
        "--tree-pickle-path",
        type=Path,
        default=None,
        help=(
            "Output pickle path for the shapelet decision tree. Default: "
            "experiments/data/<dataset-name>/shapelet_decision_tree.pkl."
        ),
    )
    parser.add_argument(
        "--tree-png-path",
        type=Path,
        default=None,
        help=(
            "Output PNG path for the shapelet decision tree. Default: "
            "experiments/data/<dataset-name>/shapelet_decision_tree.png."
        ),
    )
    parser.add_argument(
        "--run-summary-tsv-path",
        type=Path,
        default=None,
        help=(
            "Append one TSV row with execution runtimes, accuracies, selected "
            "hyperparameters, and shapelet counts. Default: "
            "experiments/data/<dataset-name>/whitebox_classifier_runs.tsv."
        ),
    )
    args = parser.parse_args()
    if args.tree_pickle_path is None:
        args.tree_pickle_path = _default_dataset_output_path(
            args.dataset_name,
            "shapelet_decision_tree.pkl",
        )
    if args.tree_png_path is None:
        args.tree_png_path = _default_dataset_output_path(
            args.dataset_name,
            "shapelet_decision_tree.png",
        )
    if args.run_summary_tsv_path is None:
        args.run_summary_tsv_path = _default_dataset_output_path(
            args.dataset_name,
            "whitebox_classifier_runs.tsv",
        )
    if args.classifier_path is None:
        if args.classifier_name is None:
            args.classifier_name = "RandomForestClassifier"
        args.classifier_path = _default_classifier_path(
            args.dataset_name,
            args.classifier_name,
        )
    elif args.classifier_name is None:
        args.classifier_name = args.classifier_path.stem
    random_state_grid = _dedupe_preserving_order(args.random_state)
    sample_random_state = random_state_grid[0]

    start = time.perf_counter()
    with args.classifier_path.open("rb") as input_file:
        classifier = pickle.load(input_file)
    _record_runtime(runtimes, "load_classifier", start)

    start = time.perf_counter()
    alpha_attributions = compute_alpha_attributions_for_balanced_sample(
        classifier=classifier,
        explainer=classifier.get_explainer(),
        classifier_explainer=args.classifier_explainer,
        reference_policy=args.reference_policy,
        dataset_name=args.dataset_name,
        sample_size=args.sample_size,
        random_state=sample_random_state,
        n_jobs=args.n_jobs,
    )
    _record_runtime(runtimes, "compute_alpha_attributions", start)

    print(f"classifier_name={args.classifier_name}")
    print(f"classifier_path={args.classifier_path}")
    print(f"sample_size={args.sample_size}")
    print(f"random_state_grid={random_state_grid}")
    print(f"sample_random_state={sample_random_state}")
    print(f"n_jobs={_resolve_n_jobs(args.n_jobs)}")
    top_k_grid = _dedupe_preserving_order(args.top_k)
    n_clusters_grid = _dedupe_preserving_order(args.n_clusters)
    threshold_scale_grid = _dedupe_preserving_order(args.threshold_scale)
    max_depth_grid = _dedupe_preserving_order(args.max_depth)
    print(f"top_k_grid={top_k_grid}")
    print(f"n_clusters_grid={n_clusters_grid}")
    print(f"threshold_scale_grid={threshold_scale_grid}")
    print(f"max_depth_grid={max_depth_grid}")
    print(f"validation_size={args.validation_size}")
    print(f"computed_alpha_attributions={len(alpha_attributions)}")

    start = time.perf_counter()
    X_train = _training_data(classifier)
    y_train = _training_labels(classifier)
    (_, _), (X_test, y_test) = _load_dataset_for_classifier(classifier, args.dataset_name)
    minirocket_train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))
    minirocket_test_accuracy = float(accuracy_score(y_test, classifier.predict(X_test)))
    train_indexes, validation_indexes = _train_validation_indexes(
        y_train,
        validation_size=args.validation_size,
        random_state=sample_random_state,
    )
    _record_runtime(runtimes, "load_data_and_split_validation", start)
    print(f"minirocket_train_accuracy={minirocket_train_accuracy:.6f}")
    print(f"minirocket_test_accuracy={minirocket_test_accuracy:.6f}")

    best_candidate = None
    grid_results = []
    grid_start = time.perf_counter()

    for top_k in top_k_grid:
        start = time.perf_counter()
        rankings = rank_top_minirocket_features_by_predicted_class_from_alphas(
            alpha_attributions,
            top_k=top_k,
            score_mode=args.score_mode,
        )
        _record_runtime(runtimes, f"rank_top_features_top_k_{top_k}", start)

        for predicted_class, feature_indexes in rankings.items():
            print(f"top_k={top_k} class {predicted_class}: {feature_indexes}", flush=True)

        start = time.perf_counter()
        positive_segments_by_class = backpropagate_selected_features_positive_segments(
            classifier=classifier,
            alpha_attributions=alpha_attributions,
            selected_feature_indexes_by_class=rankings,
            n_jobs=_resolve_n_jobs(args.n_jobs),
        )
        _record_runtime(runtimes, f"backpropagate_selected_features_top_k_{top_k}", start)

        for predicted_class, segments in positive_segments_by_class.items():
            print(
                f"top_k={top_k} class {predicted_class}: "
                f"positive_segment_observations={len(segments)}",
                flush=True,
            )

        for n_clusters in n_clusters_grid:
            for candidate_random_state in random_state_grid:
                start = time.perf_counter()
                clusters_per_class = cluster_positive_segments_by_class_with_dtw_kmeans(
                    positive_segments_by_class,
                    n_clusters=n_clusters,
                    random_state=candidate_random_state,
                )
                _record_runtime(
                    runtimes,
                    (
                        f"cluster_positive_segments_top_k_{top_k}_n_clusters_{n_clusters}"
                        f"_random_state_{candidate_random_state}"
                    ),
                    start,
                )

                for predicted_class, clusters in clusters_per_class.items():
                    print(
                        f"top_k={top_k} n_clusters={n_clusters} "
                        f"random_state={candidate_random_state} class {predicted_class}: "
                        f"clustered_segments={len(clusters['segments'])} "
                        f"effective_clusters={len(clusters['clusters'])}",
                        flush=True,
                    )

                for threshold_scale in threshold_scale_grid:
                    start = time.perf_counter()
                    detector = build_shapelet_detector_from_cluster_centers(
                        clusters_per_class,
                        threshold_scale=threshold_scale,
                    )
                    detector_results = detector(X_train)
                    test_detector_results = detector(X_test)
                    X_train_counts = detector_results_to_shapelet_count_matrix(
                        detector_results,
                        n_shapelets=len(detector.shapelets),
                    )
                    X_test_counts = detector_results_to_shapelet_count_matrix(
                        test_detector_results,
                        n_shapelets=X_train_counts.shape[1],
                    )
                    _record_runtime(
                        runtimes,
                        (
                            f"detect_shapelets_top_k_{top_k}_n_clusters_{n_clusters}"
                            f"_random_state_{candidate_random_state}"
                            f"_threshold_scale_{threshold_scale}"
                        ),
                        start,
                    )

                    for max_depth in max_depth_grid:
                        start = time.perf_counter()
                        tree = DecisionTreeClassifier(
                            random_state=candidate_random_state,
                            max_depth=max_depth,
                        )
                        tree.fit(X_train_counts[train_indexes], y_train[train_indexes])
                        validation_pred = tree.predict(X_train_counts[validation_indexes])
                        test_pred = tree.predict(X_test_counts)
                        validation_accuracy = float(
                            accuracy_score(y_train[validation_indexes], validation_pred)
                        )
                        test_accuracy = float(accuracy_score(y_test, test_pred))
                        _record_runtime(
                            runtimes,
                            (
                                f"train_tree_top_k_{top_k}_n_clusters_{n_clusters}"
                                f"_random_state_{candidate_random_state}"
                                f"_threshold_scale_{threshold_scale}_max_depth_{max_depth}"
                            ),
                            start,
                        )

                        candidate = {
                            "top_k": int(top_k),
                            "n_clusters": int(n_clusters),
                            "random_state": int(candidate_random_state),
                            "threshold_scale": float(threshold_scale),
                            "max_depth": None if max_depth is None else int(max_depth),
                            "validation_accuracy": validation_accuracy,
                            "test_accuracy": test_accuracy,
                            "n_shapelets": len(detector.shapelets),
                            "detector": detector,
                            "X_train_counts": X_train_counts,
                            "X_test_counts": X_test_counts,
                        }
                        grid_results.append(candidate)
                        print(
                            "grid_candidate "
                            f"top_k={top_k} n_clusters={n_clusters} "
                            f"random_state={candidate_random_state} "
                            f"threshold_scale={threshold_scale} max_depth={max_depth} "
                            f"validation_accuracy={validation_accuracy:.6f} "
                            f"test_accuracy={test_accuracy:.6f} "
                            f"shapelets={len(detector.shapelets)}",
                            flush=True,
                        )

                        candidate_key = (
                            validation_accuracy,
                            test_accuracy,
                            -10**9 if max_depth is None else -int(max_depth),
                            -int(top_k),
                            -int(n_clusters),
                            -float(threshold_scale),
                            -int(candidate_random_state),
                        )
                        if best_candidate is None or candidate_key > best_candidate["selection_key"]:
                            candidate["selection_key"] = candidate_key
                            best_candidate = candidate

    _record_runtime(runtimes, "grid_search_total", grid_start)

    if best_candidate is None:
        raise ValueError("Grid search did not produce any candidate decision tree.")

    print(
        "best_grid_candidate "
        f"top_k={best_candidate['top_k']} "
        f"n_clusters={best_candidate['n_clusters']} "
        f"random_state={best_candidate['random_state']} "
        f"threshold_scale={best_candidate['threshold_scale']} "
        f"max_depth={best_candidate['max_depth']} "
        f"validation_accuracy={best_candidate['validation_accuracy']:.6f} "
        f"test_accuracy_before_refit={best_candidate['test_accuracy']:.6f}",
        flush=True,
    )

    start = time.perf_counter()
    final_tree = DecisionTreeClassifier(
        random_state=best_candidate["random_state"],
        max_depth=best_candidate["max_depth"],
    )
    final_tree.fit(best_candidate["X_train_counts"], y_train)
    final_test_pred = final_tree.predict(best_candidate["X_test_counts"])
    final_test_accuracy = float(accuracy_score(y_test, final_test_pred))
    shapelet_tree_result = {
        "classifier": final_tree,
        "X_train_counts": best_candidate["X_train_counts"],
        "X_test_counts": best_candidate["X_test_counts"],
        "y_pred": final_test_pred,
        "accuracy": final_test_accuracy,
    }
    detector = best_candidate["detector"]
    _record_runtime(runtimes, "refit_best_decision_tree", start)

    print(f"shapelet_count_train_shape={shapelet_tree_result['X_train_counts'].shape}")
    print(f"shapelet_count_test_shape={shapelet_tree_result['X_test_counts'].shape}")
    print(f"shapelet_decision_tree_accuracy={shapelet_tree_result['accuracy']:.6f}")
    print(f"shapelets={len(detector.shapelets)}")

    start = time.perf_counter()
    save_decision_tree_classifier(
        decision_tree=shapelet_tree_result["classifier"],
        pickle_path=args.tree_pickle_path,
        png_path=args.tree_png_path,
        shapelets=detector.shapelets,
        detector_threshold=None,
        detector_threshold_scale=best_candidate["threshold_scale"],
        metadata={
            "dataset_name": args.dataset_name,
            "dataset_n_samples": len(X_train),
            "grid_search": {
                "top_k_values": top_k_grid,
                "n_clusters_values": n_clusters_grid,
                "random_state_values": random_state_grid,
                "sample_random_state": sample_random_state,
                "validation_split_random_state": sample_random_state,
                "threshold_scale_values": threshold_scale_grid,
                "max_depth_values": max_depth_grid,
                "validation_size": args.validation_size,
                "selection_metric": "validation_accuracy",
                "results": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        in {
                            "top_k",
                            "n_clusters",
                            "random_state",
                            "threshold_scale",
                            "max_depth",
                            "validation_accuracy",
                            "test_accuracy",
                            "n_shapelets",
                        }
                    }
                    for candidate in grid_results
                ],
                "best": {
                    "top_k": best_candidate["top_k"],
                    "n_clusters": best_candidate["n_clusters"],
                    "random_state": best_candidate["random_state"],
                    "threshold_scale": best_candidate["threshold_scale"],
                    "max_depth": best_candidate["max_depth"],
                    "validation_accuracy": best_candidate["validation_accuracy"],
                    "test_accuracy_before_refit": best_candidate["test_accuracy"],
                    "test_accuracy_after_refit": final_test_accuracy,
                    "n_shapelets": best_candidate["n_shapelets"],
                },
            },
        },
        feature_names=[
            f"shapelet_{idx}" for idx in range(shapelet_tree_result["X_train_counts"].shape[1])
        ],
    )
    _record_runtime(runtimes, "save_decision_tree", start)

    print(f"shapelet_decision_tree_pickle={args.tree_pickle_path}")
    print(f"shapelet_decision_tree_png={args.tree_png_path}")
    _record_runtime(runtimes, "total", total_start)
    run_summary = {
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "job_id": args.job_id,
        "classifier_name": args.classifier_name,
        "classifier_path": args.classifier_path,
        "tree_pickle_path": args.tree_pickle_path,
        "tree_png_path": args.tree_png_path,
        "dataset_name": args.dataset_name,
        "classifier_explainer": args.classifier_explainer,
        "reference_policy": args.reference_policy,
        "score_mode": args.score_mode,
        "sample_size": args.sample_size,
        "computed_alpha_attributions": len(alpha_attributions),
        "random_state": best_candidate["random_state"],
        "random_state_grid": random_state_grid,
        "sample_random_state": sample_random_state,
        "n_jobs_requested": args.n_jobs,
        "n_jobs_resolved": _resolve_n_jobs(args.n_jobs),
        "validation_size": args.validation_size,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "validation_train_size": len(train_indexes),
        "validation_size_count": len(validation_indexes),
        "top_k_grid": top_k_grid,
        "n_clusters_grid": n_clusters_grid,
        "threshold_scale_grid": threshold_scale_grid,
        "max_depth_grid": max_depth_grid,
        "grid_candidates": len(grid_results),
        "minirocket_train_accuracy": minirocket_train_accuracy,
        "minirocket_test_accuracy": minirocket_test_accuracy,
        "best_top_k": best_candidate["top_k"],
        "best_n_clusters": best_candidate["n_clusters"],
        "best_threshold_scale": best_candidate["threshold_scale"],
        "best_max_depth": best_candidate["max_depth"],
        "best_validation_accuracy": best_candidate["validation_accuracy"],
        "best_test_accuracy_before_refit": best_candidate["test_accuracy"],
        "shapelet_decision_tree_accuracy": shapelet_tree_result["accuracy"],
        "n_shapelets": len(detector.shapelets),
        "shapelet_count_train_rows": shapelet_tree_result["X_train_counts"].shape[0],
        "shapelet_count_train_columns": shapelet_tree_result["X_train_counts"].shape[1],
        "shapelet_count_test_rows": shapelet_tree_result["X_test_counts"].shape[0],
        "shapelet_count_test_columns": shapelet_tree_result["X_test_counts"].shape[1],
        "decision_tree_depth": final_tree.get_depth(),
        "decision_tree_leaves": final_tree.get_n_leaves(),
        "runtime_load_classifier_seconds": runtimes.get("load_classifier"),
        "runtime_compute_alpha_attributions_seconds": runtimes.get("compute_alpha_attributions"),
        "runtime_load_data_and_split_validation_seconds": runtimes.get(
            "load_data_and_split_validation"
        ),
        "runtime_grid_search_total_seconds": runtimes.get("grid_search_total"),
        "runtime_refit_best_decision_tree_seconds": runtimes.get("refit_best_decision_tree"),
        "runtime_save_decision_tree_seconds": runtimes.get("save_decision_tree"),
        "runtime_total_seconds": runtimes.get("total"),
        "runtime_breakdown_seconds": runtimes,
    }
    _write_run_summary_tsv(args.run_summary_tsv_path, run_summary)
    print(f"run_summary_tsv={args.run_summary_tsv_path}")


if __name__ == "__main__":
    main()
