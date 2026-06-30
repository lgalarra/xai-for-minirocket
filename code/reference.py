import copy
import time

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from counterfactual import (
    _as_batch,
    _classifier_classes,
    _get_minirocket_transform,
    _get_predict_proba,
    optimize_minirocket_counterfactual,
)

COUNTERFACTUAL_REFERENCE_POLICY = "counterfactual"

REFERENCE_POLICIES = ['opposite_class_closest_instance', 'opposite_class_medoid', 'opposite_class_centroid',
                      'global_medoid', 'global_centroid', 'opposite_class_farthest_instance',
                      COUNTERFACTUAL_REFERENCE_POLICY,
                      ]

REFERENCE_POLICIES_LABELS = {'opposite_class_medoid': "Medoid of Opposite Class",
                             'opposite_class_centroid': "Centroid of Opposite Class",
                             'global_medoid': "Global Medoid", 'global_centroid': "Global Centroid",
                             'opposite_class_farthest_instance': "Farthest Instance of Opposite Class",
                             'opposite_class_closest_instance': "Closest Instance of Opposite Class",
                             COUNTERFACTUAL_REFERENCE_POLICY: "Counterfactual"
                             }

COUNTERFACTUAL_REFERENCE_DEFAULT_PARAMS = {
    "seed_reference_policy": "opposite_class_medoid",
    "target_class": None,
    "weights": {
        "dwt": 1.0,
        "frequency": 1.0,
        "minirocket": 1.0,
        "probability": 1.0,
    },
    "probability_mode": "margin",
    "wavelet_levels": None,
    "initial_blend": 1.0,
    "method": "COBYLA",
    "maxiter": 100,
    "tol": 1e-4,
    "optimizer_options": {"rhobeg": 0.25},
}

COUNTERFACTUAL_REFERENCE_PARAMS = {
    "ford-a": {"maxiter": 100, "tol": 1e-4, "initial_blend": 0.5},
    "double-freq-test": {"maxiter": 100, "tol": 1e-4, "initial_blend": 0.5},
    "abnormal-heartbeat-c1": {"maxiter": 100, "tol": 1e-4},
    "starlight-c1": {"maxiter": 100, "tol": 1e-4, "initial_blend": 0.5},
    "starlight-c2": {"maxiter": 100, "tol": 1e-4, "initial_blend": 0.5},
    "starlight-c3": {"maxiter": 100, "tol": 1e-4, "initial_blend": 0.5},
    "cognitive-circles": {"maxiter": 75, "tol": 1e-4},
    "handoutlines": {"maxiter": 50, "tol": 1e-4},
}


def get_counterfactual_reference_params(dataset_name: str) -> dict:
    params = copy.deepcopy(COUNTERFACTUAL_REFERENCE_DEFAULT_PARAMS)
    dataset_params = copy.deepcopy(COUNTERFACTUAL_REFERENCE_PARAMS.get(dataset_name, {}))
    if "weights" in dataset_params:
        params["weights"].update(dataset_params.pop("weights"))
    params.update(dataset_params)
    return params


def compute_counterfactual_reference(
        x_target,
        seed_reference,
        classifier,
        dataset_name=None,
        minirocket_parameters=None,
        transform_fn=None,
        initial_blend=None) -> np.ndarray:
    params = get_counterfactual_reference_params(dataset_name)
    params.pop("seed_reference_policy", None)
    configured_initial_blend = params.pop("initial_blend", None)
    if initial_blend is None:
        initial_blend = configured_initial_blend
    if initial_blend is not None:
        if not 0.0 <= initial_blend <= 1.0:
            raise ValueError(f"initial_blend must be between 0 and 1; got {initial_blend}.")
        params["initial"] = (
            (1.0 - initial_blend) * np.asarray(x_target, dtype=np.float64)
            + initial_blend * np.asarray(seed_reference, dtype=np.float64)
        )
    transform_minirocket = _get_minirocket_transform(classifier, minirocket_parameters, transform_fn)
    predict_proba = _get_predict_proba(classifier, transform_minirocket)
    seed_reference_proba = predict_proba(_as_batch(seed_reference))[0]
    seed_reference_class_idx = int(np.argmax(seed_reference_proba))
    classes = _classifier_classes(classifier)
    params["target_class"] = (
        classes[seed_reference_class_idx]
        if classes is not None
        else seed_reference_class_idx
    )

    start = time.perf_counter()
    counterfactual_reference, info = optimize_minirocket_counterfactual(
        classifier,
        x_target,
        seed_reference,
        minirocket_parameters=minirocket_parameters,
        transform_fn=transform_fn,
        return_info=True,
        **params,
    )
    time_elapsed = time.perf_counter() - start
    print(
        "Time elapsed (counterfactual reference): "
        f"{time_elapsed}; success={info['success']}; "
        f"target_probability={info['probability_target_X_double_prime']}"
    )
    return counterfactual_reference

def medoid_time_series_idx(X: np.ndarray, distance='euclidean') -> int:
    """
    Compute the medoid of a set of time series.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, C, L)

    Returns
    -------
    medoid_idx : int
        Index of the medoid time series
    medoid_series : np.ndarray
        The medoid time series (shape (C, L))
    """
    # Flatten each time series: (N, C, L) -> (N, C*L)
    if len(X.shape) > 2:
        N, C, L = X.shape
        X_for_distance = X.reshape(N, C * L)
    else:
        X_for_distance = X

    # Pairwise distances
    distances = cdist(X_for_distance, X_for_distance, metric=distance)

    # Sum of distances for each time series
    total_distances = distances.sum(axis=1)

    # Index of the medoid
    medoid_idx = np.argmin(total_distances)

    return medoid_idx

def medoid_data_frame(X: pd.DataFrame, distance='euclidean') -> int:
    """
    Return the positional index of the medoid of a pandas DataFrame X (rows = samples).
    The medoid is the sample that minimizes the sum of distances to all other samples.
    To obtain the vector corresponding to the medoid, use X.iloc[medoid_idx].
    """
    X_array = X.to_numpy()
    distances = cdist(X_array, X_array, metric=distance)
    total_distances = distances.sum(axis=1)
    medoid_position = total_distances.argmin()
    return medoid_position

def centroid_data_frame(X: pd.DataFrame) -> pd.Series:
    return X.mean(axis=0)


def centroid_time_series(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=0)

def medoid_ids_per_class(X: np.ndarray, y, distance='euclidean') -> dict:
    """
    Return the positional index of the medoid of each class in X.
    :param X: A time series array (n, C, L)
    :param y: The class labels
    :return: A dictionary with class labels as keys and positional indices of the medoids per class
    as values
    """
    medoids = {}

    for label in np.unique(y):
        # Indices of this class in the full dataset
        class_indices = np.where(y == label)[0]

        # Extract samples of this class
        X_class = X[class_indices]

        # Index of the minimum sum distance → medoid
        medoid_idx_local = medoid_time_series_idx(X_class, distance=distance)

        medoid_idx_global = class_indices[medoid_idx_local]

        # Store medoid vector
        medoids[label] = medoid_idx_global

    return medoids

def centroid_per_class(X: np.ndarray, y) -> dict:
    """
    Compute the centroid of each class in X.
    :param X: A time series array (n, C, L)
    :param y: The class labels, as a numpy array or as pandas Series
    :return: A dictionary with class labels as keys and centroid vectors (numpy arrays) as values
    """
    centroids = {}

    for label in np.unique(y):
        # Extract samples of this class
        X_class = X[y == label]

        # Average vector
        centroids[label] = centroid_time_series(X_class)

    return centroids


def closest_series_euclidean(x, X, distance='euclidean') -> (int, np.ndarray):
    """
    Find the closest series in X to x using Euclidean distance.

    Parameters
    ----------
    x : np.ndarray
        Shape (C, L), the query time series
    X : np.ndarray
        Shape (N, C, L), the dataset of time series

    Returns
    -------
    idx : int
        Index of closest time series
    series : np.ndarray
        The closest series (C, L)
    """
    if len(X.shape) > 2:
        N, C, L = X.shape
        X_for_distance = X.reshape(N, C * L)
        x_for_distance = x.reshape(1, C * L)
    else:
        X_for_distance = X
        x_for_distance = x


    distances = cdist(x_for_distance, X_for_distance, metric=distance).flatten()
    idx = np.argmin(distances)
    return idx, X[idx]


def farthest_series_euclidean(x, X, distance='euclidean') -> (int, np.ndarray):
    """
    Find the farthest series in X to x using Euclidean distance.
    """
    if len(X.shape) > 2:
        N, C, L = X.shape
        X_for_distance = X.reshape(N, C * L)
        x_for_distance = x.reshape(1, C * L)
    else:
        X_for_distance = X
        x_for_distance = x


    distances = cdist(x_for_distance, X_for_distance, metric=distance).flatten()
    idx = np.argmax(distances)
    return idx, X[idx]
