import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def medoid_time_series_idx(X: np.ndarray) -> int:
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
    N, C, L = X.shape
    X_flat = X.reshape(N, C * L)

    # Pairwise distances
    distances = cdist(X_flat, X_flat, metric='euclidean')

    # Sum of distances for each time series
    total_distances = distances.sum(axis=1)

    # Index of the medoid
    medoid_idx = np.argmin(total_distances)

    return medoid_idx

def medoid_data_frame(X: pd.DataFrame) -> int:
    """
    Return the positional index of the medoid of a pandas DataFrame X (rows = samples).
    The medoid is the sample that minimizes the sum of distances to all other samples.
    To obtain the vector corresponding to the medoid, use X.iloc[medoid_idx].
    """
    X_array = X.to_numpy()
    distances = cdist(X_array, X_array, metric='euclidean')
    total_distances = distances.sum(axis=1)
    medoid_position = total_distances.argmin()
    return medoid_position

def centroid_data_frame(X: pd.DataFrame) -> pd.Series:
    return X.mean(axis=0)

def centroid_time_series(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=0)

def medoid_ids_per_class(X: np.ndarray, y) -> dict:
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
        medoid_idx_local = medoid_time_series_idx(X_class)

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
