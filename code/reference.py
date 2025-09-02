import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def medoid(X: pd.DataFrame) -> tuple(int, pd.Series):
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

def centroid(X: pd.DataFrame):
    return X.mean(axis=0)

def medoid_per_class(X: pd.DataFrame, y) -> dict:
    """
    Return the positional index of the medoid of each class in X.
    :param X: A time series dataset (rows = samples, columns = features)
    :param y: The class labels
    :return: A dictionary with class labels as keys and positional indices of the medoids per class
    as values
    """
    medoids = {}

    for label in np.unique(y):
        # Extract samples of this class
        X_class = X[y == label]

        # Index of the minimum sum distance → medoid
        medoid_idx = medoid(X_class)

        # Store medoid vector
        medoids[label] = medoid_idx

    return medoids

def centroid_per_class(X, y) -> dict:
    """
    Compute the centroid of each class in X.
    :param X: A time series dataset (rows = samples, columns = features)
    :param y: The class labels
    :return: A dictionary with class labels as keys and centroid vectors (pandas Series) as values
    """
    centroids = {}

    for label in np.unique(y):
        # Extract samples of this class
        X_class = X[y == label]

        # Average vector
        centroid_idx = centroid(X_class)

        # Store medoid vector
        centroids[label] = centroid_idx

    return centroids
