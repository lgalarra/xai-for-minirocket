import numpy as np
from scipy.spatial.distance import cdist

def medoid_per_class(X, y):
    X = np.array(X)  # ensure numpy array
    y = np.array(y)
    medoids = {}

    for label in np.unique(y):
        # Extract samples of this class
        X_class = X[y == label]

        # Compute distance matrix
        distances = cdist(X_class, X_class, metric='euclidean')

        # Sum distances for each point
        total_distances = distances.sum(axis=1)

        # Index of the minimum sum distance → medoid
        medoid_idx = np.argmin(total_distances)

        # Store medoid vector
        medoids[label] = X_class[medoid_idx]

    return medoids
