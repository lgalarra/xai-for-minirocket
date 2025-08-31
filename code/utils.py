import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def medoid(X: pd.DataFrame):
    """
    Return the index label of the medoid of a pandas DataFrame X (rows = samples).
    """
    X_array = X.to_numpy()
    distances = cdist(X_array, X_array, metric='euclidean')
    total_distances = distances.sum(axis=1)
    medoid_position = total_distances.argmin()
    return X.index[medoid_position]

def medoid_per_class(X, y):
    medoids = {}

    for label in np.unique(y):
        # Extract samples of this class
        X_class = X[y == label]

        # Index of the minimum sum distance → medoid
        medoid_idx = medoid(X_class)

        # Store medoid vector
        medoids[label] = medoid_idx

    return medoids

def export(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(1, N+1)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)

def export_univ_tmc(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(0, N)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)