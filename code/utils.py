import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression




def export(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(1, N+1)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)

def export_univ_tmc(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(0, N)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)


def get_cognitive_circles_data(data_dir='data/cognitive-circles'):
    train_data_path = f'{data_dir}/df40participants.h5'
    test_data_path = f'{data_dir}/df8participants.h5'
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        return pd.read_hdf(train_data_path), pd.read_hdf(test_data_path),

    return None, None

def get_cognitive_circles_data_for_classification(data_dir='data/cognitive-circles', target_col='RealDifficulty'):
    train_data, test_data = get_cognitive_circles_data(data_dir)
    return (train_data.drop(columns=[target_col]).select_dtypes(include=np.float64).dropna(axis=1), train_data[target_col],
            test_data.drop(columns=[target_col]).select_dtypes(include=np.float64).dropna(axis=1), test_data[target_col])

def prepare_cognitive_circles_data_for_minirocket(df, y, class_dict=None):
    if class_dict is None:
        class_dict = {'facil': 0, 'dificil': 1}
    n, L = df.shape
    X = df.to_numpy(dtype=np.float64).reshape(n, 7, int(L / 7))  # (n, C=1, L)
    return X, y.map(class_dict)

def logistic_gradient(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the predicted probability of the positive class
    for a logistic regression model w.r.t. input point x.

    Parameters
    ----------
    model : LogisticRegression
        A trained sklearn LogisticRegression model.
    x : np.ndarray
        Input point, shape (n_features,).

    Returns
    -------
    gradient : np.ndarray
        Gradient vector, shape (n_features,).
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D numpy array representing a single point.")

    if not hasattr(model, "coef_"):
        raise ValueError("Model must be fitted before computing the gradient.")

    # Get weights and intercept
    w = model.coef_[0]          # shape (n_features,)
    b = model.intercept_[0]

    # Compute linear combination
    z = np.dot(w, x) + b

    # Sigmoid function
    p = 1 / (1 + np.exp(-z))

    # Gradient of p with respect to x
    grad = p * (1 - p) * w
    return grad