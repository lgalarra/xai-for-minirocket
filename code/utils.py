import os
import re
from curses.ascii import isdigit
import numdifftools as nd


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
import minirocket_multivariate_variable as mmv


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sktime.datasets import load_UCR_UEA_dataset

COGNITIVE_CIRCLES_CHANNELS = dict([('X', 'X coordinate'), ('V', 'Velocity'), ('VA', 'Angular Velocity'),
 ('DR', 'Radial Velocity'), ('Y', 'Y'), ('D', 'Radius'), ('A', 'Acceleration')])

COGNITIVE_CIRCLES_BASIC_CHANNELS = dict([('X', 'X coordinate'), ('V', 'Velocity')])


COGNITIVE_CIRCLES_UNITS = {'X': 'pixel', 'Y': 'pixel', 'V': 'pixel/s', 'VA': 'radians/s^2', 'DR': 'pixel/s^2',
                           'D': 'pixel', 'A': 'pixel/s^2'}

class ChannelScaler(BaseEstimator, TransformerMixin):
    """
    Z-normalize a time series dataset (N, C, L) per channel,
    using statistics computed on the training set.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Compute per-channel mean and std from training data.
        :param X: array of shape (N, C, L)
        """
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)  # shape (1, C, 1)
        self.std_ = X.std(axis=(0, 2), keepdims=True)    # shape (1, C, 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization using training statistics.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("This ChannelScaler instance is not fitted yet. Call 'fit' first.")
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert normalization back to the original scale.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("This ChannelScaler instance is not fitted yet. Call 'fit' first.")
        return X * self.std_ + self.mean_



def export(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(1, N+1)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)

def export_univ_tmc(s, columns, suffix, folder='data'):
    N = int(len(s.index) / len(columns))
    for (VAR, VARNAME) in columns:
        s[[VAR  + str(i) for i in range(0, N)]].to_frame().to_csv(f'{folder}/{VARNAME}/{VARNAME}_{suffix}.csv', header=False)

def univariate_series_transform(X):
    return X.to_numpy().astype(np.float32).reshape(-1, 1, X.shape[1])

def get_abnormal_hearbeat_for_classification(target_class=None):
    X, y = load_UCR_UEA_dataset(name="AbnormalHeartbeat", return_X_y=True)
    X = X.iloc[:, 0].apply(pd.Series)
    y = np.where(y != target_class, 0, 1)  # Asegura etiquetas 0/1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (univariate_series_transform(X_train), y_train), (univariate_series_transform(X_test), y_test)


def get_starlightcurves_for_classification(target_class=None):
    X, y = load_UCR_UEA_dataset(name="StarLightCurves", return_X_y=True)
    X = X.iloc[:, 0].apply(pd.Series)
    y = np.where(y != target_class, 0, 1)  # Asegura etiquetas 0/1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (univariate_series_transform(X_train), y_train), (univariate_series_transform(X_test), y_test)

def get_forda_for_classification():
    X, y = load_UCR_UEA_dataset(name="FordA", return_X_y=True)
    X = X.iloc[:, 0].apply(pd.Series)
    y = np.where(y == '-1', 0, 1)  # Asegura etiquetas 0/1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (univariate_series_transform(X_train), y_train), (univariate_series_transform(X_test), y_test)

def get_cognitive_circles_data(data_dir='data/cognitive-circles') -> (pd.DataFrame, pd.DataFrame):
    train_data_path = f'{data_dir}/df40participants.h5'
    test_data_path = f'{data_dir}/df8participants.h5'
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        return pd.read_hdf(train_data_path), pd.read_hdf(test_data_path),

    return None, None

def get_cognitive_circles_data_for_classification(data_dir='data/cognitive-circles', target_col='RealDifficulty',
                                                  as_numpy=False, normalize_numpy=True):
    """
    
    :param data_dir: directory where the data is stored
    :param target_col: column name of the target variable
    :param as_numpy: return numpy arrays instead of pandas dataframes
    :param normalize_numpy: normalize the data to have zero mean and unit variance for each channel (applicable only if as_numpy=True) 
    :return: 
    """
    (X_train, y_train), (X_test, y_test)  = _get_cognitive_circles_data_for_classification(data_dir, target_col,
                                                                                                   as_numpy=as_numpy)
    if as_numpy and normalize_numpy:
        channel_scaler = ChannelScaler()
        channel_scaler.fit(X_train)
        X_train_trans = channel_scaler.transform(X_train)
        X_test_trans = channel_scaler.transform(X_test)
        return (X_train_trans, y_train), (X_test_trans, y_test)
    else:
        return (X_train, y_train), (X_test, y_test)


def cognitive_circles_get_sorted_channels_from_df(data_dir='data/cognitive-circles') -> list:
    (train_data, _), _, = _get_cognitive_circles_data_for_classification(data_dir)
    return [re.sub(r'1$', '', col) for col in train_data.columns if col.endswith("1") and not isdigit(col[len(col) - 2])]

def _get_cognitive_circles_data_for_classification(data_dir='data/cognitive-circles', target_col='RealDifficulty',
                                                  as_numpy=False):
    train_data, test_data = get_cognitive_circles_data(data_dir)
    if as_numpy:
        (df_train, df_y_train), (df_test, df_y_test) = get_cognitive_circles_data_for_classification(data_dir, target_col, as_numpy=False)
        return prepare_cognitive_circles_data_for_minirocket(df_train, df_y_train), prepare_cognitive_circles_data_for_minirocket(df_test, df_y_test)
    else:
        return ((train_data.drop(columns=[target_col]).select_dtypes(include=np.float64).dropna(axis=1), train_data[target_col]),
                (test_data.drop(columns=[target_col]).select_dtypes(include=np.float64).dropna(axis=1), test_data[target_col]))

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

def normalize_df(df):
    return (df - df.mean()) / df.std()

def make_jacobian_fn(X, L, parameters, k):
    """
    Returns a callable that computes the full Jacobian dF/dX using numdifftools.
    F = transform_heaviside(X, L, parameters, k)

    The Jacobian has shape (F.size, X.size).
    """

    original_shape_X = X.shape  # e.g. (C, total_length)
    # Compute output shape to know how to reshape the Jacobian later
    sample_output = mmv.transform_soft_heaviside(X, L, parameters, k)
    original_shape_F = sample_output.shape  # (num_examples, num_features)

    def wrapped_transform_soft_heaviside(X_flat):
        # Reshape flattened X back into its original multidimensional form
        X_matrix = X_flat.reshape(original_shape_X)
        F = mmv.transform_soft_heaviside(X_matrix, L, parameters, k)
        return F.ravel()  # flatten output for numdifftools.Jacobian

    # Build the Jacobian calculator
    jac_fn = nd.Jacobian(
        wrapped_transform_soft_heaviside,
        method='central',  # more accurate than 'forward'
        step=1e-5
    )

    def jacobian_of_X(X_val):
        """
        Computes Jacobian at given X value.
        Returns array of shape (*F.shape, *X.shape).
        """
        J = jac_fn(X_val.ravel())
        # numdifftools returns (F.size, X.size)
        return J.reshape(*original_shape_F, *original_shape_X)

    return jacobian_of_X

def transform_soft_heaviside_gradient(X, L, parameters, k):
    # ---------------------------------------------------------------------------
    # Example usage
    # ---------------------------------------------------------------------------

    # Assume you already have your inputs defined:
    # X, L, parameters, k = ...

    jacobian_fn = make_jacobian_fn(X, L, parameters, k)
    return jacobian_fn(X)  # Shape: (num_examples, num_features, *X.shape)

