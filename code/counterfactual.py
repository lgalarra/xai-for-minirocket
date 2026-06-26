from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize


ArrayLike = Any
TransformFn = Callable[[np.ndarray], np.ndarray]
_MMV = None


def _load_minirocket_module() -> Any:
    global _MMV
    if _MMV is None:
        import minirocket_multivariate_variable as loaded_mmv
        _MMV = loaded_mmv
    return _MMV


def _as_2d_timeseries(X: ArrayLike, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (C, L); got {X.shape}.")
    return X


def _as_batch(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(1, *X.shape)


def _ensure_2d_features(phi: Any) -> np.ndarray:
    if isinstance(phi, dict):
        phi = phi["phi"]
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    return phi


def _haar_dwt_coefficients_1d(x: np.ndarray, levels: int | None = None) -> np.ndarray:
    approx = np.asarray(x, dtype=np.float64).ravel()
    if approx.size == 0:
        return approx.copy()

    details = []
    level = 0
    sqrt2 = np.sqrt(2.0)
    while approx.size > 1 and (levels is None or level < levels):
        if approx.size % 2 == 1:
            approx = np.pad(approx, (0, 1), mode="edge")
        even = approx[0::2]
        odd = approx[1::2]
        details.append((even - odd) / sqrt2)
        approx = (even + odd) / sqrt2
        level += 1

    return np.concatenate([approx] + details[::-1])


def haar_dwt_coefficients(X: ArrayLike, levels: int | None = None) -> np.ndarray:
    """
    Return channel-wise Haar-DWT coefficients for a time series of shape (C, L).

    Odd-length approximation vectors are padded by repeating the last value.
    """
    X = _as_2d_timeseries(X, "X")
    return np.concatenate([_haar_dwt_coefficients_1d(channel, levels) for channel in X])


def dwt_distance(X: ArrayLike, Y: ArrayLike, levels: int | None = None) -> float:
    """Euclidean distance between channel-wise Haar-DWT coefficient vectors."""
    X = _as_2d_timeseries(X, "X")
    Y = _as_2d_timeseries(Y, "Y")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape; got {X.shape} and {Y.shape}.")
    return float(np.linalg.norm(haar_dwt_coefficients(X, levels) - haar_dwt_coefficients(Y, levels)))


def frequency_magnitude_spectrum(X: ArrayLike) -> np.ndarray:
    """
    Return channel-wise real-FFT magnitudes for a time series of shape (C, L).

    The magnitudes are divided by L so spectra are comparable across lengths.
    Phase is intentionally discarded.
    """
    X = _as_2d_timeseries(X, "X")
    return (np.abs(np.fft.rfft(X, axis=-1)) / X.shape[-1]).reshape(-1)


def frequency_magnitude_distance(X: ArrayLike, Y: ArrayLike) -> float:
    """Euclidean distance between channel-wise real-FFT magnitude spectra."""
    X = _as_2d_timeseries(X, "X")
    Y = _as_2d_timeseries(Y, "Y")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape; got {X.shape} and {Y.shape}.")
    return float(np.linalg.norm(frequency_magnitude_spectrum(X) - frequency_magnitude_spectrum(Y)))


def _get_minirocket_transform(
        minirocket_classifier: Any,
        minirocket_parameters: Any = None,
        transform_fn: TransformFn | None = None) -> Callable[[np.ndarray], np.ndarray]:
    if transform_fn is not None:
        return lambda X_batch: _ensure_2d_features(transform_fn(X_batch))

    if hasattr(minirocket_classifier, "minirocket_transform_wo_traces"):
        return lambda X_batch: _ensure_2d_features(minirocket_classifier.minirocket_transform_wo_traces(X_batch))

    if hasattr(minirocket_classifier, "minirocket_transform"):
        return lambda X_batch: _ensure_2d_features(minirocket_classifier.minirocket_transform(X_batch))

    if minirocket_parameters is not None:
        mmv = _load_minirocket_module()
        return lambda X_batch: _ensure_2d_features(
            mmv._transform_batch(np.asarray(X_batch, dtype=np.float32), parameters=minirocket_parameters)
        )

    raise ValueError(
        "Could not infer the MiniROCKET transform. Pass either a MinirocketClassifier wrapper, "
        "minirocket_parameters, or transform_fn."
    )


def _get_predict_proba(
        minirocket_classifier: Any,
        transform_minirocket: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    uses_raw_timeseries = (
        hasattr(minirocket_classifier, "minirocket_transform_wo_traces")
        or hasattr(minirocket_classifier, "minirocket_transform")
    )

    def predict_proba(X_batch: np.ndarray) -> np.ndarray:
        if uses_raw_timeseries:
            proba = minirocket_classifier.predict_proba(np.asarray(X_batch, dtype=np.float32))
        else:
            proba = minirocket_classifier.predict_proba(transform_minirocket(X_batch))
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    return predict_proba


def _classifier_classes(minirocket_classifier: Any) -> np.ndarray | None:
    classes = getattr(minirocket_classifier, "classes_", None)
    if classes is None and hasattr(minirocket_classifier, "classifier"):
        classes = getattr(minirocket_classifier.classifier, "classes_", None)
    return classes


def _class_index(minirocket_classifier: Any, target_class: Any, proba_row: np.ndarray) -> int:
    if target_class is None:
        return int(np.argmax(proba_row))

    classes = _classifier_classes(minirocket_classifier)
    if classes is not None:
        matches = np.flatnonzero(classes == target_class)
        if matches.size > 0:
            return int(matches[0])

    target_index = int(target_class)
    if target_index < 0 or target_index >= proba_row.shape[0]:
        raise ValueError(f"target_class index {target_class} is outside predict_proba output size {proba_row.shape[0]}.")
    return target_index


def _normalize_bounds(bounds: Any, shape: tuple[int, ...]) -> Any:
    if bounds is None:
        return None

    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]) and np.isscalar(bounds[1]):
        return [tuple(bounds)] * int(np.prod(shape))

    bounds_arr = np.asarray(bounds, dtype=object)
    if bounds_arr.shape == (*shape, 2):
        return [tuple(pair) for pair in bounds_arr.reshape(-1, 2)]

    if len(bounds) != int(np.prod(shape)):
        raise ValueError(
            "bounds must be None, a (low, high) tuple, an array of shape (*X.shape, 2), "
            "or a scipy-compatible sequence with one pair per variable."
        )
    return bounds


def optimize_minirocket_counterfactual(
        minirocket_classifier: Any,
        X: ArrayLike,
        X_prime: ArrayLike,
        target_class: Any = None,
        minirocket_parameters: Any = None,
        transform_fn: TransformFn | None = None,
        weights: dict[str, float] | None = None,
        probability_mode: str = "margin",
        wavelet_levels: int | None = None,
        initial: ArrayLike | None = None,
        bounds: Any = None,
        method: str = "Powell",
        maxiter: int = 100,
        tol: float = 1e-4,
        probability_eps: float = 1e-12,
        return_info: bool = False,
        optimizer_options: dict[str, Any] | None = None) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Find X_double_prime balancing three objectives:

    1. small DWT distance to X;
    2. similar real-FFT magnitude spectrum to X;
    3. small Euclidean distance between phi(X_double_prime) and phi(X_prime);
    4. a probability shift from class(X) toward class(X_prime), unless
       target_class is supplied.

    With probability_mode="margin", the optimized scalar objective is::

        w_dwt * ||DWT(Z) - DWT(X)||^2
      + w_frequency * |||rFFT(Z)| - |rFFT(X)|||^2
      + w_minirocket * ||phi(Z) - phi(X_prime)||^2
      + w_probability * (1 - (P(target_class | Z) - P(class(X) | Z)))

    If probability_mode="target_log", the probability term is instead::

      -log(P(target_class | Z))

    The two distance terms are normalized by their reference vector norms.

    Parameters
    ----------
    minirocket_classifier
        Either this repository's MinirocketClassifier wrapper, or a fitted
        classifier on MiniROCKET features with predict_proba(...).
    X, X_prime
        Multivariate time series arrays with shape (C, L).
    target_class
        Class probability to maximize. Defaults to argmax predict_proba(X_prime).
    minirocket_parameters
        MiniROCKET parameters used when minirocket_classifier is a classifier on
        already-transformed features rather than the wrapper.
    transform_fn
        Optional callable mapping a batch of shape (n, C, L) to MiniROCKET
        features. Overrides automatic transform inference.
    weights
        Non-negative weights for "dwt", "frequency", "minirocket", and
        "probability". The default frequency weight is 0.0 for backward
        compatibility.
    probability_mode
        "margin" maximizes P(target_class | Z) - P(class(X) | Z).
        "target_log" maximizes only P(target_class | Z).
    wavelet_levels
        Number of Haar-DWT levels. Defaults to all possible levels.
    initial
        Initial candidate with shape (C, L). Defaults to X.
    bounds
        Optional scipy.optimize bounds for the raw time-series variables.
    method, maxiter, tol, optimizer_options
        Parameters forwarded to scipy.optimize.minimize.
    return_info
        If True, return (X_double_prime, diagnostics).
    """
    X = _as_2d_timeseries(X, "X")
    X_prime = _as_2d_timeseries(X_prime, "X_prime")
    if X.shape != X_prime.shape:
        raise ValueError(f"X and X_prime must have the same shape; got {X.shape} and {X_prime.shape}.")

    if initial is None:
        initial = X.copy()
    else:
        initial = _as_2d_timeseries(initial, "initial")
        if initial.shape != X.shape:
            raise ValueError(f"initial must have shape {X.shape}; got {initial.shape}.")

    weights = {
        "dwt": 1.0,
        "frequency": 0.0,
        "minirocket": 1.0,
        "probability": 1.0,
        **(weights or {}),
    }
    for key in ("dwt", "frequency", "minirocket", "probability"):
        if weights[key] < 0:
            raise ValueError(f"weights[{key!r}] must be non-negative.")
    if probability_mode not in {"margin", "target_log"}:
        raise ValueError("probability_mode must be either 'margin' or 'target_log'.")

    transform_minirocket = _get_minirocket_transform(minirocket_classifier, minirocket_parameters, transform_fn)
    predict_proba = _get_predict_proba(minirocket_classifier, transform_minirocket)

    X_batch = _as_batch(X)
    X_prime_batch = _as_batch(X_prime)
    phi_prime = transform_minirocket(X_prime_batch)[0]
    dwt_X = haar_dwt_coefficients(X, levels=wavelet_levels)
    frequency_X = frequency_magnitude_spectrum(X)
    proba_X = predict_proba(X_batch)[0]
    proba_X_prime = predict_proba(X_prime_batch)[0]
    source_idx = int(np.argmax(proba_X))
    target_idx = _class_index(minirocket_classifier, target_class, proba_X_prime)
    classes = _classifier_classes(minirocket_classifier)
    source_class = classes[source_idx] if classes is not None else source_idx
    resolved_target_class = classes[target_idx] if classes is not None else target_idx

    dwt_scale = max(float(np.linalg.norm(dwt_X)), 1.0)
    frequency_scale = max(float(np.linalg.norm(frequency_X)), 1.0)
    minirocket_scale = max(float(np.linalg.norm(phi_prime)), 1.0)

    def objective_terms(flat_candidate: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
        candidate = flat_candidate.reshape(X.shape)
        candidate_batch = _as_batch(candidate)
        dwt_delta = haar_dwt_coefficients(candidate, levels=wavelet_levels) - dwt_X
        frequency_delta = frequency_magnitude_spectrum(candidate) - frequency_X
        phi_delta = transform_minirocket(candidate_batch)[0] - phi_prime
        proba = predict_proba(candidate_batch)[0]
        target_probability = float(np.clip(proba[target_idx], probability_eps, 1.0))
        source_probability = float(np.clip(proba[source_idx], probability_eps, 1.0))
        if target_idx == source_idx:
            probability_margin = target_probability
        else:
            probability_margin = target_probability - source_probability
        dwt_term = float((np.linalg.norm(dwt_delta) / dwt_scale) ** 2)
        frequency_term = float((np.linalg.norm(frequency_delta) / frequency_scale) ** 2)
        minirocket_term = float((np.linalg.norm(phi_delta) / minirocket_scale) ** 2)
        if probability_mode == "margin":
            probability_term = float(1.0 - probability_margin)
        else:
            probability_term = float(-np.log(target_probability))
        objective = (
            weights["dwt"] * dwt_term
            + weights["frequency"] * frequency_term
            + weights["minirocket"] * minirocket_term
            + weights["probability"] * probability_term
        )
        return (
            objective,
            dwt_term,
            frequency_term,
            minirocket_term,
            probability_term,
            target_probability,
            source_probability,
            probability_margin,
        )

    result = minimize(
        lambda flat_candidate: objective_terms(flat_candidate)[0],
        initial.reshape(-1),
        method=method,
        bounds=_normalize_bounds(bounds, X.shape),
        tol=tol,
        options={"maxiter": maxiter, **(optimizer_options or {})},
    )

    X_double_prime = result.x.reshape(X.shape)
    (
        final_objective,
        dwt_term,
        frequency_term,
        minirocket_term,
        probability_term,
        target_probability,
        source_probability,
        probability_margin,
    ) = objective_terms(result.x)
    (
        initial_objective,
        initial_dwt,
        initial_frequency,
        initial_minirocket,
        initial_probability_term,
        initial_target_probability,
        initial_source_probability,
        initial_probability_margin,
    ) = objective_terms(
        initial.reshape(-1)
    )

    if not return_info:
        return X_double_prime

    proba_X_double_prime = predict_proba(_as_batch(X_double_prime))[0]
    predicted_idx_X_double_prime = int(np.argmax(proba_X_double_prime))
    predicted_class_X_double_prime = classes[predicted_idx_X_double_prime] if classes is not None else predicted_idx_X_double_prime

    info = {
        "success": bool(result.success),
        "message": result.message,
        "objective": final_objective,
        "initial_objective": initial_objective,
        "dwt_term": dwt_term,
        "frequency_term": frequency_term,
        "minirocket_term": minirocket_term,
        "probability_term": probability_term,
        "initial_dwt_term": initial_dwt,
        "initial_frequency_term": initial_frequency,
        "initial_minirocket_term": initial_minirocket,
        "initial_probability_term": initial_probability_term,
        "probability_mode": probability_mode,
        "source_class": source_class,
        "source_class_index": source_idx,
        "target_class": resolved_target_class,
        "target_class_index": target_idx,
        "predicted_class_X": source_class,
        "predicted_class_X_prime": classes[int(np.argmax(proba_X_prime))] if classes is not None else int(np.argmax(proba_X_prime)),
        "predicted_class_X_double_prime": predicted_class_X_double_prime,
        "probability_source_X": float(proba_X[source_idx]),
        "probability_source_X_prime": float(proba_X_prime[source_idx]),
        "probability_source_initial": initial_source_probability,
        "probability_source_X_double_prime": source_probability,
        "probability_target_X": float(proba_X[target_idx]),
        "probability_target_X_prime": float(proba_X_prime[target_idx]),
        "probability_target_initial": initial_target_probability,
        "probability_target_X_double_prime": target_probability,
        "probability_margin_initial": initial_probability_margin,
        "probability_margin_X_double_prime": probability_margin,
        "probability_X": float(proba_X[target_idx]),
        "probability_X_prime": float(proba_X_prime[target_idx]),
        "probability_initial": initial_target_probability,
        "probability_predicted_X_double_prime": float(proba_X_double_prime[predicted_idx_X_double_prime]),
        "dwt_distance": dwt_distance(X_double_prime, X, levels=wavelet_levels),
        "frequency_magnitude_distance": frequency_magnitude_distance(X_double_prime, X),
        "minirocket_distance": float(np.linalg.norm(transform_minirocket(_as_batch(X_double_prime))[0] - phi_prime)),
        "optimizer": result,
    }
    return X_double_prime, info
