#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("JOBLIB_START_METHOD", "threading")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

TSHAP_REPO_PATH = REPO_ROOT.parent / "tshap"
if TSHAP_REPO_PATH.exists() and str(TSHAP_REPO_PATH) not in sys.path:
    sys.path.append(str(TSHAP_REPO_PATH))

from classifier import MinirocketClassifier
from counterfactual import dwt_distance, frequency_magnitude_distance, optimize_minirocket_counterfactual
from explainer import MinirocketExplainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LogisticRegression MiniROCKET classifier, compute X' with "
            "a reference policy, optimize X'' toward class(X'), and plot X, X', X''."
        )
    )
    parser.add_argument("--dataset", default="ford-a", help="Dataset name. Default: ford-a.")
    parser.add_argument("--instance-index", "--instance", type=int, default=0, help="Test instance index. Default: 0.")
    parser.add_argument("--forda-dir", type=Path, default=None, help="Directory containing FordA_TRAIN.tsv and FordA_TEST.tsv.")
    parser.add_argument("--num-features", type=int, default=504, help="MiniROCKET feature count request.")
    parser.add_argument("--max-dilations-per-kernel", type=int, default=16)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reference-policy", default="opposite_class_medoid")
    parser.add_argument("--reference-distance", choices=["euclidean", "pca-mr"], default="euclidean")
    parser.add_argument("--method", default="COBYLA", help="scipy.optimize.minimize method.")
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--initial-blend", type=float, default=0.05, help="Initial Z = (1-a)X + aX'.")
    parser.add_argument(
        "--dtw-align-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DTW-align X' to X before building the initial blend.",
    )
    parser.add_argument(
        "--aligned-reference-objective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the DTW-aligned X' in the MiniROCKET objective. Defaults to the raw reference.",
    )
    parser.add_argument("--weight-dwt", type=float, default=0.5)
    parser.add_argument(
        "--weight-frequency",
        type=float,
        default=1.0,
        help="Weight for matching |rFFT(X'')| to |rFFT(X)|. Set 0 to disable.",
    )
    parser.add_argument("--weight-minirocket", type=float, default=1.0)
    parser.add_argument("--weight-probability", type=float, default=1.0)
    parser.add_argument(
        "--probability-mode",
        choices=["margin", "target_log"],
        default="margin",
        help="Probability loss passed to optimize_minirocket_counterfactual.",
    )
    parser.add_argument("--wavelet-levels", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("/tmp/forda_counterfactual_curves.png"))
    return parser.parse_args()


def candidate_forda_dirs(args: argparse.Namespace) -> list[Path]:
    candidates = []
    if args.forda_dir is not None:
        candidates.append(args.forda_dir)
    if os.environ.get("FORDA_DIR"):
        candidates.append(Path(os.environ["FORDA_DIR"]))
    candidates.extend([
        REPO_ROOT / "data" / "FordA",
        Path.home() / "Dropbox" / "LACODAM" / "iCoRA" / "Time-Series" / "notebooks"
        / "raw.githubusercontent.com" / "hfawaz" / "cd-diagram" / "master" / "FordA",
    ])
    return candidates


def read_forda_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    labels = df.iloc[:, 0].to_numpy()
    y = np.where(labels.astype(float) < 0, 0, 1).astype(int)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32).reshape(len(df), 1, -1)
    return X, y


def as_float32_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n, C, L) or (n, L), got {X.shape}.")
    return X


def load_forda(args: argparse.Namespace) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], str]:
    for forda_dir in candidate_forda_dirs(args):
        train_path = forda_dir / "FordA_TRAIN.tsv"
        test_path = forda_dir / "FordA_TEST.tsv"
        if train_path.exists() and test_path.exists():
            return read_forda_tsv(train_path), read_forda_tsv(test_path), str(forda_dir)

    import utils

    (X_train, y_train), (X_test, y_test) = utils.get_forda_for_classification()
    return (as_float32_3d(X_train), np.asarray(y_train, dtype=int)), (
        as_float32_3d(X_test),
        np.asarray(y_test, dtype=int),
    ), "sktime"


def dataset_fetchers(args: argparse.Namespace) -> dict[str, tuple[str, object]]:
    import utils

    cognitive_path = REPO_ROOT / "data" / "cognitive-circles"
    return {
        "ford-a": ("FordA", lambda: load_forda(args)),
        "double-freq-test": (
            "DoubleFreqTest",
            lambda: (*utils.get_double_freq_test_for_classification(n_samples=200), "synthetic"),
        ),
        "abnormal-heartbeat-c1": (
            "AbnormalHeartbeat target=1",
            lambda: (*utils.get_abnormal_hearbeat_for_classification("1"), "sktime"),
        ),
        "starlight-c1": (
            "StarLightCurves target=1",
            lambda: (*utils.get_starlightcurves_for_classification("1"), "sktime"),
        ),
        "starlight-c2": (
            "StarLightCurves target=2",
            lambda: (*utils.get_starlightcurves_for_classification("2"), "sktime"),
        ),
        "starlight-c3": (
            "StarLightCurves target=3",
            lambda: (*utils.get_starlightcurves_for_classification("3"), "sktime"),
        ),
        "cognitive-circles": (
            "Cognitive Circles",
            lambda: (*utils.get_cognitive_circles_data_for_classification(
                str(cognitive_path),
                target_col="RealDifficulty",
                as_numpy=True,
            ), str(cognitive_path)),
        ),
        "handoutlines": (
            "HandOutlines target=1",
            lambda: (*utils.get_handoutlines_for_classification("1"), "sktime"),
        ),
    }


def load_dataset(args: argparse.Namespace) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        str,
        str]:
    fetchers = dataset_fetchers(args)
    dataset_key = args.dataset.lower()
    if dataset_key not in fetchers:
        valid = ", ".join(sorted(fetchers))
        raise ValueError(f"Unknown dataset '{args.dataset}'. Valid datasets: {valid}.")

    dataset_label, fetcher = fetchers[dataset_key]
    (X_train, y_train), (X_test, y_test), data_source = fetcher()
    return (
        (as_float32_3d(X_train), np.asarray(y_train, dtype=int)),
        (as_float32_3d(X_test), np.asarray(y_test, dtype=int)),
        str(data_source),
        dataset_label,
    )


def predict_label_and_probability(classifier: MinirocketClassifier, x: np.ndarray, class_idx: int | None = None) -> tuple[int, float]:
    proba = classifier.predict_proba(x[None, ...])[0]
    pred = int(np.argmax(proba))
    if class_idx is None:
        class_idx = pred
    return pred, float(proba[class_idx])


def dtw_align_to_reference(reference: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Warp `series` onto the time axis of `reference` using multivariate DTW.

    Both inputs are arrays with shape (C, L). The returned aligned series has
    shape (C, reference_length). When several source timestamps align to the
    same reference timestamp, their channel values are averaged.
    """
    reference = np.asarray(reference, dtype=np.float64)
    series = np.asarray(series, dtype=np.float64)
    if reference.ndim != 2 or series.ndim != 2:
        raise ValueError(f"DTW expects (C, L) arrays; got {reference.shape} and {series.shape}.")
    if reference.shape[0] != series.shape[0]:
        raise ValueError(f"DTW channel counts differ: {reference.shape[0]} and {series.shape[0]}.")

    _, ref_len = reference.shape
    _, series_len = series.shape
    costs = np.full((ref_len + 1, series_len + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0

    for i in range(1, ref_len + 1):
        ref_value = reference[:, i - 1]
        for j in range(1, series_len + 1):
            local_cost = np.linalg.norm(ref_value - series[:, j - 1])
            costs[i, j] = local_cost + min(costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])

    path = []
    i, j = ref_len, series_len
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        predecessor = np.argmin((costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]))
        if predecessor == 0:
            i -= 1
            j -= 1
        elif predecessor == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()

    source_indices_per_ref = [[] for _ in range(ref_len)]
    for ref_idx, series_idx in path:
        source_indices_per_ref[ref_idx].append(series_idx)

    aligned = np.empty((reference.shape[0], ref_len), dtype=np.float64)
    last_source_idx = 0
    for ref_idx, source_indices in enumerate(source_indices_per_ref):
        if source_indices:
            last_source_idx = int(round(float(np.mean(source_indices))))
            aligned[:, ref_idx] = series[:, source_indices].mean(axis=1)
        else:
            aligned[:, ref_idx] = series[:, last_source_idx]

    return aligned.astype(series.dtype, copy=False), float(costs[ref_len, series_len])


def plot_curves(
        X: np.ndarray,
        X_prime: np.ndarray,
        X_double_prime: np.ndarray,
        output: Path,
        title: str,
        X_prime_raw: np.ndarray | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    channels = X.shape[0]
    fig, axes = plt.subplots(channels, 1, figsize=(12, max(4, 2.6 * channels)), sharex=True)
    if channels == 1:
        axes = [axes]

    time_axis = np.arange(X.shape[-1])
    for channel_idx, ax in enumerate(axes):
        ax.plot(time_axis, X[channel_idx], label="X: first FordA test instance", linewidth=2.0)
        if X_prime_raw is not None:
            ax.plot(
                time_axis,
                X_prime_raw[channel_idx],
                label=f"raw X': opposite-class medoid",
                linewidth=1.0,
                linestyle="--",
                alpha=0.45,
            )
            x_prime_label = "X': DTW-aligned opposite-class medoid"
        else:
            x_prime_label = "X': opposite-class medoid"
        ax.plot(time_axis, X_prime[channel_idx], label=x_prime_label, linewidth=1.4, alpha=0.9)
        ax.plot(time_axis, X_double_prime[channel_idx], label="X'': optimized", linewidth=1.8, alpha=0.95)
        ax.set_ylabel(f"channel {channel_idx}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    (X_train, y_train), (X_test, y_test), data_source, dataset_label = load_dataset(args)
    if args.instance_index < 0 or args.instance_index >= len(X_test):
        raise ValueError(f"instance-index must be in [0, {len(X_test) - 1}], got {args.instance_index}.")
    X = X_test[args.instance_index]

    base_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.logistic_max_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        ),
    )
    classifier = MinirocketClassifier(base_classifier)

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(
            X_train,
            y_train,
            num_features=args.num_features,
            max_dilations_per_kernel=args.max_dilations_per_kernel,
            diff=True,
        )
    training_seconds = time.perf_counter() - started

    y_pred, p_x = predict_label_and_probability(classifier, X)
    MinirocketExplainer.REFERENCE_DISTANCE = args.reference_distance
    explainer = classifier.get_explainer(X_train, y_train)
    X_prime_raw = explainer.get_reference(X, y_pred, args.reference_policy)
    y_ref_pred, p_ref_pred = predict_label_and_probability(classifier, X_prime_raw)
    _, p_ref_source = predict_label_and_probability(classifier, X_prime_raw, class_idx=y_pred)

    if args.dtw_align_reference:
        X_prime_initial, dtw_cost = dtw_align_to_reference(X, X_prime_raw)
    else:
        X_prime_initial = X_prime_raw
        dtw_cost = None
    X_prime_objective = X_prime_initial if args.aligned_reference_objective else X_prime_raw

    initial = (1.0 - args.initial_blend) * X + args.initial_blend * X_prime_initial
    weights = {
        "dwt": args.weight_dwt,
        "frequency": args.weight_frequency,
        "minirocket": args.weight_minirocket,
        "probability": args.weight_probability,
    }
    X_double_prime, info = optimize_minirocket_counterfactual(
        classifier,
        X,
        X_prime_objective,
        target_class=y_ref_pred,
        weights=weights,
        probability_mode=args.probability_mode,
        wavelet_levels=args.wavelet_levels,
        initial=initial,
        method=args.method,
        maxiter=args.optimizer_maxiter,
        tol=args.tol,
        return_info=True,
        optimizer_options={"rhobeg": 0.25} if args.method.upper() == "COBYLA" else None,
    )
    y_cf_pred, p_cf_target = predict_label_and_probability(classifier, X_double_prime, class_idx=y_ref_pred)
    _, p_cf_source = predict_label_and_probability(classifier, X_double_prime, class_idx=y_pred)

    title = (
        f"{dataset_label} test[{args.instance_index}], true={int(y_test[args.instance_index])}, "
        f"pred(X)={y_pred}, pred(X')={y_ref_pred}; "
        f"reference={args.reference_policy}, probability={args.probability_mode}"
    )
    plot_curves(
        X,
        X_prime_initial,
        X_double_prime,
        args.output,
        title,
        X_prime_raw=X_prime_raw if args.dtw_align_reference else None,
    )

    print(f"data_source={data_source}")
    print(f"dataset={args.dataset.lower()}")
    print(f"dataset_label={dataset_label}")
    print(f"instance_index={args.instance_index}")
    print(f"train_shape={X_train.shape} test_shape={X_test.shape}")
    print(f"training_seconds={training_seconds:.3f}")
    print(f"true_label_X={int(y_test[args.instance_index])}")
    print(f"source_class_from_X={y_pred} probability_source_X={p_x:.6f}")
    print(f"target_class_from_X_prime={y_ref_pred} probability_target_X_prime={p_ref_pred:.6f}")
    print(f"probability_source_X_prime={p_ref_source:.6f}")
    print(f"predicted_label_X_double_prime={y_cf_pred}")
    print(f"probability_target_X_double_prime={p_cf_target:.6f}")
    print(f"probability_source_X_double_prime={p_cf_source:.6f}")
    print(f"probability_margin_X_double_prime={p_cf_target - p_cf_source:.6f}")
    print(f"dtw_align_reference={args.dtw_align_reference}")
    if dtw_cost is not None:
        print(f"dtw_cost_X_to_X_prime={dtw_cost:.6f}")
    print(f"aligned_reference_objective={args.aligned_reference_objective}")
    print(f"dwt_distance_X_to_X_prime_raw={dwt_distance(X, X_prime_raw, levels=args.wavelet_levels):.6f}")
    print(f"dwt_distance_X_to_X_prime_initial={dwt_distance(X, X_prime_initial, levels=args.wavelet_levels):.6f}")
    print(f"dwt_distance_X_to_X_double_prime={info['dwt_distance']:.6f}")
    print(f"frequency_magnitude_distance_X_to_X_prime_raw={frequency_magnitude_distance(X, X_prime_raw):.6f}")
    print(f"frequency_magnitude_distance_X_to_X_prime_initial={frequency_magnitude_distance(X, X_prime_initial):.6f}")
    print(f"frequency_magnitude_distance_X_to_X_double_prime={info['frequency_magnitude_distance']:.6f}")
    print(f"frequency_term_initial={info['initial_frequency_term']:.6f}")
    print(f"frequency_term_final={info['frequency_term']:.6f}")
    print(f"minirocket_distance_X_double_prime_to_X_prime={info['minirocket_distance']:.6f}")
    print(f"probability_mode={info['probability_mode']}")
    print(f"probability_margin_initial={info['probability_margin_initial']:.6f}")
    print(f"probability_margin_final={info['probability_margin_X_double_prime']:.6f}")
    print(f"objective_initial={info['initial_objective']:.6f} objective_final={info['objective']:.6f}")
    print(f"optimizer_success={info['success']} message={info['message']}")
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
