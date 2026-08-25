#!/usr/bin/env python3
"""Plot LRP target metric comparisons from official result CSV files."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "official-lrp-results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "metric-comparison-plots-by-explainer"
TARGET_METRICS = ("f_minus_f0", "lrp_f_minus_f0")
METRIC_LABELS = {
    "f_minus_f0": "DeepLift",
    "lrp_f_minus_f0": "LRP",
}
METRIC_VALUE_COLUMNS = {
    "f_minus_f0": ("f_minus_f0-mean", "f_minus_f0"),
    "lrp_f_minus_f0": ("lrp_f_minus_f0-mean", "lrp_f_minus_f0"),
}
POLICY_BY_CHART = {
    "instance_to_reference": {"shap", "stratoshap-k1"},
    "gaussian": {"shap", "stratoshap-k1"},
    "gradient_gaussian": {"gradient"},
}
POLICIES = tuple(POLICY_BY_CHART)
EXPLAINERS = tuple(sorted({name for names in POLICY_BY_CHART.values() for name in names}))

DATASET_COLUMNS = ("dataset", "dataset_name", "data_set", "task", "name")
EXPLAINER_COLUMNS = ("base_explainer", "explainer", "explainer_name", "method", "attribution_method")
POLICY_COLUMNS = (
    "perturbation_policy",
    "perturbation",
    "policy",
    "regime",
    "perturbation_regime",
)
PERCENTILE_COLUMNS = ("percentile_cut", "percentile", "cut_percentile")
ARGS_COLUMN = "args"


@dataclass(frozen=True)
class Observation:
    csv_path: Path
    dataset: str
    explainer: str
    policy: str
    percentile_cut: float
    metric: str
    value: float


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_explainer(value: str) -> str:
    normalized = normalize_token(value).replace("_", "-")
    aliases = {
        "stratoshap-k-1": "stratoshap-k1",
        "stratoshap-k_1": "stratoshap-k1",
        "stratoshap-k1": "stratoshap-k1",
        "gradient": "gradient",
        "gradients": "gradient",
    }
    return aliases.get(normalized, normalized)


def normalize_policy(value: str) -> str:
    normalized = normalize_token(value)
    aliases = {
        "instance_reference": "instance_to_reference",
        "instance2reference": "instance_to_reference",
        "instance_to_ref": "instance_to_reference",
        "gradient-gaussian": "gradient_gaussian",
    }
    return aliases.get(normalized, normalized)


def metric_value_column(metric: str, fieldnames: list[str]) -> str | None:
    for column in METRIC_VALUE_COLUMNS[metric]:
        if column in fieldnames:
            return column
    return None


def first_present(row: dict[str, str], columns: Iterable[str]) -> str | None:
    for column in columns:
        if column in row and row[column].strip():
            return row[column].strip()
    return None


def find_percentile_column(fieldnames: list[str]) -> str | None:
    for column in PERCENTILE_COLUMNS:
        if column in fieldnames:
            return column
    return None


def percentile_from_args(row: dict[str, str], path: Path) -> float | None:
    value = row.get(ARGS_COLUMN)
    if not value:
        return None
    try:
        args = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        print(f"Skipping row with unparsable args in {path}")
        return None
    if not isinstance(args, dict):
        return None
    return parse_float(str(args.get("percentile_cut", "")), column="args.percentile_cut", path=path)


def infer_from_path(path: Path) -> tuple[str | None, str | None, str | None]:
    text = normalize_token(" ".join(path.parts))

    policy = None
    for candidate in sorted(POLICIES, key=len, reverse=True):
        if normalize_token(candidate) in text:
            policy = candidate
            break

    explainer = None
    for candidate in sorted(EXPLAINERS, key=len, reverse=True):
        if normalize_token(candidate) in text or candidate.replace("-", "_") in text:
            explainer = normalize_explainer(candidate)
            break

    dataset = None
    if policy:
        dataset = path.stem
        dataset = re.sub(normalize_token(policy), "", normalize_token(dataset))
        if explainer:
            dataset = re.sub(normalize_token(explainer), "", dataset)
            dataset = re.sub(explainer.replace("-", "_"), "", dataset)
        dataset = dataset.strip("_-") or None

    return dataset, explainer, policy


def parse_float(value: str, *, column: str, path: Path) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        print(f"Skipping non-numeric {column!r} value {value!r} in {path}")
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_metric_value(row: dict[str, str], metric: str, fieldnames: list[str], path: Path) -> float | None:
    column = metric_value_column(metric, fieldnames)
    if column is None:
        return None

    value = row[column]
    parsed = parse_float(value, column=column, path=path)
    if parsed is not None:
        return parsed

    values = [
        parse_float(part, column=column, path=path)
        for part in value.split(";")
        if part.strip()
    ]
    values = [part for part in values if part is not None]
    if values:
        return mean(values)
    return None


def read_observations(results_dir: Path) -> list[Observation]:
    observations: list[Observation] = []
    csv_paths = sorted(results_dir.rglob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {results_dir}")

    for path in csv_paths:
        inferred_dataset, inferred_explainer, inferred_policy = infer_from_path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue

            percentile_column = find_percentile_column(reader.fieldnames)
            metric_columns = {
                metric: metric_value_column(metric, reader.fieldnames)
                for metric in TARGET_METRICS
            }
            missing_metrics = [
                metric
                for metric, column in metric_columns.items()
                if column is None
            ]
            if (percentile_column is None and ARGS_COLUMN not in reader.fieldnames) or missing_metrics:
                print(
                    f"Skipping {path}: missing "
                    f"{'percentile_cut/args' if percentile_column is None else ''} "
                    f"{', '.join(missing_metrics)}".strip()
                )
                continue

            for row in reader:
                dataset = first_present(row, DATASET_COLUMNS) or inferred_dataset
                explainer = first_present(row, EXPLAINER_COLUMNS) or inferred_explainer
                policy = first_present(row, POLICY_COLUMNS) or inferred_policy
                if not dataset or not explainer or not policy:
                    continue

                explainer = normalize_explainer(explainer)
                policy = normalize_policy(policy)
                allowed_explainers = POLICY_BY_CHART.get(policy)
                if allowed_explainers is None or explainer not in allowed_explainers:
                    continue

                if percentile_column is not None:
                    percentile_cut = parse_float(row[percentile_column], column=percentile_column, path=path)
                else:
                    percentile_cut = percentile_from_args(row, path)
                if percentile_cut is None:
                    continue

                for metric in TARGET_METRICS:
                    value = parse_metric_value(row, metric, reader.fieldnames, path)
                    if value is None:
                        continue
                    observations.append(
                        Observation(
                            csv_path=path,
                            dataset=dataset,
                            explainer=explainer,
                            policy=policy,
                            percentile_cut=percentile_cut,
                            metric=metric,
                            value=value,
                        )
                    )

    if not observations:
        raise SystemExit(
            "No usable rows found. Expected columns for percentile_cut, "
            "f_minus_f0, lrp_f_minus_f0, and either columns or filenames "
            "identifying dataset, explainer, and perturbation policy."
        )
    return observations


def aggregate(
    observations: Iterable[Observation],
) -> dict[tuple[str, str, str, str, float], list[float]]:
    grouped: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    for obs in observations:
        x_value = 100.0 - obs.percentile_cut
        key = (obs.dataset, obs.policy, obs.explainer, obs.metric, x_value)
        grouped[key].append(obs.value)
    return grouped


def sanitize_filename(value: str) -> str:
    return normalize_token(value).replace("_", "-") or "unknown"


def plot_charts(observations: list[Observation], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = aggregate(observations)
    datasets = sorted({obs.dataset for obs in observations})
    written: list[Path] = []

    linestyles = {
        "f_minus_f0": "-",
        "lrp_f_minus_f0": "--",
    }
    metric_colors = {
        "f_minus_f0": "#2f6fbb",
        "lrp_f_minus_f0": "#b13f48",
    }

    for dataset in datasets:
        for policy, explainers in POLICY_BY_CHART.items():
            for explainer in sorted(explainers):
                series_keys = [
                    key
                    for key in grouped
                    if key[0] == dataset and key[1] == policy and key[2] == explainer
                ]
                if not series_keys:
                    continue

                fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
                for metric in TARGET_METRICS:
                    points = sorted(
                        (
                            (x_value, values)
                            for (dset, pol, exp, met, x_value), values in grouped.items()
                            if dset == dataset
                            and pol == policy
                            and exp == explainer
                            and met == metric
                        ),
                        key=lambda item: item[0],
                    )
                    if not points:
                        continue

                    x_values = [point[0] for point in points]
                    y_values = [mean(point[1]) for point in points]
                    label = METRIC_LABELS[metric]
                    ax.plot(
                        x_values,
                        y_values,
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        color=metric_colors[metric],
                        linestyle=linestyles[metric],
                        label=label,
                    )

                    y_errors = [
                        stdev(values) if len(values) > 1 else 0.0
                        for _, values in points
                    ]
                    if any(error > 0 for error in y_errors):
                        lower = [y - error for y, error in zip(y_values, y_errors)]
                        upper = [y + error for y, error in zip(y_values, y_errors)]
                        ax.fill_between(
                            x_values,
                            lower,
                            upper,
                            color=metric_colors[metric],
                            alpha=0.12,
                            linewidth=0,
                        )

                ax.set_title(f"{dataset} - {policy} - {explainer}")
                ax.set_xlabel("% observations perturbed")
                ax.set_ylabel("Probability drop")
                ax.grid(True, axis="both", alpha=0.25)
                ax.legend(frameon=False, fontsize=9)

                output_path = output_dir / f"{sanitize_filename(dataset)}__{policy}__{explainer}.png"
                fig.savefig(output_path, dpi=200)
                plt.close(fig)
                written.append(output_path)

    if not written:
        raise SystemExit("No charts were generated for the requested explainers and policies.")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare f_minus_f0 and lrp_f_minus_f0 across percentile cuts for "
            "SHAP, StratoSHAP-k1, and gradient perturbation regimes."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing official result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG charts are written.",
    )
    args = parser.parse_args()

    observations = read_observations(args.results_dir)
    written = plot_charts(observations, args.output_dir)
    print(f"Wrote {len(written)} chart(s):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
