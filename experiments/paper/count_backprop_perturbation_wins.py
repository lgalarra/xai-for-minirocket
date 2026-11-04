#!/usr/bin/env python3
"""Count when backpropagation beats alternative explanation/baseline curves."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import generate_all_official_perturbation_charts as all_charts
import generate_official_perturbation_charts as charts


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "official-results"
DEFAULT_OUTPUT = SCRIPT_DIR / "backprop_perturbation_win_counts.csv"
DEFAULT_LATEX_OUTPUT = SCRIPT_DIR / "backprop_perturbation_win_counts.tex"
PERTURBATION_SETTINGS = ("instance_to_reference", "gaussian")
LATEX_PERTURBATION_SETTINGS = ("instance_to_reference", "gaussian")
PRIMARY_METRIC = "f_minus_f0-mean"
E2E_METRIC = "p2p_f_minus_f0-mean"
COMPARATORS = ("e2e", "segmented", "tshap", "random")
BOTTOM_COMPARATOR = "bottom"
UNIT_COLUMNS = ("dataset", "base_explainer", "mr_classifier")
CLASSIFIER_ORDER = ("LogisticRegression", "MLPClassifier", "RandomForestClassifier")
EXCLUDED_REFERENCE_POLICIES = ("counterfactual",)
EXCLUDED_SETTING_EXPLAINERS = {("instance_to_reference", "gradients")}
SETTING_LABELS = {
    "instance_to_reference": "Occlusion",
    "gaussian": "Noise addition",
}
COMPARATOR_LABELS = {
    "e2e": "End-to-end",
    "segmented": "Segmented",
    "tshap": "TSHAP",
    "random": "Random",
    "bottom": "Bottom",
}


@dataclass(frozen=True)
class Args:
    data_dir: Path
    out_dir: Path
    evolution_factor: str
    metric_kind: str
    perturbation_policy: str
    percentile_cut: float
    sigma: float
    base_explainer: str
    model: str
    reference_policy: str
    best_method_random_regime: str
    include_bottom: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count dataset/classifier/explainer units where f_minus_f0 "
            "outperforms e2e, segmented, TSHAP, and random baselines."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory with perturbation-results-* CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV path for detailed and summary counts. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=DEFAULT_LATEX_OUTPUT,
        help=f"Path for the LaTeX summary table. Default: {DEFAULT_LATEX_OUTPUT}",
    )
    parser.add_argument(
        "--percentile-cut",
        type=float,
        default=90.0,
        help="percentile_cut value to analyze. Default: 90.0.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma value used by the official chart analysis. Default: 3.0.",
    )
    parser.add_argument(
        "--best-method-random-regime",
        choices=("random", "random_no_positive"),
        default="random_no_positive",
        help="Random baseline regime used by the official best-method charts.",
    )
    parser.add_argument(
        "--include-bottom",
        action="store_true",
        help="Include the bottom baseline comparator in the counts and table.",
    )
    return parser.parse_args()


def analysis_args(
    data_dir: Path,
    setting: str,
    explainer: str,
    percentile_cut: float,
    sigma: float,
    random_regime: str,
    include_bottom: bool = False,
) -> Args:
    policy = all_charts.policy_for_explainer(setting, explainer)
    return Args(
        data_dir=data_dir,
        out_dir=SCRIPT_DIR / "official-perturbation-figures",
        evolution_factor="percentile_cut",
        metric_kind="probability",
        perturbation_policy=policy,
        percentile_cut=percentile_cut,
        sigma=sigma,
        base_explainer=explainer,
        model="all",
        reference_policy="all",
        best_method_random_regime=random_regime,
        include_bottom=include_bottom,
    )


def metric_columns_for_family(metrics: list[str], family: str) -> list[str]:
    return [metric for metric in metrics if charts.metric_family(metric) == family]


def row_best_value(row: pd.Series, metrics: list[str]) -> float:
    values = pd.to_numeric(row.reindex(metrics), errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


def compare_values(backprop: float, candidate: float) -> str:
    if pd.isna(backprop) or pd.isna(candidate):
        return "missing"
    if np.isclose(backprop, candidate):
        return "tie"
    if backprop > candidate:
        return "win"
    return "loss"


def restrict_to_percentile(data: pd.DataFrame, percentile_cut: float) -> pd.DataFrame:
    if data.empty:
        return data
    return data[np.isclose(data["percentile_cut"], percentile_cut, equal_nan=False)].copy()


def exclude_reference_policies(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "reference_policy" not in data:
        return data
    return data[~data["reference_policy"].isin(EXCLUDED_REFERENCE_POLICIES)].copy()


def aggregate_primary_data(data: pd.DataFrame, percentile_cut: float) -> pd.DataFrame:
    data = restrict_to_percentile(data, percentile_cut)
    if data.empty:
        return data
    return (
        data.groupby(list(UNIT_COLUMNS), as_index=False)
        .mean(numeric_only=True)
        .sort_values(list(UNIT_COLUMNS))
    )


def aggregate_baseline_data(
    data: pd.DataFrame,
    args: Args,
) -> pd.DataFrame:
    data = restrict_to_percentile(data, args.percentile_cut)
    if data.empty or PRIMARY_METRIC not in data:
        return pd.DataFrame(columns=list(UNIT_COLUMNS) + ["average_policy_suffix", PRIMARY_METRIC])

    suffixes = charts.average_policy_suffixes(args)
    data = data.copy()
    data["average_policy_suffix"] = data["perturbation_policy"].apply(
        lambda policy: charts.average_policy_suffix(policy, suffixes)
    )
    data = data[pd.notna(data["average_policy_suffix"])]
    if data.empty:
        return pd.DataFrame(columns=list(UNIT_COLUMNS) + ["average_policy_suffix", PRIMARY_METRIC])

    return (
        data.groupby(list(UNIT_COLUMNS) + ["average_policy_suffix"], as_index=False)[PRIMARY_METRIC]
        .mean(numeric_only=True)
        .sort_values(list(UNIT_COLUMNS) + ["average_policy_suffix"])
    )


def baseline_value(
    baseline_data: pd.DataFrame,
    unit: tuple[str, str, str],
    comparator: str,
) -> float:
    suffix = "random_no_positive" if comparator == "random" else None
    if comparator == "bottom":
        suffixes = ("bottom", "bottom_unsigned")
    else:
        suffixes = (suffix,)

    dataset, explainer, classifier = unit
    rows = baseline_data[
        (baseline_data["dataset"] == dataset)
        & (baseline_data["base_explainer"] == explainer)
        & (baseline_data["mr_classifier"] == classifier)
        & (baseline_data["average_policy_suffix"].isin(suffixes))
    ]
    if rows.empty:
        return np.nan
    values = pd.to_numeric(rows[PRIMARY_METRIC], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


def unit_sort_key(unit: tuple[str, str, str]) -> tuple[str, str, int, str]:
    dataset, explainer, classifier = unit
    classifier_rank = CLASSIFIER_ORDER.index(classifier) if classifier in CLASSIFIER_ORDER else 99
    return dataset, explainer, classifier_rank, classifier


def unit_rows_for_analysis(
    setting: str,
    explainer: str,
    args: Args,
) -> list[dict[str, object]]:
    raw_data, _, _ = charts.load_data(args.data_dir, args)
    raw_data = exclude_reference_policies(raw_data)
    primary_data = charts.filter_data(raw_data, args)
    primary_agg = aggregate_primary_data(primary_data, args.percentile_cut)
    baseline_data = charts.filter_average_policy_data(raw_data, args)
    baseline_agg = aggregate_baseline_data(baseline_data, args)

    if primary_agg.empty or PRIMARY_METRIC not in primary_agg:
        return []

    metrics = charts.discover_metrics(primary_agg, args)
    segmented_metrics = metric_columns_for_family(metrics, "segmented")
    tshap_metrics = metric_columns_for_family(metrics, "tshap")
    rows = []
    for _, row in primary_agg.iterrows():
        unit = tuple(str(row[column]) for column in UNIT_COLUMNS)
        backprop_value = row.get(PRIMARY_METRIC, np.nan)
        candidates = {
            "e2e": row.get(E2E_METRIC, np.nan),
            "segmented": row_best_value(row, segmented_metrics),
            "tshap": row_best_value(row, tshap_metrics),
            "random": baseline_value(baseline_agg, unit, "random"),
        }
        if args.include_bottom:
            candidates[BOTTOM_COMPARATOR] = baseline_value(
                baseline_agg, unit, BOTTOM_COMPARATOR
            )
        for comparator, candidate_value in candidates.items():
            rows.append(
                {
                    "setting": setting,
                    "actual_policy": args.perturbation_policy,
                    "dataset": unit[0],
                    "base_explainer": unit[1],
                    "mr_classifier": unit[2],
                    "comparator": comparator,
                    "backprop_value": backprop_value,
                    "comparator_value": candidate_value,
                    "result": compare_values(backprop_value, candidate_value),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["setting"],
            row["comparator"],
            unit_sort_key((row["dataset"], row["base_explainer"], row["mr_classifier"])),
        ),
    )


def discover_explainers(data_dir: Path) -> list[str]:
    _, explainers, _, _ = all_charts.discover_values(data_dir)
    return explainers


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    counts = (
        details.groupby(["setting", "comparator", "result"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary = (
        counts.pivot_table(
            index=["setting", "comparator"],
            columns="result",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    for column in ("win", "loss", "tie", "missing"):
        if column not in summary:
            summary[column] = 0
    summary["evaluated"] = summary["win"] + summary["loss"] + summary["tie"]
    summary["total_units"] = summary["evaluated"] + summary["missing"]
    return summary[
        ["setting", "comparator", "win", "loss", "tie", "missing", "evaluated", "total_units"]
    ].sort_values(["setting", "comparator"])


def summarize_by_explainer(details: pd.DataFrame) -> pd.DataFrame:
    counts = (
        details.groupby(["setting", "base_explainer", "comparator", "result"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary = (
        counts.pivot_table(
            index=["setting", "base_explainer", "comparator"],
            columns="result",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    for column in ("win", "loss", "tie", "missing"):
        if column not in summary:
            summary[column] = 0
    summary["evaluated"] = summary["win"] + summary["loss"] + summary["tie"]
    return summary[
        ["setting", "base_explainer", "comparator", "win", "loss", "tie", "missing", "evaluated"]
    ].sort_values(["setting", "base_explainer", "comparator"])


def write_output(output: Path, details: pd.DataFrame, summary: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# summary\n")
        summary.to_csv(handle, index=False)
        handle.write("\n# details\n")
        details.to_csv(handle, index=False)


def format_win_count(win: int, evaluated: int) -> str:
    if evaluated == 0:
        return "--"
    return f"{win}/{evaluated}"


def build_latex_summary(
    summary: pd.DataFrame,
    comparators: tuple[str, ...] = COMPARATORS,
) -> str:
    indexed = summary.set_index(["setting", "comparator"])
    header_labels = [SETTING_LABELS[setting] for setting in LATEX_PERTURBATION_SETTINGS]
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Comparator & " + " & ".join(header_labels) + r" & Overall \\",
        r"\midrule",
    ]

    for comparator in comparators:
        cells = []
        total_wins = 0
        total_evaluated = 0
        for setting in LATEX_PERTURBATION_SETTINGS:
            if (setting, comparator) in indexed.index:
                row = indexed.loc[(setting, comparator)]
                wins = int(row["win"])
                evaluated = int(row["evaluated"])
            else:
                wins = 0
                evaluated = 0
            cells.append(format_win_count(wins, evaluated))
            total_wins += wins
            total_evaluated += evaluated
        cells.append(format_win_count(total_wins, total_evaluated))
        latex.append(f"{COMPARATOR_LABELS[comparator]} & " + " & ".join(cells) + r" \\")

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Number of cases in which back-propagation outperforms each "
                r"competitor. Entries report wins/evaluated comparisons; comparisons "
                r"with missing values are excluded from the denominator. Results using "
                r"the counterfactual reference policy are excluded.}"
            ),
            r"\label{tab:backprop-perturbation-wins}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def write_latex_output(
    output: Path,
    summary: pd.DataFrame,
    comparators: tuple[str, ...] = COMPARATORS,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_latex_summary(summary, comparators) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    cli_args = parse_args()
    explainers = discover_explainers(cli_args.data_dir)
    rows = []
    for setting in PERTURBATION_SETTINGS:
        for explainer in explainers:
            if (setting, explainer) in EXCLUDED_SETTING_EXPLAINERS:
                continue
            args = analysis_args(
                cli_args.data_dir,
                setting,
                explainer,
                cli_args.percentile_cut,
                cli_args.sigma,
                cli_args.best_method_random_regime,
                cli_args.include_bottom,
            )
            rows.extend(unit_rows_for_analysis(setting, explainer, args))

    if not rows:
        raise SystemExit("No rows found for the requested analysis.")

    details = pd.DataFrame(rows)
    summary = summarize(details)
    by_explainer = summarize_by_explainer(details)
    comparators = COMPARATORS + (
        (BOTTOM_COMPARATOR,) if cli_args.include_bottom else ()
    )
    write_output(cli_args.output, details, summary)
    write_latex_output(cli_args.latex_output, summary, comparators)

    print(f"Analyzed percentile_cut={cli_args.percentile_cut:g} using probability drops.")
    print(f"Excluded reference policies: {', '.join(EXCLUDED_REFERENCE_POLICIES)}")
    print(f"Included bottom comparator: {cli_args.include_bottom}")
    print(f"Wrote detailed CSV: {cli_args.output}")
    print(f"Wrote LaTeX summary table: {cli_args.latex_output}")
    print("\nOverall counts:")
    print(summary.to_string(index=False))
    print("\nCounts by explainer:")
    print(by_explainer.to_string(index=False))


if __name__ == "__main__":
    main()
