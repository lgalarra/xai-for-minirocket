#!/usr/bin/env python3
"""Generate LaTeX figure* grids for reference-policy boxplots."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "bar_charts_reference_policy"
DEFAULT_OUTPUT = DEFAULT_SOURCE_DIR / "reference_policy_figures.tex"
DEFAULT_LATEX_FIGURES_DIR = Path("figures")
DEFAULT_COLUMNS = 2
DEFAULT_CHARTS_PER_FIGURE = 4
DEFAULT_CHART_FORMAT = "tikz"
EXCLUDED_PERTURBATION_POLICIES = {"gradient_gaussian"}
EXPLAINER_LABELS = {
    "extreme_feature_coalitions": "EFC",
    "gradients": "Gradients",
    "shap": "SHAP",
    "stratoshap-k1": "ST-SHAP",
}
DATASET_LABELS = {
    "abnormal-heartbeat": "abnormal-heartbeat",
    "cognitive-circles": "cognitive-circles",
    "double-freq-test": "double-freq-test",
    "ford-a": "ford-a",
    "handoutlines": "handoutlines",
    "starlight": "starlight",
}
PERTURBATION_LABELS = {
    "gaussian": "noise addition perturbation regime",
    "gradient_gaussian": "noise addition perturbation regime",
    "gradient_gaussian_opposite": "opposite-class noise addition perturbation regime",
    "instance_to_reference": "occlusion perturbation regime",
    "reference_to_instance": "reference-to-instance perturbation policy",
}
EXPLAINER_PATTERN = "|".join(
    re.escape(value) for value in sorted(EXPLAINER_LABELS, key=len, reverse=True)
)
FILENAME_RE = re.compile(
    rf"^(?P<dataset>.+)_(?P<explainer>{EXPLAINER_PATTERN})_"
    rf"(?P<metric>f_minus_f0)_(?P<perturbation_policy>.+)_"
    rf"reference_policy_boxplot\.(?P<chart_format>tex|png)$"
)


@dataclass(frozen=True)
class Chart:
    path: Path
    dataset: str
    explainer: str
    metric: str
    perturbation_policy: str
    chart_format: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write LaTeX figure* grids for the charts produced by "
            "generate_bars_per_reference_policy.py."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing the generated charts. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path where the LaTeX snippet is written. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--latex-figures-dir",
        type=Path,
        default=DEFAULT_LATEX_FIGURES_DIR,
        help=(
            "Directory prefix used in LaTeX chart paths. "
            "Default: figures"
        ),
    )
    parser.add_argument(
        "--chart-format",
        choices=("tikz", "png"),
        default=DEFAULT_CHART_FORMAT,
        help=(
            "Discover TikZ .tex charts and serialize them with \\appendixchart, or use "
            f"legacy PNG charts. Default: {DEFAULT_CHART_FORMAT}."
        ),
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help=f"Number of chart columns in each figure*. Default: {DEFAULT_COLUMNS}.",
    )
    parser.add_argument(
        "--charts-per-figure",
        type=int,
        default=DEFAULT_CHARTS_PER_FIGURE,
        help=f"Maximum charts in each figure*. Default: {DEFAULT_CHARTS_PER_FIGURE}.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def label_part(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def discover_charts(
    source_dir: Path,
    chart_format: str = DEFAULT_CHART_FORMAT,
) -> list[Chart]:
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    charts = []
    extension = ".tex" if chart_format == "tikz" else ".png"
    for path in sorted(source_dir.glob(f"*{extension}")):
        if path.name == DEFAULT_OUTPUT.name:
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"Skipping chart with unrecognized name: {path}")
            continue
        chart = Chart(path=path, **match.groupdict())
        if chart.perturbation_policy in EXCLUDED_PERTURBATION_POLICIES:
            continue
        if (
            chart.perturbation_policy == "instance_to_reference"
            and chart.explainer == "gradients"
        ):
            continue
        charts.append(chart)

    if not charts:
        raise SystemExit(
            f"No reference-policy {chart_format} charts found under {source_dir}"
        )
    return charts


def grouped_charts(charts: list[Chart]) -> dict[tuple[str, str, str], list[Chart]]:
    groups: dict[tuple[str, str, str], list[Chart]] = defaultdict(list)
    for chart in charts:
        groups[(chart.perturbation_policy, chart.explainer, chart.metric)].append(chart)
    return {
        key: sorted(group, key=lambda chart: (chart.dataset, chart.path.name))
        for key, group in sorted(groups.items())
    }


def chunked(charts: list[Chart], chunk_size: int) -> list[list[Chart]]:
    if chunk_size < 1:
        raise SystemExit("--charts-per-figure must be at least 1")
    return [
        charts[index : index + chunk_size]
        for index in range(0, len(charts), chunk_size)
    ]


def write_figure(
    lines: list[str],
    charts: list[Chart],
    *,
    caption: str,
    label: str,
    columns: int,
    latex_figures_dir: Path,
) -> None:
    if columns < 1:
        raise SystemExit("--columns must be at least 1")

    width = min(0.98 / columns, 0.98)
    lines.extend([r"\begin{figure*}[htbp]", r"\centering"])
    for index, chart in enumerate(charts, start=1):
        figure_path = (latex_figures_dir / chart.path.name).as_posix()
        dataset = DATASET_LABELS.get(chart.dataset, chart.dataset)
        lines.extend(
            [
                rf"\begin{{subfigure}}[t]{{{width:.3f}\textwidth}}",
                r"\centering",
                (
                    rf"\appendixchart{{{figure_path}}}"
                    if chart.chart_format == "tex"
                    else rf"\includegraphics[width=\linewidth]{{{figure_path}}}"
                ),
                rf"\caption{{{latex_escape(dataset)}}}",
                r"\end{subfigure}",
            ]
        )
        if index % columns == 0 and index != len(charts):
            lines.append(r"\par\medskip")
        elif index != len(charts):
            lines.append(r"\hfill")

    lines.extend(
        [
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{figure*}",
            "",
        ]
    )


def write_latex_figures(
    charts: list[Chart],
    output: Path,
    latex_figures_dir: Path,
    columns: int,
    charts_per_figure: int,
) -> tuple[int, int]:
    groups = grouped_charts(charts)
    lines = [
        "% Generated by generate_reference_policy_latex_figures.py.",
        "% Assumes these chart files are available under the LaTeX figures/ directory.",
        "% Requires \\usepackage{graphicx}.",
        "% TikZ charts also require \\usepackage{pgfplots},",
        "% \\usepgfplotslibrary{statistics}, and \\pgfplotsset{compat=1.18}.",
        "% Requires \\usepackage{subcaption}.",
        "% TikZ charts require the paper preamble's \\appendixchart helper,",
        "% which immediately serializes each chart with \\pdfxform to bound pdfTeX memory.",
        "",
    ]
    figure_count = 0
    for (policy, explainer, metric), group in groups.items():
        parts = chunked(group, charts_per_figure)
        explainer_label = EXPLAINER_LABELS.get(explainer, explainer)
        perturbation_label = PERTURBATION_LABELS.get(
            policy, f"{policy.replace('_', '-')} perturbation policy"
        )
        caption = (
            f"Reference-policy comparison for {explainer_label} using the "
            f"{perturbation_label}."
        )
        base_label = (
            "fig:reference-policy-"
            f"{label_part(policy)}-{label_part(explainer)}-{label_part(metric)}"
        )
        for part_number, part in enumerate(parts, start=1):
            part_caption = caption
            part_label = base_label
            if len(parts) > 1:
                part_caption = f"{caption} Part {part_number} of {len(parts)}."
                part_label = f"{base_label}-part-{part_number}"
            write_figure(
                lines,
                part,
                caption=part_caption,
                label=part_label,
                columns=columns,
                latex_figures_dir=latex_figures_dir,
            )
            figure_count += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return len(groups), figure_count


def main() -> None:
    args = parse_args()
    charts = discover_charts(args.source_dir, args.chart_format)
    group_count, figure_count = write_latex_figures(
        charts,
        args.output,
        args.latex_figures_dir,
        args.columns,
        args.charts_per_figure,
    )
    print(f"Wrote LaTeX figure snippet: {args.output}")
    print(f"Charts: {len(charts)}")
    print(f"Chart groups: {group_count}")
    print(f"figure* environments: {figure_count}")


if __name__ == "__main__":
    main()
