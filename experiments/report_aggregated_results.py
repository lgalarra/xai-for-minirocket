"""Generate Markdown and LaTeX comparisons of aggregated explanation results."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIELDS = ["label", "distance", "reference_strategy", "seed", "args", "count_basis"]
COMPACT_POLICIES = {
    "M": "opposite_class_medoid",
    "C": "opposite_class_closest_instance",
    "F": "opposite_class_farthest_instance",
}
REGIME_LABELS = {
    "instance_to_reference": "O",
    "gaussian": "N",
}
CLASSIFIER_LABELS = {
    "LogisticRegression": "LR",
    "MLPClassifier": "MLP",
    "RandomForestClassifier": "RF",
}


def load_comparisons(directory, explainer="shap", attribution="backpropagated", metric="f_minus_f0-mean"):
    records = []
    sources = []
    for path in sorted(Path(directory).glob("*.csv")):
        if path.stem.endswith(("-instances", "-samples")):
            continue
        frame = pd.read_csv(path)
        required = {"dataset", "mr_classifier", "base_explainer", "attribution_type", "reference_policy",
                    "aggregation", "perturbation_policy", "instance_ids", "n_perturbed_points", metric, *CONFIG_FIELDS}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
        frame = frame[(frame.base_explainer == explainer) & (frame.attribution_type == attribution)].copy()
        if frame.empty:
            continue
        frame["args"] = frame["args"].map(lambda value: json.dumps(ast.literal_eval(value), sort_keys=True))
        contexts = ["dataset", "mr_classifier", "aggregation", "perturbation_policy", *CONFIG_FIELDS]
        used = False
        for context, rows in frame.groupby(contexts, dropna=False, sort=True):
            combined = rows[rows.reference_policy.str.contains("+", regex=False)]
            for _, aggregate in combined.iterrows():
                policies = aggregate.reference_policy.split("+")
                if len(set(policies)) != len(policies) or not all(policies):
                    raise ValueError(f"{path}: invalid policy group {aggregate.reference_policy}")
                selected = rows[rows.reference_policy.isin([aggregate.reference_policy, *policies])]
                if len(selected) != len(policies) + 1 or selected.reference_policy.duplicated().any():
                    raise ValueError(f"{path}: missing or duplicate individual comparisons for {aggregate.reference_policy}")
                ids = str(aggregate.instance_ids).split(";")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"{path}: duplicate instance IDs")
                if selected.instance_ids.astype(str).nunique() != 1 or selected.n_perturbed_points.astype(str).nunique() != 1:
                    raise ValueError(f"{path}: comparison methods use different instances or point limits")
                if aggregate.count_basis != "shared_minimum":
                    raise ValueError(f"{path}: comparisons require a shared perturbation-count limit")
                values = pd.to_numeric(selected[metric], errors="raise")
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: nonfinite {metric}")
                record = dict(zip(contexts, context))
                record.update(group=aggregate.reference_policy, instances=len(ids), source=path.name,
                              combined=float(aggregate[metric]))
                record["individual"] = {policy: float(selected.loc[selected.reference_policy == policy, metric].iloc[0]) for policy in policies}
                records.append(record)
                used = True
        if used:
            sources.append(path.name)
    if not records:
        raise ValueError("No matching aggregate/individual comparisons found.")
    seen = set()
    for record in records:
        key = tuple(str(record[field]) for field in ["dataset", "mr_classifier", "group", "aggregation", "perturbation_policy", *CONFIG_FIELDS])
        if key in seen:
            raise ValueError(f"Duplicate experiment across summary files: {record['source']}. Select a directory containing one run per configuration.")
        seen.add(key)
    return records, sources


def build_tables(records):
    policies = sorted({policy for record in records for policy in record["individual"]})
    policy_labels = {policy: f"P{index + 1}" for index, policy in enumerate(policies)}
    groups = sorted({record["group"] for record in records})
    group_labels = {group: f"G{index + 1}" for index, group in enumerate(groups)}
    configurations = sorted({tuple(str(record[field]) for field in CONFIG_FIELDS) for record in records})
    config_labels = {config: f"C{index + 1}" for index, config in enumerate(configurations)}
    detailed = []
    for record in records:
        row = {"Dataset": record["dataset"], "Classifier": record["mr_classifier"],
               "Group": group_labels[record["group"]],
               "Config": config_labels[tuple(str(record[field]) for field in CONFIG_FIELDS)],
               "Perturbation": record["perturbation_policy"], "Aggregation": record["aggregation"],
               "Instances": record["instances"]}
        row.update({policy_labels[policy]: record["individual"].get(policy, np.nan) for policy in policies})
        row["Combined"] = record["combined"]
        row["Delta best"] = record["combined"] - max(record["individual"].values())
        detailed.append(row)
    classifier_table = pd.DataFrame(detailed).sort_values(["Dataset", "Group", "Config", "Perturbation", "Classifier"])
    dataset_rows = []
    keys = ["Dataset", "Group", "Config", "Perturbation", "Aggregation"]
    for context, frame in classifier_table.groupby(keys, sort=True):
        row = dict(zip(keys, context))
        row["Classifiers"] = len(frame)
        counts = sorted(frame.Instances.unique())
        row["Instances/classifier"] = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        for column in [*policy_labels.values(), "Combined"]:
            row[column] = frame[column].mean()
        row["Delta best"] = row["Combined"] - max(row[column] for column in policy_labels.values() if pd.notna(row[column]))
        dataset_rows.append(row)
    legends = {
        "Policies": {label: policy for policy, label in policy_labels.items()},
        "Groups": {label: " + ".join(policy_labels[policy] for policy in group.split("+")) for group, label in group_labels.items()},
        "Configurations": {label: "; ".join(f"{field}={value}" for field, value in zip(CONFIG_FIELDS, config)) for config, label in config_labels.items()},
    }
    return pd.DataFrame(dataset_rows), classifier_table, legends


def build_aggregation_comparison(records, by_classifier=False):
    def dataset_label(dataset):
        return "starlight" if dataset.startswith("starlight-c") else dataset

    def dataset_macro_mean(aggregation_records, value):
        dataset_means = []
        for source_dataset in sorted({record["dataset"] for record in aggregation_records}):
            values = [value(record) for record in aggregation_records if record["dataset"] == source_dataset]
            if values:
                dataset_means.append(np.mean(values))
        return np.mean(dataset_means) if dataset_means else np.nan

    compact_policy_set = set(COMPACT_POLICIES.values())
    policy_sets = {"min": compact_policy_set, "avg": compact_policy_set}
    selected = [record for record in records
                if record["aggregation"] in policy_sets
                and set(record["individual"]) == policy_sets[record["aggregation"]]
                and record["perturbation_policy"] in REGIME_LABELS]
    rows = []
    def context_key(record):
        key = [dataset_label(record["dataset"])]
        if by_classifier:
            key.append(record["mr_classifier"])
        return (*key, record["perturbation_policy"])

    contexts = sorted({context_key(record) for record in selected},
                      key=lambda context: (*context[:-1], list(REGIME_LABELS).index(context[-1])))
    for table_context in contexts:
        dataset, perturbation = table_context[0], table_context[-1]
        classifier = table_context[1] if by_classifier else None
        context = [record for record in selected
                   if dataset_label(record["dataset"]) == dataset
                   and record["perturbation_policy"] == perturbation
                   and (not by_classifier or record["mr_classifier"] == classifier)]
        by_aggregation = {aggregation: [record for record in context if record["aggregation"] == aggregation]
                          for aggregation in policy_sets}
        configurations = {
            aggregation: {tuple(str(record[field]) for field in CONFIG_FIELDS) for record in aggregation_records}
            for aggregation, aggregation_records in by_aggregation.items() if aggregation_records
        }
        if any(len(values) != 1 for values in configurations.values()):
            raise ValueError(f"Compact table has multiple {dataset}/{perturbation} configurations")
        if len({next(iter(values)) for values in configurations.values()}) > 1:
            raise ValueError(f"Compact table cannot mix {dataset}/{perturbation} configurations")
        classifier_sets = {
            aggregation: {(record["dataset"], record["mr_classifier"]) for record in aggregation_records}
            for aggregation, aggregation_records in by_aggregation.items() if aggregation_records
        }
        if len({tuple(sorted(values)) for values in classifier_sets.values()}) > 1:
            shared_classifiers = set.intersection(*classifier_sets.values())
            if not shared_classifiers:
                raise ValueError(f"Compact table has no shared {dataset}/{perturbation} classifiers")
            by_aggregation = {
                aggregation: [record for record in aggregation_records
                              if (record["dataset"], record["mr_classifier"]) in shared_classifiers]
                for aggregation, aggregation_records in by_aggregation.items()
            }

        baseline_records = by_aggregation["min"] or by_aggregation["avg"]
        row = {"Dataset": dataset}
        if by_classifier:
            row["Classifier"] = CLASSIFIER_LABELS.get(classifier, classifier)
        row["Pert."] = REGIME_LABELS[perturbation]
        for label, policy in COMPACT_POLICIES.items():
            policy_records = [record for record in baseline_records if policy in record["individual"]]
            row[label] = dataset_macro_mean(policy_records, lambda record: record["individual"][policy])
        for aggregation in policy_sets:
            row[aggregation] = dataset_macro_mean(by_aggregation[aggregation], lambda record: record["combined"])
        rows.append(row)
    columns = ["Dataset", *(["Classifier"] if by_classifier else []), "Pert.", "M", "C", "F", "min", "avg"]
    return pd.DataFrame(rows, columns=columns)


def format_value(value, digits):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame, digits=5):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(escape(format_value(value, digits)) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "—": "--", "–": "--"}
    return "".join(replacements.get(character, character) for character in str(value))


def latex_table(frame, digits=5):
    lines = [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\hline",
             " & ".join(map(latex_escape, frame.columns)) + r" \\", r"\hline"]
    lines.extend(" & ".join(latex_escape(format_value(value, digits)) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    return "\n".join([*lines, r"\hline", r"\end{tabular}"])


def latex_aggregation_comparison(frame, digits=3, compact=True):
    def format_score(value, bold=False):
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            if abs(value) < 0.001:
                formatted = r"$< 0.001$"
            else:
                formatted = f"{value:.{digits}f}"
            return rf"\textbf{{{formatted}}}" if bold else formatted
        return latex_escape(value)

    score_columns = {*COMPACT_POLICIES, "min", "avg"}
    alignment_parts = [r"@{}"] if compact else []
    for index, column in enumerate(frame.columns):
        if compact and index:
            alignment_parts.append(r"@{\hspace{3pt}}")
        alignment_parts.append("c" if column in score_columns or column == "Pert." else "l")
    if compact:
        alignment_parts.append(r"@{}")
    alignment = "".join(alignment_parts)
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(map(latex_escape, frame.columns)) + r" \\", r"\midrule"]
    dataset_groups = list(frame.groupby("Dataset", sort=False))
    for group_index, (dataset, rows) in enumerate(dataset_groups):
        row_count = len(rows)
        classifier_counts = rows["Classifier"].value_counts(sort=False) if "Classifier" in rows else {}
        seen_classifiers = set()
        table_rows = list(rows.itertuples(index=False, name=None))
        classifier_index = frame.columns.get_loc("Classifier") if "Classifier" in frame.columns else None
        for index, row in enumerate(table_rows):
            dataset_cell = (rf"\multirow{{{row_count}}}{{*}}{{{latex_escape(dataset)}}}"
                            if index == 0 else "")
            row_values = dict(zip(frame.columns, row))
            finite_scores = [row_values[column] for column in score_columns
                             if column in row_values and pd.notna(row_values[column])]
            best_score = max(finite_scores) if finite_scores else None
            values = [dataset_cell]
            for column in frame.columns[1:]:
                value = row_values[column]
                if column == "Classifier":
                    if value in seen_classifiers:
                        values.append("")
                    else:
                        values.append(rf"\multirow{{{classifier_counts[value]}}}{{*}}{{{latex_escape(value)}}}")
                        seen_classifiers.add(value)
                elif column in score_columns:
                    values.append(format_score(value, best_score is not None and value == best_score))
                else:
                    values.append(latex_escape(value))
            lines.append(" & ".join(values) + r" \\")
            if (classifier_index is not None and index < len(table_rows) - 1
                    and row[classifier_index] != table_rows[index + 1][classifier_index]):
                lines.append(rf"\cmidrule(lr){{2-{len(frame.columns)}}}")
        if group_index < len(dataset_groups) - 1:
            lines.append(r"\midrule")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "experiments/aggregated-results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/aggregated-results/tables")
    parser.add_argument("--explainer", default="shap")
    parser.add_argument("--attribution-type", default="backpropagated")
    parser.add_argument("--metric", default="f_minus_f0-mean")
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--stdout", choices=("markdown", "latex", "none"), default="markdown")
    args = parser.parse_args(argv)
    if not 0 <= args.digits <= 12:
        parser.error("--digits must be between 0 and 12")
    records, sources = load_comparisons(args.input_dir, args.explainer, args.attribution_type, args.metric)
    dataset, classifier, legends = build_tables(records)
    aggregation_comparison = build_aggregation_comparison(records)
    classifier_aggregation_comparison = build_aggregation_comparison(records, by_classifier=True)
    description = (f"Metric: {args.metric}; explainer: {args.explainer}; attribution: {args.attribution_type}. "
                   "Dataset scores are equal-weight means of classifier scores. Delta best is combined minus the best individual policy "
                   "at the displayed level. Missing/nonmember policies are shown as a dash. Groups retain their own baselines and point limits; "
                   "different groups/configurations need not use the same instances or perturbation budgets. "
                   "No significance test is implied.")
    markdown = [description]
    latex = [latex_escape(description) + "\n"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("by-dataset", dataset), ("by-dataset-classifier", classifier)]:
        md_table, tex_table = markdown_table(frame, args.digits), latex_table(frame, args.digits)
        (args.output_dir / f"{name}.md").write_text(md_table + "\n")
        (args.output_dir / f"{name}.tex").write_text(tex_table + "\n")
        markdown.append(f"## {name}\n\n{md_table}")
        latex.append(r"\subsection*{" + latex_escape(name) + "}\n" + tex_table)
    comparison_tex = latex_aggregation_comparison(aggregation_comparison)
    (args.output_dir / "aggregation-comparison.tex").write_text(comparison_tex + "\n")
    latex.append(r"\subsection*{aggregation-comparison}" + "\n" + comparison_tex)
    classifier_comparison_tex = latex_aggregation_comparison(classifier_aggregation_comparison, compact=False)
    (args.output_dir / "aggregation-comparison-by-classifier.tex").write_text(classifier_comparison_tex + "\n")
    latex.append(r"\subsection*{aggregation-comparison-by-classifier}" + "\n" + classifier_comparison_tex)
    for title, entries in legends.items():
        markdown.append(f"## {title}\n\n" + "\n".join(f"- {key}: {value}" for key, value in entries.items()))
        latex.append(r"\subsection*{" + title + "}\n" + "\n\n".join(latex_escape(f"{key}: {value}") for key, value in entries.items()))
    markdown.append("## Sources\n\n" + "\n".join(f"- {source}" for source in sources))
    (args.output_dir / "report.md").write_text("\n\n".join(markdown) + "\n")
    (args.output_dir / "report.tex").write_text("\n\n".join(latex) + "\n")
    if args.stdout != "none":
        print("\n\n".join(markdown if args.stdout == "markdown" else latex))


if __name__ == "__main__":
    main()
