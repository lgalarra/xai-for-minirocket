"""Evaluate pointwise aggregates of precomputed reference-policy explanations."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd

from plot_aggregated_explanations import aggregate_explanations


ROOT = Path(__file__).resolve().parents[1]


def parse_reference_group(value):
    policies = [policy.strip() for policy in value.split("+")]
    if not all(policies) or len(set(policies)) != len(policies):
        raise argparse.ArgumentTypeError("Use distinct, nonempty reference policies separated by '+'.")
    return policies


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def load_inputs(evaluator, args, policies):
    importer = evaluator.DataImporter(args.dataset)
    importer.data_path = f"{args.data_root}/{args.dataset}"
    frames = {}
    for policy in policies:
        frame = importer.get_metadata(args.classifier, args.explainer, args.label, args.distance, policy)
        policy_index = evaluator.get_reference_policy_indices([policy])[0][0]
        reference_column = f"reference_{policy_index}"
        if reference_column not in frame:
            raise ValueError(f"Missing reference metadata for {policy}")
        frame = frame.dropna(subset=[reference_column])
        if args.instances is not None:
            frame = frame[frame.instance_id.isin(args.instances)]
        elif args.start is not None:
            frame = frame[frame.instance_id >= args.start]
        frame = frame.drop_duplicates()
        if frame.empty or frame.duplicated(["instance_id", "channel"]).any():
            raise ValueError(f"Missing or conflicting metadata for {policy}")
        frames[policy] = frame.sort_values(["instance_id", "channel"])
    first = frames[policies[0]]
    keys = list(zip(first.instance_id, first.channel))
    instance_ids = sorted(first.instance_id.unique())
    if args.instances is not None and set(instance_ids) != set(args.instances):
        raise ValueError("Some requested instances are missing.")
    inputs = {}
    for policy, frame in frames.items():
        if list(zip(frame.instance_id, frame.channel)) != keys:
            raise ValueError(f"Instance/channel coverage differs for {policy}")
        series, labels, references, backprop, p2p, segmented, tshap = (
            evaluator.DataImporter.get_series_from_metadata(frame, [policy])
        )
        variants = {"backpropagated": backprop[policy], "p2p": p2p[policy]}
        variants.update({f"segmented_n{count}": values for count, values in segmented[policy].items()})
        variants.update({f"tshap_{key}": values for key, values in tshap[policy].items()})
        if args.data_root == "lrp-data":
            variants["lrp"] = evaluator.get_lrp_explanations_from_metadata(frame, [policy])[policy]
        for name, values in list(variants.items()):
            if not evaluator.has_explanations(values):
                del variants[name]
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != series.shape or not np.isfinite(values).all():
                raise ValueError(f"Incomplete or misaligned {name} explanations for {policy}")
            variants[name] = values
        reference = np.asarray(references[policy], dtype=float)
        if reference.shape != series.shape or not np.isfinite(reference).all() or not np.isfinite(series).all():
            raise ValueError(f"Invalid series/reference for {policy}")
        inputs[policy] = {"series": series, "labels": labels, "reference": reference, "variants": variants}
        if not np.array_equal(series, inputs[policies[0]]["series"]) or not np.array_equal(labels, inputs[policies[0]]["labels"]):
            raise ValueError(f"Series or target labels differ for {policy}")
    return instance_ids, inputs


def combine_variants(inputs, policies, aggregation):
    names = set(inputs[policies[0]]["variants"])
    for policy in policies[1:]:
        names &= set(inputs[policy]["variants"])
    if not names:
        raise ValueError("No explanation type is available for every constituent policy.")
    return {
        name: aggregate_explanations([inputs[policy]["variants"][name] for policy in policies], aggregation)
        for name in sorted(names)
    }


def count_basis(variants):
    counts = {name: np.count_nonzero(values, axis=tuple(range(1, values.ndim))) for name, values in variants.items()}
    if "p2p" in counts and counts["p2p"].sum() > 0:
        return counts["p2p"]
    positive = [values for values in counts.values() if values.sum() > 0]
    return min(positive, key=lambda values: values.sum()) if positive else next(iter(counts.values()))


def reference_targets(inputs, policies, strategy):
    references = [inputs[policy]["reference"] for policy in policies]
    if strategy == "mean-reference":
        return [("mean-reference", np.mean(references, axis=0))]
    if strategy == "first":
        return [(policies[0], references[0])]
    return list(zip(policies, references))


def evaluate(evaluator, classifier, inputs, instance_ids, args):
    policies = args.reference_policy
    group = "+".join(policies)
    variants = combine_variants(inputs, policies, args.aggregation)
    configurations = {group: (policies, variants)}
    if args.compare_individual:
        configurations.update({policy: ([policy], {name: inputs[policy]["variants"][name] for name in variants}) for policy in policies})
    counts = {name: count_basis(values) for name, (_, values) in configurations.items()}
    if args.compare_individual:
        common_counts = np.min(list(counts.values()), axis=0)
        counts = {name: common_counts for name in counts}
    first = inputs[policies[0]]
    series = first["series"]
    summary, per_instance, raw = [], [], []
    perturbations = ["gaussian", "instance_to_reference"] if args.perturbation == "both" else [args.perturbation]
    for perturbation in perturbations:
        budget = args.budget if perturbation == "gaussian" else 1
        for group_name, (group_policies, group_variants) in configurations.items():
            targets = reference_targets(inputs, group_policies, args.reference_strategy)
            parameters = {"percentile_cut": args.percentile_cut, "budget": budget}
            parameters.update({"sigma": args.sigma} if perturbation == "gaussian" else {"interpolation": args.interpolation})
            call_parameters = dict(parameters, n_perturbed_points=evaluator.get_n_perturbed_points(counts[group_name], args.percentile_cut))
            for name, explanations in group_variants.items():
                metrics, normalized, changes = [], [], []
                for target_name, reference in targets:
                    random_state = np.random.get_state()
                    try:
                        np.random.seed(args.seed)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            metric, norm, change, perturbed_count = evaluator.compute_perturbation_metrics(
                                classifier, series, reference, explanations, args.explainer,
                                perturbation, call_parameters, budget,
                            )
                    finally:
                        np.random.set_state(random_state)
                    metric = np.asarray(metric).reshape(len(instance_ids), budget)
                    metrics.append(metric)
                    normalized.append(np.asarray(norm).reshape(len(instance_ids), budget))
                    changes.append(change)
                    for instance_index, instance_id in enumerate(instance_ids):
                        for repeat in range(budget):
                            raw.append({"reference_policy": group_name, "reference_target": target_name,
                                        "perturbation_policy": perturbation, "attribution_type": name,
                                        "instance_id": instance_id, "repeat": repeat,
                                        "f_minus_f0": metric[instance_index, repeat],
                                        "perturbed_points_total": perturbed_count})
                metric = np.mean(metrics, axis=0)
                norm = np.mean(normalized, axis=0)
                row = {"dataset": args.dataset, "mr_classifier": args.classifier,
                       "base_explainer": args.explainer, "label": args.label, "distance": args.distance,
                       "reference_policy": group_name, "aggregation": args.aggregation,
                       "reference_strategy": args.reference_strategy, "seed": args.seed,
                       "perturbation_policy": perturbation, "attribution_type": name,
                       "args": str(parameters), "instance_ids": ";".join(map(str, instance_ids)),
                       "count_basis": "shared_minimum" if args.compare_individual else "per_group",
                       "n_perturbed_points": ";".join(map(str, call_parameters["n_perturbed_points"]))}
                metric_columns = {}
                evaluator.add_metric_columns(metric_columns, "")
                evaluator.append_metrics(metric_columns, "", metric.ravel(), norm.ravel(), np.mean(changes))
                row.update({key: values[0] for key, values in metric_columns.items()})
                summary.append(row)
                for instance_id, value in zip(instance_ids, metric.mean(axis=1)):
                    per_instance.append({"reference_policy": group_name, "perturbation_policy": perturbation,
                                         "attribution_type": name, "instance_id": instance_id, "f_minus_f0": value})
    return pd.DataFrame(summary), pd.DataFrame(per_instance), pd.DataFrame(raw)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("label", choices=("predicted", "training"))
    parser.add_argument("explainer")
    parser.add_argument("reference_policy", type=parse_reference_group)
    parser.add_argument("perturbation", choices=("gaussian", "instance_to_reference", "both"))
    parser.add_argument("classifier")
    parser.add_argument("distance", nargs="?", default="euclidean")
    parser.add_argument("data_root", nargs="?", choices=("data", "lrp-data"), default="data")
    parser.add_argument("--aggregation", choices=("max", "min", "avg", "sum"), default="max")
    parser.add_argument("--reference-strategy", choices=("average-scores", "mean-reference", "first"), default="average-scores")
    parser.add_argument("--compare-individual", action=argparse.BooleanOptionalAction, default=True,
                        help="Compare against each constituent policy (default: enabled)")
    parser.add_argument("--instances", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--start", type=int, help="First instance ID (inclusive)")
    parser.add_argument("--end", type=int, help="Last instance ID (inclusive); defaults to all remaining exported test instances; requires --start")
    parser.add_argument("--percentile-cut", type=float, default=90)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--interpolation", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output", type=Path, help="Override the automatically named CSV in experiments/aggregated-results/")
    args = parser.parse_args(argv)
    if args.start is not None or args.end is not None:
        if args.instances is not None:
            parser.error("Use either --instances or --start/--end, not both.")
        if args.start is None:
            parser.error("--end requires --start.")
        if args.start < 0 or (args.end is not None and args.end < args.start):
            parser.error("Instance bounds must satisfy 0 <= start <= end.")
        if args.end is not None:
            args.instances = list(range(args.start, args.end + 1))
    if not 0 <= args.percentile_cut <= 100 or not 0 <= args.interpolation <= 1:
        parser.error("Percentile cut must be in [0, 100] and interpolation in [0, 1].")
    if not np.isfinite(args.sigma) or args.sigma < 0 or args.budget < 1:
        parser.error("Sigma must be finite and nonnegative; budget must be positive.")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("Seed must be in [0, 2**32).")
    if args.output is None:
        components = [args.dataset, args.label, args.explainer, "+".join(args.reference_policy),
                      args.classifier, args.aggregation, args.perturbation]
        filename = "__".join(re.sub(r"[^A-Za-z0-9_.+-]", "_", component) for component in components)
        args.output = ROOT / "experiments/aggregated-results" / f"{filename}.csv"
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    sys.path.insert(0, str(ROOT / "code"))
    tshap_path = Path(os.environ.get("TSHAP_REPO_PATH", ROOT.parent / "tshap"))
    if tshap_path.exists():
        sys.path.append(str(tshap_path))
    with working_directory(args.export_root.resolve()):
        import evaluate_explanations as evaluator

        evaluator.get_reference_policy_indices(args.reference_policy)
        if not evaluator.supports_explainer_for_classifier(args.explainer, args.classifier):
            raise ValueError("Explainer is unsupported for this classifier.")
        instance_ids, inputs = load_inputs(evaluator, args, args.reference_policy)
        with open(f"data/{args.dataset}/{args.classifier}.pkl", "rb") as handle:
            classifier = pickle.load(handle)
        summary, per_instance, raw = evaluate(evaluator, classifier, inputs, instance_ids, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    per_instance.to_csv(output.with_name(output.stem + "-instances.csv"), index=False)
    raw.to_csv(output.with_name(output.stem + "-samples.csv"), index=False)
    print(summary[["reference_policy", "perturbation_policy", "attribution_type", "f_minus_f0-mean"]].to_string(index=False))
    print(f"Saved summary, per-instance metrics, and per-target samples to {output.parent}")


if __name__ == "__main__":
    main()
