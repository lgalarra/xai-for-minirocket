import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "official-results"
DEFAULT_OUT_DIR = SCRIPT_DIR / "official-perturbation-figures"
GENERATOR = SCRIPT_DIR / "generate_official_perturbation_charts.py"
DEFAULT_PERTURBATION_POLICIES = (
    "gaussian",
    "instance_to_reference",
    "reference_to_instance",
    "reference_to_instance_bottom",
    "reference_to_instance_random",
    "reference_to_instance_random_no_positive",
)
GRADIENT_PERTURBATION_POLICY_REPLACEMENTS = {
    "gaussian": "gradient_gaussian",
    "gaussian_bottom": "gradient_gaussian_bottom",
    "gaussian_random": "gradient_gaussian_random",
    "gaussian_random_no_positive": "gradient_gaussian_random_no_positive",
}
DEFAULT_EVOLUTION_FACTORS = ("percentile_cut",)
DEFAULT_METRIC_KINDS = ("probability",)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run generate_official_perturbation_charts.py for every perturbation "
            "policy and base explainer found in the official result CSVs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory with perturbation-results-* CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory where figures are written. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--evolution-factor",
        action="append",
        choices=("percentile_cut", "interpolation", "sigma", "perturbation_policy"),
        dest="evolution_factors",
        help=(
            "Evolution factor to generate. May be repeated. "
            f"Default: {', '.join(DEFAULT_EVOLUTION_FACTORS)}"
        ),
    )
    parser.add_argument(
        "--metric-kind",
        action="append",
        choices=("probability", "probability-norm", "change-ratio"),
        dest="metric_kinds",
        help=(
            "Metric kind to generate. May be repeated. "
            f"Default: {', '.join(DEFAULT_METRIC_KINDS)}"
        ),
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help=(
            "Perturbation policy to generate. May be repeated. "
            f"Default: {', '.join(DEFAULT_PERTURBATION_POLICIES)}"
        ),
    )
    parser.add_argument(
        "--explainer",
        action="append",
        dest="explainers",
        help="Base explainer to generate. May be repeated. Default: discover all.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="mr_classifier filter passed through to the generator. Default: all.",
    )
    parser.add_argument(
        "--reference-policy",
        default="all",
        help="reference_policy filter passed through to the generator. Default: all.",
    )
    parser.add_argument(
        "--best-method-random-regime",
        choices=("random", "random_no_positive"),
        default="random_no_positive",
        help=(
            "Random perturbation regime shown in best-method charts. "
            "Use 'random' for *_random policies or 'random_no_positive' for "
            "*_random_no_positive policies. Default: random_no_positive."
        ),
    )
    parser.add_argument(
        "--percentile-cut",
        type=float,
        default=90.0,
        help="percentile_cut filter passed through when it is not the evolution factor.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="sigma filter passed through when it is not the evolution factor.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generator commands without running them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failed combinations and report failures at the end.",
    )
    return parser.parse_args()


def discover_values(data_dir):
    files = sorted(data_dir.glob("perturbation-results-*"))
    if not files:
        raise FileNotFoundError(f"No perturbation-results-* files found in {data_dir}")

    policies = set()
    explainers = set()
    for csv_file in files:
        df = pd.read_csv(
            csv_file,
            usecols=lambda col: col in {"perturbation_policy", "base_explainer"},
            low_memory=False,
        )
        policies.update(df["perturbation_policy"].dropna().astype(str).unique())
        explainers.update(df["base_explainer"].dropna().astype(str).unique())

    return sorted(policies), sorted(explainers), len(files)


def expand_requested_policies(requested_policies, discovered_policies):
    discovered_policies = sorted(discovered_policies)
    expanded = []
    for policy in requested_policies:
        if policy.endswith("_"):
            matches = [
                discovered_policy
                for discovered_policy in discovered_policies
                if discovered_policy.startswith(policy)
            ]
            expanded.extend(matches)
        else:
            expanded.append(policy)
    return sorted(dict.fromkeys(expanded))


def build_command(args, policy, explainer, evolution_factor, metric_kind):
    policy = policy_for_explainer(policy, explainer)
    return [
        sys.executable,
        str(GENERATOR),
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(args.out_dir),
        "--evolution-factor",
        evolution_factor,
        "--metric-kind",
        metric_kind,
        "--perturbation-policy",
        policy,
        "--base-explainer",
        explainer,
        "--model",
        args.model,
        "--reference-policy",
        args.reference_policy,
        "--best-method-random-regime",
        args.best_method_random_regime,
        "--percentile-cut",
        str(args.percentile_cut),
        "--sigma",
        str(args.sigma),
    ]


def policy_for_explainer(policy, explainer):
    if explainer == "gradients":
        return GRADIENT_PERTURBATION_POLICY_REPLACEMENTS.get(policy, policy)
    return policy


def expand_policy_explainer_combinations(policies, explainers):
    combinations = []
    for policy, explainer in itertools.product(policies, explainers):
        combinations.append((policy_for_explainer(policy, explainer), explainer))
    return sorted(dict.fromkeys(combinations))


def main():
    args = parse_args()
    discovered_policies, discovered_explainers, file_count = discover_values(args.data_dir)

    requested_policies = args.policies or list(DEFAULT_PERTURBATION_POLICIES)
    policies = expand_requested_policies(requested_policies, discovered_policies)
    explainers = sorted(args.explainers) if args.explainers else discovered_explainers
    policy_explainer_pairs = expand_policy_explainer_combinations(policies, explainers)
    resolved_policies = sorted({policy for policy, _ in policy_explainer_pairs})
    missing_policies = sorted(set(resolved_policies) - set(discovered_policies))
    if missing_policies:
        raise ValueError(
            "Requested policies are not present in the result files: "
            + ", ".join(missing_policies)
        )
    evolution_factors = args.evolution_factors or list(DEFAULT_EVOLUTION_FACTORS)
    metric_kinds = args.metric_kinds or list(DEFAULT_METRIC_KINDS)

    combinations = list(
        (policy, explainer, evolution_factor, metric_kind)
        for (policy, explainer), evolution_factor, metric_kind
        in itertools.product(policy_explainer_pairs, evolution_factors, metric_kinds)
    )

    print(f"Discovered {file_count} result files in {args.data_dir}")
    print(f"Policies: {', '.join(resolved_policies)}")
    print(f"Explainers: {', '.join(explainers)}")
    print(f"Evolution factors: {', '.join(evolution_factors)}")
    print(f"Metric kinds: {', '.join(metric_kinds)}")
    print(f"Best-method random regime: {args.best_method_random_regime}")
    print(f"Running {len(combinations)} chart-generation commands")

    failures = []
    for index, (policy, explainer, evolution_factor, metric_kind) in enumerate(
        combinations,
        start=1,
    ):
        command = build_command(args, policy, explainer, evolution_factor, metric_kind)
        label = (
            f"[{index}/{len(combinations)}] policy={policy} "
            f"explainer={explainer} evolution={evolution_factor} metric={metric_kind}"
        )
        print(label, flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue

        result = subprocess.run(command, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            failures.append((label, result.returncode))
            if not args.keep_going:
                raise SystemExit(result.returncode)

    if failures:
        print("Failed combinations:")
        for label, returncode in failures:
            print(f"  returncode={returncode} {label}")
        raise SystemExit(1)

    print(f"Done. Wrote figures under {args.out_dir}")


if __name__ == "__main__":
    main()
