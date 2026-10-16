"""Stage data/ metadata from matching lrp-data/ templates, without copying explanations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["opposite_class_closest_instance", "opposite_class_medoid", "opposite_class_centroid",
            "global_medoid", "global_centroid", "opposite_class_farthest_instance",
            "counterfactual", "counterfactual_dtw"]


def map_path(value):
    path = str(value).split("/experiments/", 1)[-1]
    if not path.startswith("lrp-data/") or ".." in Path(path).parts:
        raise ValueError(f"Unexpected template path: {value}")
    return "data/" + path[len("lrp-data/"):]


def stage_frame(template, base, configuration, policy_index):
    frame = template.drop(columns=[column for column in template if column.startswith("beta_lrp_")]).copy()
    if frame.empty or frame.duplicated(["instance_id", "channel"]).any():
        raise ValueError("Template is empty or has duplicate instance/channel keys")
    required = ["series", f"reference_{policy_index}", f"beta_{policy_index}_attributions"]
    if frame[required].isna().any().any():
        raise ValueError("Template has incomplete required paths")
    for column in frame:
        if column == "series" or re.fullmatch(r"reference_\d+", column) or column.endswith("_attributions"):
            populated = frame[column].notna()
            frame.loc[populated, column] = frame.loc[populated, column].map(map_path)
            for value in frame.loc[populated, column]:
                expected = f"data/{configuration.split('/')[0]}/" if column == "series" else f"data/{configuration}/"
                if not value.startswith(expected):
                    raise ValueError(f"Path belongs to another configuration: {value}")
    optional_missing = []

    @lru_cache(maxsize=None)
    def directory_files(directory):
        return {path.name for path in (base / directory).iterdir()}

    for row_index, row in frame.iterrows():
        expected_beta = f"beta_instance_{int(row.instance_id)}_{policy_index}.csv"
        if Path(row[f"beta_{policy_index}_attributions"]).name != expected_beta:
            raise ValueError(f"Unexpected backpropagated attribution filename for instance {row.instance_id}")
        for column in frame:
            if not (column == "series" or re.fullmatch(r"reference_\d+", column) or column.endswith("_attributions")) or pd.isna(row[column]):
                continue
            path = Path(row[column])
            if path.parent.name != str(row.channel):
                raise ValueError(f"Channel mismatch: {path}")
            if path.name not in directory_files(str(path.parent)):
                if column in required:
                    raise FileNotFoundError(base / path)
                optional_missing.append(str(path))
                frame.loc[row_index, column] = np.nan
    return frame, sorted(set(optional_missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--dataset", default="cognitive-circles")
    parser.add_argument("--classifier", default="LogisticRegression")
    parser.add_argument("--explainer", default="shap")
    parser.add_argument("--policies", default="opposite_class_closest_instance+opposite_class_medoid+opposite_class_farthest_instance")
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True, help="Local copy of the source checkout's saved classifier")
    parser.add_argument("--sample-size", type=int, default=7)
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    policies = args.policies.split("+")
    if len(set(policies)) != len(policies) or any(policy not in POLICIES for policy in policies):
        parser.error("Policies must be distinct known policy names")
    base = args.experiments_root
    configuration = f"{args.dataset}/{args.classifier}/{args.explainer}/predicted/euclidean"
    staged = {}
    audit = {"configuration": configuration, "verification": "sampled contents and predictions; complete path coverage",
             "model_sha256": hashlib.sha256(args.model_file.read_bytes()).hexdigest(), "policies": {}, "installed": False}
    for policy in policies:
        filename = f"metadata_ref_policy_{policy}.csv"
        source = base / "lrp-data" / configuration / filename
        target = base / "data" / configuration / filename
        if target.exists() and not pd.read_csv(target).empty:
            raise ValueError(f"Refusing to reconstruct populated metadata: {target}")
        template = pd.read_csv(source)
        frame, missing = stage_frame(template, base, configuration, POLICIES.index(policy))
        staged[policy] = (template, frame)
        audit["policies"][policy] = {"rows": len(frame), "instances": int(frame.instance_id.nunique()),
                                     "source": str(source), "target": str(target), "optional_missing": missing,
                                     "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    first = staged[policies[0]][1]
    keys = set(zip(first.instance_id, first.channel))
    for _, frame in staged.values():
        if set(zip(frame.instance_id, frame.channel)) != keys:
            raise ValueError("Policy coverage differs")
    ids = sorted(first.instance_id.unique())
    sample_ids = [int(ids[position]) for position in np.unique(np.linspace(0, len(ids) - 1, min(args.sample_size, len(ids)), dtype=int))]
    audit["sample_ids"] = sample_ids
    print(f"Validated paths for {len(ids)} instances; checking contents and predictions for {sample_ids}", flush=True)

    @lru_cache(maxsize=None)
    def read_series(path):
        frame = pd.read_csv(base / path, header=None, index_col=0)
        if frame.shape[1] != 1 or frame.empty or frame.index.has_duplicates:
            raise ValueError(f"Invalid series: {path}")
        values = frame.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite series: {path}")
        return frame.index.to_numpy(), values

    required_paths = set()
    for policy, (template, frame) in staged.items():
        index = POLICIES.index(policy)
        for _, row in frame[frame.instance_id.isin(sample_ids)].iterrows():
            for column in ["series", f"reference_{index}", f"beta_{index}_attributions"]:
                required_paths.add(row[column])
                if column != f"beta_{index}_attributions":
                    required_paths.add(row[column].replace("data/", "lrp-data/", 1))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(read_series, sorted(required_paths)))
    sys.path.insert(0, str(ROOT / "code"))
    with args.model_file.open("rb") as handle:
        classifier = pickle.load(handle)
    classes = np.asarray(classifier.classifier.classes_)
    checks = []
    for policy, (template, frame) in staged.items():
        index = POLICIES.index(policy)
        instances, references, labels, reference_labels, probabilities, reference_probabilities = [], [], [], [], [], []
        for instance_id in sample_ids:
            rows = frame[frame.instance_id == instance_id]
            if rows.label.nunique() != 1 or rows.label_probability.nunique() != 1:
                raise ValueError(f"Inconsistent target across channels: {instance_id}")
            channels, reference_channels = [], []
            for _, row in rows.iterrows():
                time, series = read_series(row.series)
                reference_time, reference = read_series(row[f"reference_{index}"])
                beta_time, beta = read_series(row[f"beta_{index}_attributions"])
                if not np.array_equal(time, reference_time) or not np.array_equal(time, beta_time):
                    raise ValueError(f"Series/reference/attribution time indices differ: {instance_id}, {row.channel}")
                for column in ["series", f"reference_{index}"]:
                    original = read_series(row[column].replace("data/", "lrp-data/", 1))
                    current = read_series(row[column])
                    if not all(np.array_equal(left, right) for left, right in zip(original, current)):
                        raise ValueError(f"Template series/reference content differs: {row[column]}")
                channels.append(series)
                reference_channels.append(reference)
            instances.append(channels)
            references.append(reference_channels)
            labels.append(rows.label.iloc[0])
            reference_labels.append(rows[f"reference_{index}_label"].iloc[0])
            probabilities.append(rows.label_probability.iloc[0])
            reference_probabilities.append(rows[f"reference_{index}_label_probability"].iloc[0])
        batch = np.asarray(instances + references, dtype=np.float32)
        predictions = classifier.predict_proba(batch)
        count = len(sample_ids)
        target_columns = np.array([np.flatnonzero(classes == label)[0] for label in labels])
        instance_probs = predictions[np.arange(count), target_columns]
        reference_probs = predictions[np.arange(count) + count, target_columns]
        matches = (np.array_equal(classes[predictions[:count].argmax(axis=1)], labels)
                   and np.array_equal(classes[predictions[count:].argmax(axis=1)], reference_labels)
                   and np.allclose(instance_probs, probabilities, atol=1e-6, rtol=1e-5)
                   and np.allclose(reference_probs, reference_probabilities, atol=1e-6, rtol=1e-5))
        checks.append({"policy": policy, "prediction_metadata_matches": bool(matches),
                       "instance_probability_max_difference": float(np.max(np.abs(instance_probs - probabilities))),
                       "reference_probability_max_difference": float(np.max(np.abs(reference_probs - reference_probabilities)))})
        print(checks[-1], flush=True)
    audit["prediction_checks"] = checks
    audit["passed"] = all(check["prediction_metadata_matches"] for check in checks)
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    (args.staging_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise ValueError("Prediction metadata does not match the saved classifier; see audit.json. No metadata CSVs written.")
    for policy, (_, frame) in staged.items():
        frame.to_csv(args.staging_dir / f"metadata_ref_policy_{policy}.csv", index=False)
    print(f"Staged reconstructed metadata in {args.staging_dir}; remote files unchanged.", flush=True)


if __name__ == "__main__":
    main()
