#!/usr/bin/env python3
"""Post-process saved heartbeat explanations; never load or execute a model.

Example:
python3 explain_heartbeat.py --instance-id 22 \
  --reference-policy opposite_class_closest_instance --selected-count 16 \
  --output-dir reports/heartbeat_22
Requires numpy and matplotlib. Paths are relative to --repo-root, except output-dir.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/heartbeat-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap

DATASET_URL = "https://www.timeseriesclassification.com/description.php?Dataset=AbnormalHeartbeat"
NAMES = {0: "Other", 1: "Extrastole"}


def load_series(path):
    """Writer: DataFrame(...).T.to_csv(header=False), with pandas index retained."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError(f"Expected finite, headerless index/value columns: {path}")
    if not np.array_equal(a[:, 0], np.arange(len(a))):
        raise ValueError(f"Indices must be ordered, unique and consecutive from zero: {path}")
    return a[:, 1]


def source_record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def regions_from_values(x, b, quantile=0.):
    # Sign runs preserve raw attribution signs; neither smoothing nor gap merging.
    threshold = float(np.quantile(np.abs(b), quantile))
    region_sign = np.where(np.abs(b) >= threshold, np.sign(b), 0)
    edges = np.r_[0, np.flatnonzero(np.diff(region_sign)) + 1, len(b)]
    regions = []
    total = float(np.abs(b).sum())
    for lo, stop in zip(edges[:-1], edges[1:]):
        if region_sign[lo] == 0:
            continue
        signal = x[lo:stop]
        delta = np.diff(signal)
        variation = float(np.abs(delta).sum())
        trend_ratio = float((signal[-1] - signal[0]) / variation) if variation else 0.
        description = ("a rising segment" if np.all(delta >= 0) and np.any(delta > 0)
                       else "a falling segment" if np.all(delta <= 0) and np.any(delta < 0)
                       else "a flat segment" if np.all(delta == 0)
                       else "a mostly rising segment" if trend_ratio >= .6
                       else "a mostly falling segment" if trend_ratio <= -.6
                       else "a segment with rises and falls")
        mass = float(np.abs(b[lo:stop]).sum())
        regions.append({
            "start_index": int(lo), "end_index": int(stop - 1),
            "contribution": "positive" if b[lo] > 0 else "negative",
            "signed_attribution": float(b[lo:stop].sum()), "magnitude": mass,
            "relative_importance": mass / total if total else 0.,
            "relative_importance_definition": "fraction of all absolute saved observation attributions",
            "position": ("spans both halves of the recording" if lo < len(x) / 2 <= stop - 1
                         else "first half of the recording" if stop - 1 < len(x) / 2
                         else "second half of the recording"),
            "visual_description": description,
            "net_change_over_total_variation": trend_ratio,
            "signal_minimum": float(signal.min()), "signal_maximum": float(signal.max()),
            "minimum_index": int(lo + signal.argmin()), "maximum_index": int(lo + signal.argmax()),
            "signal_start": float(signal[0]), "signal_end": float(signal[-1]),
        })
    selected = []
    for direction, count in [("positive", 3), ("negative", 2)]:
        ranked = sorted((r for r in regions if r["contribution"] == direction),
                        key=lambda r: (-r["magnitude"], r["start_index"]))
        selected.extend(ranked[:count])
    for i, region in enumerate(selected, 1):
        region["region_id"] = i
    return selected


def narrative_from_json(d):
    pred = d["prediction"]
    name = pred["class_name"]
    target = d["attribution"]["target_class_name"] or "the saved explanation target (class unavailable)"
    prediction = (f'The model classified this recording as “{name}”'
                  if name else "The saved files do not identify the model's predicted class")
    if name == "Other":
        prediction += ", the combined group outside the extrasystole category."
    else:
        prediction += "."
    selected = d["influential_regions"]
    positive_count = 3 if d["attribution"]["negative_sum"] == 0 else 2
    positive = [r for r in selected if r["contribution"] == "positive"][:positive_count]
    negative = [r for r in selected if r["contribution"] == "negative"][:1]
    claims = []
    for r in positive + negative:
        action = "supports" if r["contribution"] == "positive" else "opposes"
        claims.append(f'Samples {r["start_index"]:,}–{r["end_index"]:,} contain '
                      f'{r["visual_description"]} that {action} “{target}”.')
    share = sum(r["relative_importance"] for r in selected)
    subset_count = d["attribution"]["selected_feature_count"]
    subset = (f"These colors show only {subset_count} selected patterns used by the model, "
              "so they explain part of its decision." if subset_count is not None else
              "The colors describe the saved explanation relative to its comparison recording.")
    if d["attribution"]["negative_sum"] == 0:
        claims.append(f'No saved contribution opposes “{target}”.')
    region_word = "region contains" if len(selected) == 1 else "regions together contain"
    return ("Understanding the heartbeat classification\n\n"
            f"Prediction: {prediction}\n\n"
            f"How to read the figure: Red supports “{target}”; blue opposes it. "
            "Stronger color means a larger contribution, while pale color means little contribution.\n\n"
            "What influenced the prediction: " + " ".join(claims) +
            f" The {len(selected)} highlighted {region_word} {share:.1%} of the total contribution "
            "magnitude.\n\n"
            f"What this means: {subset} “Other” includes several categories and does not mean healthy.\n\n"
            "Important limitation: This explains an automated classification, not a medical diagnosis.\n")


def draw_colored(ax, indices, signal, values, cmap, norm):
    # Split each existing straight line at its midpoint: each observation owns
    # the adjacent half-segments. No attribution averaging or signal smoothing.
    points = np.column_stack([indices, signal])
    midpoints = (points[:-1] + points[1:]) / 2
    segments = np.empty((2 * (len(points) - 1), 2, 2))
    segments[0::2] = np.stack([points[:-1], midpoints], axis=1)
    segments[1::2] = np.stack([midpoints, points[1:]], axis=1)
    colors = np.empty(len(segments))
    colors[0::2], colors[1::2] = values[:-1], values[1:]
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.3)
    collection.set_array(colors)
    ax.add_collection(collection)
    ax.update_datalim(points)
    ax.autoscale_view()


def make_figures(d, out):
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "path.simplify": False})
    x = np.array(d["signal"]["values"])
    b = np.array(d["attribution"]["values"])
    indices = np.arange(len(x))
    limit = d["visualization"]["symmetric_color_limit"] or 1.
    norm = Normalize(-limit, limit)
    cmap = LinearSegmentedColormap.from_list("contribution", ["#2166ac", "#eeeeee", "#b2182b"])
    regions = d["influential_regions"]
    positives = [r for r in regions if r["contribution"] == "positive"]
    negatives = [r for r in regions if r["contribution"] == "negative"]
    zooms = (positives[:2] + negatives[:1] + positives[2:] + negatives[1:])[:3]
    target = d["attribution"]["target_class_name"] or "unknown target"
    for colored, stem in [(False, "heartbeat_heatmap"), (True, "heartbeat_colored_signal")]:
        fig = plt.figure(figsize=(13, 8.4), layout="constrained")
        grid = fig.add_gridspec(4, 3, height_ratios=[3.2, .45, 2.2, .23])
        ax = fig.add_subplot(grid[0, :])
        if colored:
            draw_colored(ax, indices, x, b, cmap, norm)
        else:
            ax.plot(indices, x, color="#303640", linewidth=.65)
        ax.set(xlim=(-.5, len(x) - .5), ylabel="Recorded value (units unspecified)",
               xlabel="Observation index (sample number)")
        for r in regions:
            center = (r["start_index"] + r["end_index"]) / 2
            ax.axvspan(r["start_index"] - .5, r["end_index"] + .5,
                       facecolor="none", edgecolor="#777777", linewidth=.65)
            ax.text(center, 1.02 + .055 * (r["region_id"] % 2), f'R{r["region_id"]}',
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        strip = fig.add_subplot(grid[1, :], sharex=ax)
        if not colored:
            strip.pcolormesh(np.arange(len(b) + 1) - .5, [0, 1], b[None, :],
                             cmap=cmap, norm=norm, shading="flat", rasterized=False)
            strip.set_yticks([])
            strip.set_ylabel("Influence", rotation=0, ha="right", va="center")
            strip.tick_params(axis="x", labelbottom=False)
        else:
            strip.axis("off")
            strip.text(.5, .5, "The curve keeps every saved sample; close-ups below reveal short regions.",
                       transform=strip.transAxes, ha="center")
        for col, r in enumerate(zooms):
            z = fig.add_subplot(grid[2, col])
            lo, hi = max(0, r["start_index"] - 15), min(len(x), r["end_index"] + 16)
            if colored:
                draw_colored(z, indices[lo:hi], x[lo:hi], b[lo:hi], cmap, norm)
            else:
                z.plot(indices[lo:hi], x[lo:hi], color="#303640", linewidth=1.1)
                ymin, ymax = z.get_ylim()
                z.pcolormesh(np.arange(lo, hi + 1) - .5, [ymin, ymax], b[None, lo:hi],
                             cmap=cmap, norm=norm, shading="flat", zorder=-1)
            z.axvline(r["start_index"] - .5, color="#555555", linestyle="--", linewidth=.7)
            z.axvline(r["end_index"] + .5, color="#555555", linestyle="--", linewidth=.7)
            action = "supports" if r["contribution"] == "positive" else "opposes"
            z.set(title=f'R{r["region_id"]}: {r["start_index"]:,}–{r["end_index"]:,} · {action} {target}',
                  xlabel="Observation index", ylabel="Recorded value", xlim=(lo, hi - 1))
            z.tick_params(labelsize=9)
        cax = fig.add_subplot(grid[3, :])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([-limit, 0, limit], labels=[f'Blue: opposes {target}', 'Pale: little / no contribution',
                                               f'Red: supports {target}'])
        cb.set_label("Color strength shows contribution size, not recording loudness or medical severity")
        prediction = d["prediction"]["class_name"] or "unavailable"
        count = d["attribution"]["selected_feature_count"]
        scope = f"{count} selected model features" if count is not None else "saved feature explanation"
        fig.suptitle(f'Heartbeat recording {d["instance_id"]} · Model prediction: {prediction}\n'
                     f'{scope} · “Other” does not mean healthy · Not a medical diagnosis', fontsize=14)
        for ext in ["png", "svg"]:
            fig.savefig(out / f"{stem}.{ext}", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", "--INSTANCE_ID", required=True, type=int)
    parser.add_argument("--reference-policy", "--REFERENCE_POLICY", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--selected-count", type=int, help="Select the explicit changeclass-top-N export")
    parser.add_argument("--region-quantile", type=float, default=0.,
                        help="Minimum absolute-attribution quantile for region selection (0 to <1); figures retain all values")
    args = parser.parse_args()
    if not 0 <= args.region_quantile < 1:
        parser.error("--region-quantile must be >=0 and <1")
    if args.instance_id < 0 or Path(args.reference_policy).name != args.reference_policy:
        parser.error("Invalid instance ID or reference policy")
    root = args.repo_root.resolve()
    base = root / "output/abnormal-heartbeat-c1/LogisticRegression/shap/predicted" / str(args.instance_id)
    stem = f"betas_backpropagated_explanations_ref_policy_{args.reference_policy}_instance_{args.instance_id}"
    if args.selected_count is not None:
        stem += f"-changeclass-top-{args.selected_count}"
    instance_path = base / f"instance_{args.instance_id}.csv"
    beta_path = base / f"{stem}.csv"
    metadata_path = base / f"metadata_ref_policy_{args.reference_policy}_instance_{args.instance_id}.json"
    inputs = [instance_path, beta_path]
    x, b = load_series(instance_path), load_series(beta_path)
    if len(x) != len(b):
        raise ValueError("Signal and attributions have different lengths")
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if meta:
        inputs.append(metadata_path)
    target = meta.get("instance_prediction")
    if target not in (None, 0, 1):
        raise ValueError("Unexpected binary class ID")
    if meta.get("instance_label", target) != target:
        raise ValueError("Metadata label disagrees with predicted explanation target")
    probability = meta.get("instance_predicted_probability")
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("Invalid saved probability")
    class1_probability = None if probability is None or target is None else (
        probability if target == 1 else 1. - probability)
    selected_ids, audit = None, {}
    if args.selected_count is not None:
        selected_path = base / f"selected_features_{args.instance_id}-top{args.selected_count}.csv"
        ids = load_series(selected_path)
        if len(ids) != args.selected_count or not np.all(ids == ids.astype(int)) or len(set(ids)) != len(ids):
            raise ValueError("Invalid selected-feature list")
        selected_ids = ids.astype(int).tolist()
        inputs.append(selected_path)
        alpha_path = base / f"alphas_mr_explanations_ref_policy_{args.reference_policy}_instance_{args.instance_id}.csv"
        if alpha_path.exists():
            alphas = load_series(alpha_path)
            if min(selected_ids) < 0 or max(selected_ids) >= len(alphas):
                raise ValueError("Selected features outside saved feature vector")
            inputs.append(alpha_path)
            selected_sum = float(alphas[selected_ids].sum())
            audit.update(feature_count=len(alphas), full_feature_sum=float(alphas.sum()),
                         selected_feature_sum=selected_sum,
                         beta_minus_selected_sum=float(b.sum() - selected_sum),
                         beta_matches_selected_sum=bool(np.isclose(b.sum(), selected_sum, atol=1e-10, rtol=1e-7)))
            if not audit["beta_matches_selected_sum"]:
                raise ValueError("Saved observation sum disagrees with selected feature sum")
    else:
        alpha_path = base / f"alphas_mr_explanations_ref_policy_{args.reference_policy}_instance_{args.instance_id}.csv"
        if alpha_path.exists():
            alphas = load_series(alpha_path)
            inputs.append(alpha_path)
            audit.update(feature_count=len(alphas), full_feature_sum=float(alphas.sum()),
                         beta_minus_full_feature_sum=float(b.sum() - alphas.sum()),
                         beta_matches_full_feature_sum=bool(np.isclose(b.sum(), alphas.sum(), atol=1e-10, rtol=1e-7)))
    reference_probability = meta.get("reference_predicted_probability")
    if probability is not None and reference_probability is not None:
        audit["full_prediction_difference"] = probability - reference_probability
        if "full_feature_sum" in audit:
            audit["full_feature_minus_prediction_difference"] = (
                audit["full_feature_sum"] - audit["full_prediction_difference"])
        if "selected_feature_sum" in audit:
            audit["reference_plus_selected_sum"] = reference_probability + audit["selected_feature_sum"]
            audit["subset_fraction_of_full_signed_difference"] = (
                audit["selected_feature_sum"] / audit["full_prediction_difference"]
                if audit["full_prediction_difference"] else None)
    regions = regions_from_values(x, b, args.region_quantile)
    d = {
        "instance_id": args.instance_id, "model": "MiniROCKET + LogisticRegression",
        "prediction": {"class_id": target, "class_name": NAMES.get(target),
                       "predicted_class_probability": probability, "class_1_probability": class1_probability,
                       "probability_provenance": "Saved metadata predict_proba value for predicted class; class 1 is its binary complement when prediction is 0. No inference rerun.",
                       "missing_metadata": not bool(meta)},
        "dataset": {"documentation_url": DATASET_URL, "documentation_checked": "2026-09-25",
                    "original_classes": {"0": "Artifact", "1": "ExtraStole", "2": "Murmur", "3": "Normal", "4": "ExtraHLS"},
                    "binary_mapping": {"0": "Other (original classes 0, 2, 3, 4)", "1": "Extrastole (original class 1)"},
                    "mapping_source": "code/utils.py:get_abnormal_hearbeat_for_classification; export_data.py display-name array has reversed order and is not used",
                    "archive_documented_length": 3053, "saved_length": len(x),
                    "length_matches_archive": len(x) == 3053,
                    "length_caveat": "Saved CSV is authoritative. Archive describes 3053 samples; origin/version of any length discrepancy is not established. No conversion to seconds or resampling."},
        "signal": {"indices": list(range(len(x))), "values": x.tolist(), "units": None,
                   "schema": "Headerless: column 0 is zero-based pandas observation index, column 1 is recorded signal value."},
        "attribution": {"target_class": target, "target_class_name": NAMES.get(target),
                        "target_verification": "predicted directory and -L predicted select y_test_pred; metadata instance_prediction agrees with instance_label; explainer passes predict_proba[:, y_label] to SHAP",
                        "reference_policy": args.reference_policy,
                        "reference_prediction": meta.get("reference_prediction"),
                        "reference_probability_for_explained_class": reference_probability,
                        "method": "Saved SHAP-based backpropagated observation-level explanation",
                        "values": b.tolist(), "selected_feature_count": args.selected_count,
                        "selected_feature_ids": selected_ids,
                        "schema": "Headerless: column 0 is zero-based pandas observation index, column 1 is saved beta for the single channel.",
                        "scale": "Probability-output contribution units (identity-link KernelExplainer on predict_proba), not log odds, physical units or clinical risk.",
                        "sign": "Positive supports the explained class relative to the reference; negative opposes it. No sign reversal for class 1.",
                        "propagation": "back_propagate_attribution_2 multiplies input-reference differences by chained feature/activation/convolution difference ratios and kernel weights. Zero-denominator terms are skipped. These are propagated feature attributions, not exact original-input SHAP values.",
                        "selection_caveat": "Exporter uses changeclass-top-N for both top_alpha and top_alpha_that_change_class. Saved selection can be audited but exact historical option is not recorded; crossing 0.5 by additive feature contributions is not a verified prediction on a modified recording.",
                        "reference_caveat": "Opposite class uses predicted training groups. Code supports raw Euclidean or PCA-feature distance; historical metric is not saved in this metadata.",
                        "sum": float(b.sum()), "positive_sum": float(b[b > 0].sum()),
                        "negative_sum": float(b[b < 0].sum()), "absolute_sum": float(np.abs(b).sum())},
        "pipeline_inspection": {
            "model_artifact": "experiments/data/abnormal-heartbeat-c1/LogisticRegression.pkl",
            "preprocessing": "Code casts the archive signal to float32 and reshapes to one channel; 80/20 split with random_state=42. No signal scaler in this loader. Saved signal is used unchanged.",
            "transform": "Classifier wrapper stores fitted minirocket_params and logistic classifier. Current training code requests diff=True for LogisticRegression: smooth Heaviside PPV, not time differencing. Historical fitted parameters were not loaded or executed; saved alpha count is used for observed feature count.",
            "shap": "KernelExplainer with one transformed reference, default identity link; current code requests 4 times feature count samples and a 10% feature sparsity cap.",
            "provenance_limit": "Current code explains CSV semantics; metadata has no run hash, timestamp or selection option. Association rests on exact instance/policy filenames, common folder, selected IDs and numerical consistency."},
        "region_method": {"procedure": "Keep observations at or above the specified absolute-attribution quantile, split at every sign change, zero or excluded observation; rank contiguous retained sign runs by absolute sum; retain up to 3 positive and 2 negative. Ties use earliest start. Inclusive zero-based bounds. No smoothing or gap merging; threshold affects region selection only, not figures or saved values.",
                          "absolute_attribution_quantile": args.region_quantile,
                          "absolute_attribution_threshold": float(np.quantile(np.abs(b), args.region_quantile)),
                          "shape_rule": "Use raw first differences: exact monotonic/flat first; otherwise mostly rising/falling if signed net change divided by total absolute variation is >=0.6 / <=-0.6; otherwise rises and falls. No physiological labels.",
                          "selected_absolute_mass_fraction": sum(r["relative_importance"] for r in regions)},
        "influential_regions": regions, "numerical_audit": audit,
        "visualization": {"symmetric_color_limit": float(np.abs(b).max()), "center": 0,
                          "colors": {"negative": "#2166ac", "zero": "#eeeeee", "positive": "#b2182b"},
                          "geometry": "Every signal sample retained. Colored line splits original straight segments at midpoints to assign each observation its adjacent half-segments; no averaging of attributions. Heatmap has one cell per observation. No clipping or normalization of model data."},
        "sources": [source_record(p) for p in inputs],
    }
    out = args.output_dir.resolve()
    if out == base.resolve():
        raise ValueError("Use a separate output directory for derived artifacts")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "heartbeat_explanation.json"
    # Persist structured facts BEFORE creating prose. Narrative reads only those facts.
    json_path.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    d = json.loads(json_path.read_text())
    d["narrative"] = narrative_from_json(d)
    (out / "heartbeat_explanation.txt").write_text(d["narrative"])
    json_path.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    make_figures(d, out)
    for before in d["sources"]:
        if source_record(Path(before["path"])) != before:
            raise RuntimeError("An authoritative input changed during post-processing")
    print(d["narrative"])
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
