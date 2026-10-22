# XAI for MiniROCKET

This repository is a fork of
[MiniROCKET](https://github.com/angus924/minirocket) on which we implemented XAI
for MiniROCKET. It compares the proposed back-propagated explanations with
end-to-end explanations, fixed-segment explanations (called LEFTIST in the
paper), TSHAP, and an LRP-style back-propagation variant.

## Reproducing the experiments

### Reproducibility status

The experiment pipeline is reproducible from this repository, but exact
reproduction of the numbers currently typeset in the manuscript requires the
paper-era model configuration and the complete historical result set. There
are two important qualifications:

1. The paper states that MiniROCKET used 500 features for `ford-a` and the three
   `starlight` one-vs-rest datasets, and 1,000 features for the other datasets.
   The current `MINIROCKET_PARAMS_DICT` in
   [`experiments/compute_explanations.py`](./experiments/compute_explanations.py)
   uses a newer configuration (5,000 for `ford-a`, `starlight-c1`,
   `starlight-c3`, and `handoutlines`; 1,000 for `starlight-c2`; 500 for
   `cognitive-circles`; and 10,000 for `abnormal-heartbeat-c1`). The original
   500/1,000 mapping is visible at Git revision `1b6a61ec` in
   `predict_and_explain.py`. Use that mapping and retrain all classifiers for an
   exact paper-era rerun; do not mix classifiers trained with different feature
   counts.
2. The manuscript tables were assembled before several reporting scripts were
   generalized. Current generators can therefore report additional datasets,
   segment counts, TSHAP settings, or slightly different aggregates. The asset
   inventory below distinguishes exact manuscript filters from current
   supersets.

The paper reports runs with 8 CPU cores and at least 64 GB of RAM. Runtime
ratios are meaningful only when the target instance, explainer, and reference
policy being compared ran on the same node.

### Environment and working directory

Python 3.11 is the tested environment represented by
[`requirements-3_11.txt`](./requirements-3_11.txt). From the repository root:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements-3_11.txt
export TSHAP_REPO_PATH=/path/to/tshap
```

`compute_explanations.py` imports TSHAP from a separate checkout. Set
`TSHAP_REPO_PATH` explicitly. Most legacy experiment scripts assume that the
current directory is `experiments/` and that `../code` is importable:

```bash
cd experiments
export PYTHONPATH=../code:.
```

The public benchmark loaders download/cache data through `sktime`; the
`cognitive-circles` loader instead expects the HDF5 data under
`data/cognitive-circles/`. All input arrays are converted to the shapes and
types expected by the repository loaders.

### Datasets, models, explainers, and references

The paper uses these dataset identifiers:

| Paper name | CLI identifier | Test instances | Notes |
| --- | --- | ---: | --- |
| abnormal-heartbeat | `abnormal-heartbeat-c1` | 82 | Binary class 1 task |
| cognitive-circles | `cognitive-circles` | 48 | 7 channels |
| hand-outlines | `handoutlines` | 100 | Binary class 1 task |
| ford-a | `ford-a` | 500 | Univariate |
| starlight | `starlight-c1`, `starlight-c2`, `starlight-c3` | 1,500 each | Three one-vs-rest tasks |

The models are `LogisticRegression`, `MLPClassifier`, and
`RandomForestClassifier`. The base explainers are `shap` (KernelSHAP),
`stratoshap-k1` (ST-SHAP), `extreme_feature_coalitions` (EFC), and `gradients`.
For the paper's original experiments, gradients should be restricted to
logistic regression.

The six paper reference policies are:

```text
global_centroid
global_medoid
opposite_class_closest_instance
opposite_class_farthest_instance
opposite_class_centroid
opposite_class_medoid
```

`counterfactual` and `counterfactual_dtw` are later additions and must be
excluded when reproducing the paper tables and figures. The paper uses
`--labels predicted` and Euclidean reference distance.

### 1. Train or load classifiers

Explanation scripts load
`experiments/data/<dataset>/<Classifier>.pkl`, training and saving a model if it
does not already exist. Use `train_classifiers.py` to generate the classifiers:
you can train the classifier families selected for the paper
(`LogisticRegression`, `MLPClassifier`, and `RandomForestClassifier`) or choose
any supported classifier configuration that better suits your experiment. For
a controlled run, train them first and keep the accuracy/provenance report:

```bash
python train_classifiers.py \
  --datasets abnormal-heartbeat-c1,cognitive-circles,handoutlines,ford-a,starlight-c1,starlight-c2,starlight-c3 \
  --models LogisticRegression,MLPClassifier,RandomForestClassifier \
  --random-state 42 --n-jobs 8 \
  --output results/classifier-accuracies.csv
```

For exact paper values, first restore the paper's 500/1,000 feature mapping
described above. The paper says that scikit-learn defaults were used and does
not record random seeds; the command above is the deterministic current
protocol, not a claim about the original seed.

### 2. Compute explanation data

[`experiments/compute_explanations.py`](./experiments/compute_explanations.py)
is the primary producer. For every selected test instance it:

- computes attribution `alpha` on the MiniROCKET feature vector;
- back-propagates `alpha` through the MiniROCKET traces to observation-level
  attribution `beta`;
- optionally computes an end-to-end (`p2p`) explanation on the original series;
- computes fixed-segment explanations with 10, 20, 50, and 100 segments;
- computes TSHAP explanations for window percentages 10, 15, and 20 with
  strides 5 and 20;
- exports series, references, attributions, metadata, runtime, local-accuracy,
  complexity, and Kendall statistics.

The paper used the back-propagated, end-to-end, and 10-segment outputs. A
single reproducible shard looks like this (`--end` is exclusive):

```bash
python compute_explanations.py \
  --dump-data yes \
  --datasets ford-a \
  --labels predicted \
  --models LogisticRegression \
  --explainers shap \
  --reference_policy opposite_class_medoid \
  --start 0 --end 50 \
  --metric euclidean \
  --p2p_explanations yes \
  --tshap_explanations yes \
  --n-jobs 8 --parallel-backend threading
```

Repeat over all datasets, supported model/explainer pairs, six paper reference
policies, and non-overlapping instance ranges. Use the same range boundaries on
every run. Each shard writes:

- `results/approximation-results-*.csv`, the run-level runtime/complexity data;
- `data/<dataset>/<classifier>/<explainer>/predicted/euclidean/`, containing
  range- and reference-policy-specific metadata plus the exported arrays.

Do not run two jobs that write the same metadata filename. Give each job a
distinct instance range and reference policy. Preserve the classifier pickles
with the exports because evaluation always loads classifiers from
`experiments/data/`, even when explanation metadata comes from another root.

[`experiments/compute_backpropagated_explanations.py`](./experiments/compute_backpropagated_explanations.py)
is the newer BP-vs-LRP producer. It computes the existing
`back_propagate_attribution_2` result, an epsilon-rule LRP result, and optionally
the end-to-end result, exporting them under `experiments/lrp-data/`:

```bash
python compute_backpropagated_explanations.py \
  --dump-data yes --datasets ford-a --labels predicted \
  --models LogisticRegression --explainers shap \
  --reference_policy opposite_class_medoid \
  --start 0 --end 50 --metric euclidean \
  --lrp-epsilon 1e-6 --lrp-stabilizer paper --n-jobs 8
```

This LRP pipeline post-dates the current manuscript figures. Use it for the LRP
appendix/metric comparisons produced by `experiments/paper/plot_lrp_metric_comparison.py`,
not as a silent replacement for the paper's `beta` files.

### 3. Evaluate fidelity

[`experiments/evaluate_explanations.py`](./experiments/evaluate_explanations.py)
loads precomputed explanations and measures predicted-class probability drop,
normalized probability drop, and class-change ratio. Its positional interface
is:

```text
evaluate_explanations.py DATASET LABEL EXPLAINER REFERENCE_POLICY PERTURBATION CLASSIFIER [DISTANCE] [DATA_ROOT]
```

For example:

```bash
python evaluate_explanations.py \
  ford-a predicted shap opposite_class_medoid gaussian \
  LogisticRegression euclidean data

python evaluate_explanations.py \
  ford-a predicted stratoshap-k1 opposite_class_medoid instance_to_reference \
  LogisticRegression euclidean data
```

One invocation evaluates the full hard-coded parameter grid for the requested
perturbation. The paper assets select these rows afterward:

| Paper operation | `PERTURBATION` | Selected parameters |
| --- | --- | --- |
| Occlusion | `instance_to_reference` | `percentile_cut=90`, `interpolation=1.0`, `budget=1` |
| Noise addition | `gaussian` | `percentile_cut=90`, `sigma=3.0`, `budget=10` |
| Top-k evolution | `gaussian` | `percentile_cut` 90, 75, 50 (plotted as top 10%, 25%, 50%), `sigma=3.0`, `budget=10` |

For gradients, the current reporting pipeline maps Gaussian noise to
`gradient_gaussian_opposite`; this is a later correction. Record which regime
is used rather than merging `gaussian` and `gradient_gaussian_opposite` rows.
The evaluator writes `results/perturbation-results-*.csv`. Copy completed,
non-overlapping result shards into `experiments/official-results/` before
running paper generators.

### 4. Tables and figures in the manuscript

Run reporting commands from `experiments/paper/` unless noted otherwise.

| Paper asset | Producer and required parameters |
| --- | --- |
| Table `tab:datasets` - dataset statistics | No generator exists. The table is authored in `sections/experiments.tex` from the dataset loaders. Use the paper-era train/test splits and report length, class count, channel count, and split sizes. |
| Table `tab:overhead` - trace overhead | From `experiments/`, run `python traces_overhead.py`; then from `experiments/paper/`, run `python generate_overhead_table.py`. The first command writes `experiments/traces_overhead.tsv`; the second has no CLI parameters. The producer now appends rows whose `Instance` is `BATCH`, while the paper reports prediction on one instance; filter those rows out before the generator. Runtime values are hardware-sensitive, and the current checked-in inputs do not reproduce the typeset values exactly. |
| Table `tab:runtime` and appendix runtime table | Input: `approximation-results-*` produced with `predicted`, all six paper references, all supported classifiers, and explainers. The paper compares `runtimes-p2p-seconds / runtimes-seconds` and `runtimes-segmented-seconds / runtimes-seconds` for 10 segments. Revision `7e598047` of `generate_speedup_table.py`, run against `experiments/paper/runtime-results-full/`, matches the original P2P/10-segment table layout. The current `python generate_speedup_table.py --data-dir ../official-results` is a generalized superset with extra segment/TSHAP settings and is not a bit-for-bit generator for the typeset table. The main paper shows only cognitive-circles, handoutlines, and starlight; the appendix shows all datasets. |
| Appendix Table `tab:accuracy_results` | From `experiments/`, run `python generate_accuracies_table.py`. It takes no parameters and loads all seven task-specific classifier pickles. The newer `python train_classifiers.py ... --output results/classifier-accuracies.csv` also records provenance but emits CSV rather than the manuscript LaTeX. |
| Appendix Table `tab:alpha-nonzero-ratio` | The ratios are computed from exported `alphas_instance_*.csv`, grouped by base explainer, as `count_nonzero(alpha) / len(alpha)` and then averaged. There is no dedicated table generator in the current tree; `generate_alphas_attribution.py` creates a class-wise attribution/dilation histogram, not this ratio table. |
| Appendix Tables `tab:complexity_shap`, `tab:complexity_stratoshap-k1`, `tab:complexity_gradients`, `tab:complexity_extreme_feature_coalitions` | Set `METHOD` in `generate_complexity_table.py` to `shap`, `stratoshap-k1`, `gradients`, or `extreme_feature_coalitions`, respectively, keep `INPUT_DIR=runtime-results-full`, and run the script once per method. It reads `complexity-mean`, `complexity-p2p-mean`, and the 10-segment `complexity-segmented-mean`. |
| Figure `fig:infidelity_r2i_reference_policy` - occlusion reference-policy boxplots | Run `python generate_bars_per_reference_policy.py`. Select `instance_to_reference`, `predicted`, `percentile_cut=90`, `interpolation=1.0`, `budget=1`; manuscript panels are `starlight` and `ford-a`, base explainer `stratoshap-k1`, all classifiers and six paper references. |
| Figure `fig:infidelity_gaussian_reference_policy` - noise reference-policy boxplots | Same generator; select `gaussian`, `predicted`, `percentile_cut=90`, `sigma=3.0`, `budget=10`. Manuscript panels are abnormal-heartbeat/EFC and handoutlines/gradients. The second panel is captioned “cognitive-circles” in the manuscript although the included filename is `handoutlines_gradients_...`; treat the filename as authoritative. |
| Figures `fig:fidelity_vs_top_k` and `fig:fidelity_vs_top_k_2` | In `generate_perturbation_charts.py`, set `DATA_DIR=../official-results`, `PERTURBATION_POLICY='gaussian'`, `EVOLUTION_FACTOR='percentile_cut'`, `MODEL=None`, and `EXPLANATION_METHOD='shap'` or `'gradients'`; run once for each explainer. The selected fixed values are `sigma=3.0`, `budget=10`. Copy the `ford-a`, `handoutlines`, and `cognitive-circles` PNGs. |
| Figures `fig:backpropagated-attribution` and `fig:second_feature` - qualitative case study | The raw arrays come from `predict_and_explain.py` with `--datasets abnormal-heartbeat-c1 --labels predicted --models RandomForestClassifier --explainers stratoshap-k1`, using `opposite_class_medoid` for the first analysis and `opposite_class_farthest_instance` for the reference example. First export every test instance's `alphas_mr_explanations_*` values and aggregate positive attribution per predicted class to obtain the global top 10; `--propagate_top_features 10` alone selects features per instance, not globally. `--propagate_this_feature FEATURE_ID` exports one feature's mask, convolution, thresholded trace, beta, series, and reference. The manuscript files indicate features 88 and 209, but the exact test-instance ID and plotting code are not recorded in the repository, so the six final PNG panels cannot currently be reproduced bit-for-bit. |
| Appendix Figures `fig:infidelity_gaussian_reference_policy_for_other_datasets_*` | Run `python generate_bars_per_reference_policy.py` with the same noise settings (`predicted`, top 10%, `sigma=3`, `budget=10`) for each dataset and base explainer. The panels cover `ford-a`, `starlight`, `abnormal-heartbeat-c1`, `handoutlines`, and `cognitive-circles` with SHAP, ST-SHAP, gradients, and EFC. One manuscript panel captioned “ST-SHAP on hand-outlines” includes `starlight_extreme_feature_coalitions_...`; this is a source-file mismatch and should be corrected when rebuilding the paper. |

The newer parameterized chart driver can generate a consistent superset of the
legacy figures:

```bash
python generate_all_official_perturbation_charts.py \
  --data-dir ../official-results \
  --out-dir official-perturbation-figures \
  --evolution-factor percentile_cut \
  --metric-kind probability \
  --percentile-cut 90 --sigma 3
```

This driver also emits best-method and per-classifier charts, so its directory
is not a one-to-one list of the figures included in the current manuscript.

### 5. White-box shapelet classifier

[`experiments/whitebox_classifier.py`](./experiments/whitebox_classifier.py) is
a downstream interpretability experiment. It samples instances balanced by the
MiniROCKET classifier's predicted class, computes feature-space attribution,
ranks features per predicted class, back-propagates positive regions, clusters
the resulting segments with DTW k-means, builds shapelet-count features, and
selects a decision tree on a held-out validation split. It saves the selected
tree, a PNG rendering, and an append-only TSV with accuracy and timing.

The default grid can be reproduced for one dataset with:

```bash
cd ../..
python experiments/whitebox_classifier.py \
  --dataset-name double-freq-test \
  --classifier-name RandomForestClassifier \
  --classifier-explainer shap \
  --reference-policy global_centroid \
  --sample-size 100 \
  --top-k 5,10 \
  --n-clusters 10,20 \
  --min-length 5 \
  --threshold-scale 0.05,0.1 \
  --max-depth 2,3,4,5 \
  --validation-size 0.2 \
  --random-state 0 \
  --score-mode positive \
  --n-jobs 8
```

Run once per dataset. Then generate the LaTeX accuracy table from the latest
completed RF-based run in each dataset directory:

```bash
python experiments/paper/generate_whitebox_accuracy_table.py \
  --data-root experiments/data \
  --output experiments/paper/whitebox_accuracy_table.tex \
  --digits 3
```

This experiment is newer than the tables and figures enumerated above.

### 6. Aggregated-reference explanations

[`experiments/evaluate_aggregated_explanations.py`](./experiments/evaluate_aggregated_explanations.py)
does not recompute explanations. It aligns the exported explanations for two or
more reference policies, aggregates signed observation attribution pointwise
(`max`, `min`, `avg`, or `sum`), evaluates the aggregate, and optionally evaluates
each constituent policy using paired perturbations. Example:

```bash
python experiments/evaluate_aggregated_explanations.py \
  ford-a predicted shap \
  opposite_class_medoid+opposite_class_closest_instance+opposite_class_farthest_instance \
  both LogisticRegression euclidean data \
  --aggregation min \
  --reference-strategy average-scores \
  --compare-individual \
  --percentile-cut 90 --sigma 3 --interpolation 1 \
  --budget 10 --seed 0
```

Run the same policy triple for `--aggregation min` and `--aggregation avg`, for
every desired dataset/classifier. Results go to `experiments/aggregated-results/`
as summary, per-instance, and per-sample CSVs. Generate the reports requested by
the aggregation study with:

```bash
python experiments/report_aggregated_results.py \
  --input-dir experiments/aggregated-results \
  --output-dir experiments/aggregated-results/tables \
  --explainer shap \
  --attribution-type backpropagated \
  --metric f_minus_f0-mean \
  --digits 5 --stdout markdown
```

The compact `aggregation-comparison.tex` and
`aggregation-comparison-by-classifier.tex` tables use M/C/F for the medoid,
closest, and farthest enemy baselines and compare them with the three-policy
`min` and `avg` aggregations. These aggregation assets are post-paper analyses,
not figures in the current manuscript.

### Output lineage

```text
train_classifiers.py
  -> experiments/data/<dataset>/<classifier>.pkl

compute_explanations.py
  -> experiments/data/... attribution arrays + metadata
  -> experiments/results/approximation-results-*.csv

compute_backpropagated_explanations.py
  -> experiments/lrp-data/... BP2 + LRP attribution arrays + metadata
  -> experiments/results/backpropagated-explanation-results-*.csv

evaluate_explanations.py
  -> experiments/results/perturbation-results-*.csv

evaluate_aggregated_explanations.py
  -> experiments/aggregated-results/{summary,instances,samples}.csv

experiments/paper/*.py and report_aggregated_results.py
  -> manuscript PNGs and Markdown/LaTeX tables
```
