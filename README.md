[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET

***MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification***

[KDD 2021](https://dl.acm.org/doi/abs/10.1145/3447548.3467231) / [arXiv:2012.08791](https://arxiv.org/abs/2012.08791) (preprint)

> <div align="justify">Until recently, the most accurate methods for time series classification were limited by high computational complexity.  ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier.  We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy.  Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes.  MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense.  As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.</div>

Please cite as:

```bibtex
@inproceedings{dempster_etal_2021,
  author    = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title     = {{MiniRocket}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  address   = {New York},
  year      = {2021},
  pages     = {248--257}
}
```

## Podcast

Hear more about MINIROCKET and time series classification on the [Data Skeptic](https://dataskeptic.com/blog/episodes/2021/minirocket) podcast!

## GPU Implementation \*NEW\*

A GPU implementation of MINIROCKET, developed by Malcolm McLean and Ignacio Oguiza, is available through [`tsai`](https://github.com/timeseriesAI/tsai).  See the [examples](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb).  Many thanks to Malcolm and Ignacio for their work in developing the GPU implementation and making it part of `tsai`.

## `sktime`\* / Multivariate

MINIROCKET (including a basic multivariate implementation) is also available through [`sktime`](https://github.com/alan-turing-institute/sktime).  See the [examples](https://github.com/alan-turing-institute/sktime/blob/master/examples/minirocket.ipynb).

\* *for larger datasets (10,000+ training examples), the `sktime` methods should be integrated with SGD or similar as per [`softmax.py`](./code/softmax.py) (replace calls to `fit(...)` and `transform(...)` from `minirocket.py` with calls to the relevant `sktime` methods as appropriate)*

## Results

* UCR Archive (109 Datasets, 30 Resamples)
  * [Mean Accuracy + Training/Test Times](./results/results_ucr109_mean.csv)
  * [Accuracy Per Resample](./results/accuracy_ucr109_resamples.csv)
* Scalability / Training Set Size\*
  * [MosquitoSound](./results/time_training_MosquitoSound.csv) (139,780 &times; 3,750)
  * [InsectSound](./results/time_training_InsectSound.csv) (25,000 &times; 600)
  * [FruitFlies](./results/time_training_FruitFlies.csv) (17,259 &times; 5,000)
* Scalability / Time Series Length
  * [DucksAndGeese](./results/time_training_DucksAndGeese.csv) (50 &times; 236,784)

\* *`num_training_examples` does* ***not*** *include the validation set of 2,048 training examples, but the transform time for the validation set* ***is*** *included in `time_training_seconds`*

## Requirements\*

* Python, NumPy, pandas
* Numba (0.50+)
* scikit-learn or similar
* PyTorch or similar (for larger datasets)

\* *all pre-packaged with or otherwise available through Anaconda*

## Code

### [`minirocket.py`](./code/minirocket.py)
### [`minirocket_dv.py`](./code/minirocket_dv.py) (MINIROCKET<sub>DV</sub>)
### [`softmax.py`](./code/softmax.py) (PyTorch / 10,000+ Training Examples)
### [`minirocket_multivariate.py`](./code/minirocket_multivariate.py) (equivalent to [sktime/MiniRocketMultivariate](https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformations/panel/rocket/_minirocket_multivariate.py))
### [`minirocket_variable.py`](./code/minirocket_variable.py) (variable-length input; *experimental*)

## Important Notes

### Compilation

The functions in [`minirocket.py`](./code/minirocket.py) and [`minirocket_dv.py`](./code/minirocket_dv.py) are compiled by Numba on import, which may take some time.  By default, the compiled functions are now cached, so this should only happen once (i.e., on the first import).

### Input Data Type

Input data should be of type `np.float32`.  Alternatively, you can change the Numba signatures to accept, e.g., `np.float64`.

### Normalisation

Unlike ROCKET, MINIROCKET does **not** require the input time series to be normalised.  (However, whether or not it makes sense to normalise the input time series may depend on your particular application.)

## Examples

### Aggregate precomputed explanations

Run from the repository root:

```bash
python experiments/plot_aggregated_explanations.py \
  --dataset abnormal-heartbeat-c1 --classifier LogisticRegression --explainer shap \
  --reference-policies opposite_class_closest_instance,opposite_class_medoid,global_centroid \
  --aggregation avg --instances 0,22 \
  --output-dir reports/aggregated_explanations
```

This reads existing `DataExporter` CSVs from
`experiments/data/<dataset>/<classifier>/<explainer>/<label>/<metric>/`.
Use `--export-root PATH` for a different directory containing `data/`; relative
paths inside metadata are resolved against that directory. Defaults are
`--label predicted`, `--metric euclidean`, and `--attribution beta`.
Policy-specific metadata, range-sharded metadata, and legacy `metadata.csv`
are supported. No model or explainer is executed.

Aggregation (`max`, `avg`, `sum`, or `min`) operates on signed, unnormalized
attributions at each time step independently across the selected policies.
Each instance produces a PNG with one subplot per channel: a black series line
over a heatmap (red positive, blue negative), with a shared symmetric color scale.
Omit `--instances` to plot all exported instances. Missing explanations,
conflicting metadata, mismatched series/targets, or time indices raise errors.
Use a separate output directory for each policy group to avoid overwriting images.
Other exported attribution types can be selected with `--attribution`, for
example `beta_p2p`, `beta_lrp`, `beta_segmented`, or `beta_tshap_w10_s5`.

### Evaluate aggregated explanations

`experiments/evaluate_aggregated_explanations.py` reuses the original evaluator's
perturbation and metric functions, without modifying `evaluate_explanations.py`.
Its positional arguments are dataset, label, explainer, reference policy group,
perturbation, classifier, optional distance, and optional data root (`data` or
`lrp-data`). Join constituent policies with `+`:

```bash
python experiments/evaluate_aggregated_explanations.py \
  double-freq-test predicted shap \
  opposite_class_medoid+opposite_class_closest_instance \
  both RandomForestClassifier \
  --aggregation max --compare-individual --start 0 --end 9 \
  --output experiments/aggregated-results/evaluation-max.csv
```

`--aggregation` accepts `max`, `min`, `avg` (default), or `sum` and combines
signed attributions pointwise. All explanation types available for every
constituent policy are evaluated independently, including backpropagated, P2P,
segmented, TSHAP, and LRP exports. Missing types are omitted, while partially
missing or mismatched arrays fail validation. Individual policy names also work.

Select instance IDs with `--instances 0,3,9` or an inclusive interval such as
`--start 86 --end 99`. Omit `--end` to evaluate all remaining exported test
instances from `--start` onward. `--end` requires `--start`; bounds must satisfy
`0 <= start <= end` when both are supplied, and cannot be combined with `--instances`.
Every requested ID must be available for all selected policies; missing IDs
raise an error. Omit both selection forms to evaluate all exported instances.

This script supports `gaussian`, `instance_to_reference`, or `both`, with one
explicit configuration per invocation: `--percentile-cut 90`, `--sigma 3`,
`--interpolation 1`, `--budget 10`, and `--seed 0` by default. The budget applies
to Gaussian trials; deterministic instance-to-reference uses one trial.
The existing Gaussian implementation uses clipped random interpolation toward
the reference (sigma 3 maps to interpolation standard deviation 0.25).

Both perturbations require replacement references. By default,
`--reference-strategy average-scores` evaluates the aggregated explanation
against each constituent reference, then averages the probability drops.
`mean-reference` instead uses the mean reference series; `first` uses the first
policy's reference. Individual baselines use their own references.
Individual comparisons are enabled by default; use `--no-compare-individual`
to evaluate only the aggregate. `--compare-individual` explicitly enables them.
Comparisons use a shared per-instance perturbation-count limit:
the minimum count basis among the aggregate and individual explanations,
using the original evaluator's P2P-first rule. Percentile/positive-attribution
selection can reduce the actual count further. Random draws are paired by seed.

By default, results are saved in `experiments/aggregated-results/`, which is
created automatically if it does not exist. Use `--output` to override the path.
The default summary filename is
`<dataset>__<label>__<explainer>__<policies>__<classifier>__<aggregation>__<perturbation>.csv`,
with policies joined by `+`. Per-instance and sample files use the same stem.
Runs with the same filename overwrite previous results; use `--output` to
distinguish changes to instance selection, seed, or other evaluation settings.
The output consists of a summary CSV, an `-instances.csv` file (trial- and
reference-averaged scores), and a `-samples.csv` file retaining each target and
trial. `f_minus_f0` is the original predicted-class probability minus its
perturbed probability; higher values indicate larger drops. Normalized metrics
can be nonfinite when the reference probability equals the instance probability.
Model and export loading require the same dependencies as the original evaluator;
`TSHAP_REPO_PATH` can locate its TSHAP dependency. No explanations are recomputed.

### Reconstruct missing metadata

`experiments/reconstruct_explanation_metadata.py` stages metadata for header-only
`data/` exports using matching `lrp-data/` templates. It keeps the original
`data/` backpropagated explanation files, removes LRP columns, checks all required
file paths, and verifies sampled series/reference contents and classifier
predictions before writing staged CSVs and `audit.json`. It never installs files
or overwrites populated metadata. Optional explanation paths that do not exist
are cleared and recorded. Defaults target cognitive-circles, LogisticRegression,
SHAP, predicted/euclidean, and closest + medoid + farthest opposite-class policies.

```bash
python experiments/reconstruct_explanation_metadata.py \
  --experiments-root /path/to/source/experiments \
  --model-file /path/to/source/experiments/data/cognitive-circles/LogisticRegression.pkl \
  --staging-dir reports/reconstructed-metadata
```

Use the classifier saved with the source exports. Content/prediction checks use
seven evenly spaced instances by default (`--sample-size`); passing this check
does not prove historical model provenance for every attribution. Staged paths
are relative to the source experiments directory. Inspect the audit and back up
original metadata before installing the reconstructed CSVs.

### Report aggregated evaluation results

```bash
python experiments/report_aggregated_results.py
```

Reads summary CSVs from `experiments/aggregated-results/` (excluding per-instance
and per-sample files), prints Markdown to the terminal, and writes two tables
under `experiments/aggregated-results/tables/`: `by-dataset.md/.tex` and
`by-dataset-classifier.md/.tex`. `report.md` and `report.tex` include both tables,
policy/group/configuration legends, and reporting methodology. LaTeX files are
fragments suitable for inclusion in a document; wide tables may need landscape
pages or scaling. The Markdown report also lists input files for provenance.

The script also writes `aggregation-comparison.tex`, a compact dataset-level
table with O for occlusion (`instance_to_reference`) and N for noise addition
(`gaussian`). Its M, C, and F columns are the medoid, closest-instance, and
farthest-instance baselines from the three-policy `min` run; `min` and `avg` are
the corresponding three-policy combined scores. Scores are equal-weight means
over classifier coverage shared by both aggregations, and unavailable
combinations are shown as dashes. The `starlight-c1`, `starlight-c2`,
and `starlight-c3` scores are first averaged per dataset and then reported as a
single, equal-weight `starlight` result.
The dataset cells use `\multirow`, and `booktabs` rules separate dataset blocks,
so documents including this fragment must load the LaTeX `multirow` and
`booktabs` packages. Scores in this compact table use three decimal places;
absolute values below 0.001 are displayed as `< 0.001`. Three-point spacing
between all columns and suppressed outer padding keep the table compact enough
for a two-column document. The best score in each row is bold; exact ties are
all bolded.

`aggregation-comparison-by-classifier.tex` provides the same comparison with a
classifier column. Logistic regression, MLP, and random forest are abbreviated
as LR, MLP, and RF. Starlight scores remain equal-weight averages across C1--C3,
computed separately for each classifier. This appendix-oriented table uses
standard LaTeX intercolumn spacing for a full-width, two-column float. Classifier
labels use `\multirow` across their O and N rows, with partial `\cmidrule`
separators that leave the dataset column open.

Defaults are `--explainer shap --attribution-type backpropagated
--metric f_minus_f0-mean`. Use `--attribution-type p2p` (or another exported
variant), `--metric f_minus_f0_norm-mean`, `--digits 7`, `--stdout latex`, or
`--stdout none` as needed. `--input-dir` and `--output-dir` override directories.
Use different output directories to preserve reports for different variants.

Dataset tables average classifier scores with equal weight. Both tables keep
Gaussian and instance-to-reference results, aggregations, and evaluation
configurations separate. Individual policies receive dynamic P1, P2, ... columns;
each combination gets its own group row and Combined column, so adding results
for three or more policies needs no code change. Baselines stay within their
original group because instance coverage and perturbation-count limits may
differ between groups. Delta best is Combined minus the largest individual
score at that table's level. For probability-drop metrics, larger is better.
Missing baselines, unequal within-run coverage/count limits, nonfinite scores,
and duplicate experiments cause errors rather than silent averaging.

**MINIROCKET**

```python
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters = fit(X_training)

X_training_transform = transform(X_training, parameters)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**MINIROCKET<sub>DV</sub>**

```python
from minirocket_dv import fit_transform
from minirocket import transform
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters, X_training_transform = fit_transform(X_training)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**PyTorch / 10,000+ Training Examples**

```python
from softmax import train, predict

model_etc = train("InsectSound_TRAIN_shuffled.csv", num_classes = 10, training_size = 22952)
# note: 22,952 = 25,000 - 2,048 (validation)

predictions, accuracy = predict("InsectSound_TEST.csv", *model_etc)
```

**Variable-Length Input** (*Experimental*)

```python
from minirocket_variable import fit, transform, filter_by_length
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

# special instructions for variable-length input:
# * concatenate variable-length input time series into a single 1d numpy array
# * provide another 1d array with the lengths of each of the input time series
# * input data should be np.float32 (as above); lengths should be np.int32

# optionally, use a different reference length when setting dilation (default is
# the length of the longest time series), and use fit(...) with time series of
# at least this length, e.g.:
# >>> reference_length = X_training_lengths.mean()
# >>> X_training_1d_filtered, X_training_lengths_filtered = \
# >>> filter_by_length(X_training_1d, X_training_lengths, reference_length)
# >>> parameters = fit(X_training_1d_filtered, X_training_lengths_filtered, reference_length)

parameters = fit(X_training_1d, X_training_lengths)

X_training_transform = transform(X_training_1d, X_training_lengths, parameters)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test_1d, X_test_lengths, parameters)

predictions = classifier.predict(X_test_transform)
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:<sub>:rocket:</sub><sub><sub>:rocket:</sub></sub></div>
