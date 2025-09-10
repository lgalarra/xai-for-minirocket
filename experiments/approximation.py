#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os

import numpy as np
import pandas
import pandas as pd

from exputils import to_sep_list

# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")   # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import importlib

import minirocket_multivariate_variable as mmv

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import (get_cognitive_circles_data, get_cognitive_circles_data_for_classification,
                      prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score

MR_CLASSIFIERS = [LogisticRegression(), RandomForestClassifier()]
REFERENCE_POLICIES = ['opposite_class_medoid', 'opposite_class_centroid',
                      'global_medoid', 'global_centroid', 'opposite_class_farthest_instance',
                      'opposite_class_closest_instance']
LABELS = ['training', 'predicted']
DATASET_FETCH_FUNCTIONS = {
    "ford-a": "get_forda_for_classification()",
    "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)"
}
EXPLAINERS = ['stratoshap-k1', 'extreme_feature_coalitions', 'shap']

# ## Fetching data

# In[42]:
OUTPUT_FILE = 'approximation-results.csv'

def compare_explanations(explanation, explanation_p2p) -> float:
    return r2_score(explanation_p2p.get_attributions(), explanation.get_attributions())

df_schema = {'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
             'dataset': [], 'classifier_accuracy': [], 'r2s': [], 'r2s-mean': [], 'r2s-std': [],
             'local_accuracy': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
             'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
             'runtimes-segmented-seconds': [], 'runtimes-segmented-mean': [], 'runtimes-segmented-std': [],
             'complexity': [], 'complexity-mean': [], 'complexity-std': [],
             'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': []}
final_df = pd.DataFrame(df_schema.copy())
pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)


for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
    (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
    for mr_classifier in MR_CLASSIFIERS:
        classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
        classifier.fit(X_train, y_train)
        for reference_policy in REFERENCE_POLICIES:
            for label in LABELS:
                for explainer_method in EXPLAINERS:
                    print(f"Running: {dataset_name} {mr_classifier.__class__.__name__} {label} {explainer_method} {reference_policy}")
                    results_df = df_schema.copy()
                    results_df['base_explainer'].append(explainer_method)
                    results_df['mr_classifier'].append(mr_classifier.__class__.__name__)
                    results_df['label'].append(label)
                    results_df['reference_policy'].append(reference_policy)
                    results_df['dataset'].append(dataset_name)
                    results_df['classifier_accuracy'].append(accuracy_score(y_test, classifier.predict(X_test)))
                    explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                    r2s = []
                    local_accuracy = 0
                    runtimes_backpropagated = []
                    runtimes_p2p = []
                    runtimes_segmented = []
                    complexity_backpropagated = []
                    complexity_p2p = []
                    for explanation in explainer.explain_instances(X_test[0:1], y_test[0:1],
                                                              classifier_explainer=explainer_method,
                                                              reference_policy=reference_policy):
                        ## Point to point explanation
                        explanation_p2p = classifier.explain_instances(explanation.get_instance(),
                                                                       explanation.get_reference(),
                                                                        explainer=explainer_method)
                        local_accuracy += 1 if explanation.check_explanation_local_accuracy() else 0
                        r2 = compare_explanations(explanation, explanation_p2p)
                        r2s.append(r2)
                        runtimes_backpropagated.append(explanation.get_runtime())
                        runtimes_p2p.append(explanation_p2p.get_runtime())

                        ## Segmented explanation
                        segmented_explanation = MinirocketSegmentedClassifier(classifier, explanation.get_instance(),
                                                       explanation.get_reference()).explain_instances(
                            explanation.get_instance(),
                            explanation.get_reference(),
                            explainer=explainer_method
                        )
                        runtimes_segmented.append(segmented_explanation.get_runtime())
                        complexity_backpropagated.append(np.count_nonzero(explanation.get_attributions()))
                        complexity_p2p.append(np.count_nonzero(explanation_p2p.get_attributions()))

                    results_df['complexity'] = to_sep_list(complexity_backpropagated)
                    results_df['complexity-mean'] = np.mean(complexity_backpropagated)
                    results_df['complexity-std'] = np.std(complexity_backpropagated)
                    results_df['complexity-p2p'] = to_sep_list(complexity_p2p)
                    results_df['complexity-p2p-mean'] = np.mean(complexity_p2p)
                    results_df['complexity-p2p-std'] = np.std(complexity_p2p)
                    results_df['runtimes-seconds'] = to_sep_list(runtimes_backpropagated)
                    results_df['runtimes-mean'] = np.mean(runtimes_backpropagated)
                    results_df['runtimes-std'] = np.std(runtimes_backpropagated)
                    results_df['runtimes-p2p-seconds'] = to_sep_list(runtimes_p2p)
                    results_df['runtimes-p2p-mean'] = np.mean(runtimes_p2p)
                    results_df['runtimes-p2p-std'] = np.std(runtimes_p2p)
                    results_df['runtimes-segmented-seconds'] = to_sep_list(runtimes_segmented)
                    results_df['runtimes-segmented-mean'] = np.mean(runtimes_segmented)
                    results_df['runtimes-segmented-std'] = np.std(runtimes_segmented)
                    results_df['r2s'] = to_sep_list(r2s)
                    results_df['r2s-mean'] = np.mean(r2s)
                    results_df['r2s-std'] = np.std(r2s)
                    results_df['local_accuracy'] = local_accuracy / len(r2s)
                    pd.DataFrame(results_df).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)




