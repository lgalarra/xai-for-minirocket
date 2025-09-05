#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os

import numpy as np
import pandas as pd

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
from classifier import MinirocketClassifier
from sklearn.metrics import accuracy_score, r2_score

MR_CLASSIFIERS = [LogisticRegression(), RandomForestClassifier()]
REFERENCE_POLICIES = ['opposite_class_medoid', 'opposite_class_centroid',
                      'global_medoid', 'global_centroid']
LABELS = ['training', 'predicted']
DATASET_FETCH_FUNCTIONS = {
    "ford-a": "get_forda_for_classification()",
    "cognitive-circles": "get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)"
}
EXPLAINERS = ['extreme_feature_coalitions', 'shap', 'stratoshap-k1']

# ## Fetching data

# In[42]:
results_df = {'mr_classifier': [], 'reference_policy': [], 'label': [], 'dataset': [],
              'classifier_accuracy': [], 'base_explainer': []}


def compare_explanations(explanation, explanation_p2p) -> float:
    return r2_score(explanation_p2p.get_attributions(), explanation.get_attributions())


for dataset_name, dataset_fetch_function in DATASET_FETCH_FUNCTIONS.items():
    (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
    for mr_classifier in MR_CLASSIFIERS:
        classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
        classifier.fit(X_train, y_train)
        for reference_policy in REFERENCE_POLICIES:
            for label in LABELS:
                for explainer_method in EXPLAINERS:
                    results_df['base_explainer'].append(explainer_method)
                    results_df['mr_classifier'].append(mr_classifier.__class__.__name__)
                    results_df['label'].append(label)
                    results_df['reference_policy'].append(reference_policy)
                    results_df['dataset'].append(dataset_name)
                    results_df['classifier_accuracy'].append(accuracy_score(y_test, classifier.predict(X_test)))
                    explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))
                    r2s = []
                    local_accuracy = 0
                    for explanation in explainer.explain_instances(X_test, y_test,
                                                              classifier_explainer=explainer_method,
                                                              reference_policy=reference_policy):
                        explanation_p2p = classifier.explain_instances(explanation['instance'],
                                                                       explanation['reference'],
                                                                        explainer=explainer_method)
                        local_accuracy += 1 if explanation.check_explanation_local_accuracy() else 0
                        r2 = compare_explanations(explanation, explanation_p2p)
                        r2s.append(r2)
                    results_df['r2s'] = r2s
                    results_df['r2s-mean'] = np.mean(r2s)
                    results_df['r2s-std'] = np.std(r2s)
                    results_df['local_accuracy'] = local_accuracy / len(r2s)


results_df = pd.DataFrame(results_df)
results_df.to_csv('approximation-results.csv')

# In[21]:



