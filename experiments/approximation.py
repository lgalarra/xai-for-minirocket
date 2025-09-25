#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
import sys

import numpy as np
import pandas
import pandas as pd

from explainer import Explanation
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
                   prepare_cognitive_circles_data_for_minirocket, get_forda_for_classification,
                   get_starlightcurves_for_classification, COGNITIVE_CIRCLES_CHANNELS,
                   cognitive_circles_get_sorted_channels_from_df)
from classifier import MinirocketClassifier, MinirocketSegmentedClassifier
from sklearn.metrics import accuracy_score, r2_score
from export_data import export_instance_and_explanations, prepare_output_folder_for_export
from reference import REFERENCE_POLICIES


if __name__ == '__main__':
    export_data = len(sys.argv) > 1 and sys.argv[1].lower() in ('1', "true", "yes")
    MR_CLASSIFIERS = [LogisticRegression(), RandomForestClassifier()]

    LABELS = ['training', 'predicted']
    DATASET_FETCH_FUNCTIONS = {
        "ford-a": ("get_forda_for_classification()", [('C', 'Noise intensity')]),
        "startlight-c1": ("get_starlightcurves_for_classification('1')", [('B', 'Brightness')]),
        "startlight-c2": ("get_starlightcurves_for_classification('2')", [('B', 'Brightness')]),
        "startlight-c3": ("get_starlightcurves_for_classification('3')", [('B', 'Brightness')]),
        "cognitive-circles": ("get_cognitive_circles_data_for_classification('../data/cognitive-circles', target_col='RealDifficulty', as_numpy=True)",
                              [(x, COGNITIVE_CIRCLES_CHANNELS[x]) for x in cognitive_circles_get_sorted_channels_from_df(data_dir='../data/cognitive-circles')]
                            )
    }
    EXPLAINERS = ['shap', 'stratoshap-k1', 'extreme_feature_coalitions']
    
    # ## Fetching data
    if export_data:
        prepare_output_folder_for_export(DATASET_FETCH_FUNCTIONS)

    # In[42]:
    OUTPUT_FILE = 'approximation-results.csv'
    
    def compare_explanations(explanation: Explanation, explanation_p2p: Explanation) -> float:
        return r2_score(explanation_p2p.get_attributions_as_single_vector(), explanation.get_attributions_as_single_vector())
    
    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'classifier_accuracy': [], 'r2s': [], 'r2s-mean': [], 'r2s-std': [],
                 'local_accuracy': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
                 'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
                 'runtimes-segmented-seconds': [], 'runtimes-segmented-mean': [], 'runtimes-segmented-std': [],
                 'complexity': [], 'complexity-mean': [], 'complexity-std': [],
                 'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': []}
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)
    
    
    for dataset_name, (dataset_fetch_function, features) in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        for mr_classifier in MR_CLASSIFIERS:
            classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
            classifier.fit(X_train, y_train)
            y_test_pred = classifier.predict(X_test)
            classifier.save(f'data/{dataset_name}_{mr_classifier.__class__.__name__}.pkl')
            for reference_policy in REFERENCE_POLICIES:
                for label in LABELS:
                    for explainer_method in EXPLAINERS:
                        print(f"Running: {dataset_name} {mr_classifier.__class__.__name__} {label} {explainer_method} {reference_policy}")
                        results_df = df_schema.copy()
                        results_df['timestamp'].append(pd.Timestamp.now())
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
                        idx = 0
                        for explanation in explainer.explain_instances(X_test[1:2], y_test[1:2] if label == 'training' else y_test_pred[1:2],
                                                                  classifier_explainer=explainer_method,
                                                                  reference_policy=reference_policy):
                            ## Segmented explanation
                            segmented_explanation = MinirocketSegmentedClassifier(mr_classifier, explanation.get_instance(),
                                                           explanation.get_reference()).explain_instances(
                                explanation.get_instance(),
                                explanation.get_reference(),
                                explainer=explainer_method,
                                reference_policy=reference_policy
                            )

                            ## Point to point explanation
                            explanation_p2p = classifier.explain_instances(explanation.get_instance(),
                                                                           explanation.get_reference(),
                                                                            explainer=explainer_method,
                                                                            reference_policy=reference_policy
                                                                           )
                            ## Checking the respect of properties
                            (respects_local_accuracy, delta) = explanation.check_explanation_local_accuracy_wrt_minirocket()
                            local_accuracy += 1 if respects_local_accuracy else 0
                            if not respects_local_accuracy:
                                print(f"Local accuracy not respected w.r.t Minirocket's explanations, "
                                      f"difference of {delta}", file=sys.stderr)
                            (respects_local_accuracy, delta) = explanation.check_mr_explanation_local_accuracy_wrt_classifier()
                            print("Local accuracy, ", respects_local_accuracy)
                            if not respects_local_accuracy:
                                print(f"Local accuracy not respected w.r.t. classifier, "
                                      f"difference of {delta}", file=sys.stderr)

                            r2 = compare_explanations(explanation, explanation_p2p)
                            r2s.append(r2)
                            runtimes_backpropagated.append(explanation.get_runtime())
                            runtimes_p2p.append(explanation_p2p.get_runtime())
                            runtimes_segmented.append(segmented_explanation.get_runtime())
                            complexity_backpropagated.append(np.count_nonzero(explanation.explanation['coefficients']))
                            complexity_p2p.append(np.count_nonzero(explanation_p2p.explanation['coefficients']))
                            idx += 1

                            if export_data:
                                export_instance_and_explanations(idx, y_test[idx], dataset_name,
                                                                 features, explanation,
                                                                 explanation_p2p, segmented_explanation,
                                                                 explainer_method, mr_classifier, label_type=label)

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
    
    
    
    
