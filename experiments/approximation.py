#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
import sys

import numpy as np
import pandas
import pandas as pd
import copy

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
from export_data import export_instance_and_explanations, prepare_output_folder_for_export, \
    create_output_folder_for_export, DataExporter
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
    

    # In[42]:
    OUTPUT_FILE = 'approximation-results.csv'
    
    def compare_explanations(explanation: Explanation, explanation_p2p: Explanation) -> float:
        return r2_score(explanation_p2p.get_attributions_as_single_vector(), explanation.get_attributions_as_single_vector())
    
    df_schema = {'timestamp': [], 'base_explainer': [], 'mr_classifier': [], 'reference_policy': [], 'label': [],
                 'dataset': [], 'r2s': [], 'r2s-mean': [], 'r2s-std': [],
                 'local_accuracy': [], 'runtimes-seconds': [], 'runtimes-mean': [], 'runtimes-std': [],
                 'runtimes-p2p-seconds': [], 'runtimes-p2p-mean': [], 'runtimes-p2p-std': [],
                 'runtimes-segmented-seconds': [], 'runtimes-segmented-mean': [], 'runtimes-segmented-std': [],
                 'complexity': [], 'complexity-mean': [], 'complexity-std': [],
                 'complexity-p2p': [], 'complexity-p2p-mean': [], 'complexity-p2p-std': [],
                 'complexity-segmented': [], 'complexity-segmented-mean': [], 'complexity-segmented-std': []}
    final_df = pd.DataFrame(df_schema.copy())
    pd.DataFrame(final_df).to_csv(OUTPUT_FILE, mode='w', index=False, header=True)

    exporters_dict = {}
    for dataset_name, (dataset_fetch_function, features) in DATASET_FETCH_FUNCTIONS.items():
        (X_train, y_train), (X_test, y_test) = eval(dataset_fetch_function)
        for mr_classifier in MR_CLASSIFIERS:
            classifier = MinirocketClassifier(minirocket_features_classifier=mr_classifier)
            classifier.fit(X_train, y_train)
            y_test_pred = classifier.predict(X_test)
            mr_classifier_name = f'{dataset_name}_{mr_classifier.__class__.__name__}'
            classifier.save(f'data/{mr_classifier_name}.pkl')
            for explainer_method in EXPLAINERS:
                for label in LABELS:
                    configuration = (dataset_name, mr_classifier_name, explainer_method, label)
                    if export_data:
                        exporter = DataExporter(dataset_name, mr_classifier.__class__.__name__, label)
                        exporter.prepare_export(DATASET_FETCH_FUNCTIONS)
                        exporters_dict[configuration] = exporter
                    results_df = df_schema.copy()
                    results_df['timestamp'].append(pd.Timestamp.now())
                    results_df['base_explainer'].append(explainer_method)
                    results_df['mr_classifier'].append(mr_classifier.__class__.__name__)
                    results_df['label'].append(label)
                    results_df['dataset'].append(dataset_name)
                    idx = -1
                    for x_target, y_target in (X_test, y_test if label == 'training' else y_test_pred):
                        idx += 1
                        explanations_for_instance = {}
                        measures_for_instance = {}
                        r2s = {}
                        runtimes_backpropagated = {}
                        runtimes_p2p = {}
                        runtimes_segmented = {}
                        complexity_backpropagated = {}
                        complexity_p2p = {}
                        complexity_segmented = {}
                        local_accuracy = {}
                        for reference_policy in REFERENCE_POLICIES:
                            explainer = classifier.get_explainer(X=X_train, y=classifier.predict(X_train))

                            explanation = explainer.explain_instances(x_target, y_target,
                                                                      classifier_explainer=explainer_method,
                                                                      reference_policy=reference_policy)
                            ## Segmented explanation
                            segmented_explanation = MinirocketSegmentedClassifier(mr_classifier, explanation.get_instance(),
                                                           explanation.get_reference()).explain_instances(
                                explanation.get_instance(),
                                explanation.get_reference(),
                                explainer=explainer_method,
                                reference_policy=reference_policy
                            )
                            r2s[reference_policy] = []
                            runtimes_backpropagated[reference_policy] = []
                            runtimes_p2p[reference_policy] = []
                            runtimes_segmented[reference_policy] = []
                            complexity_backpropagated[reference_policy] = []
                            complexity_p2p[reference_policy] = []
                            complexity_segmented[reference_policy] = []
                            local_accuracy[reference_policy] = 0

                            ## Point to point explanation
                            explanation_p2p = classifier.explain_instances(explanation.get_instance(),
                                                                           explanation.get_reference(),
                                                                            explainer=explainer_method,
                                                                            reference_policy=reference_policy
                                                                           )

                            r2 = compare_explanations(explanation, explanation_p2p)
                            r2s[reference_policy].append(r2)
                            runtimes_backpropagated[reference_policy].append(explanation.get_runtime())
                            runtimes_p2p[reference_policy].append(explanation_p2p.get_runtime())
                            runtimes_segmented[reference_policy].append(segmented_explanation.get_runtime())
                            complexity_backpropagated[reference_policy].append(np.count_nonzero(explanation.explanation['coefficients']))
                            complexity_p2p[reference_policy].append(np.count_nonzero(explanation_p2p.explanation['coefficients']))
                            complexity_segmented[reference_policy].append(np.count_nonzero(segmented_explanation.get_distributed_explanations_in_original_space()))
                            (respects_local_accuracy, delta) = explanation.check_explanation_local_accuracy_wrt_minirocket()
                            local_accuracy[reference_policy] += 1 if respects_local_accuracy else 0
                            explanations_for_instance[reference_policy] = (explanation, explanation_p2p, segmented_explanation)

                            measures_for_instance[reference_policy] = (r2s[reference_policy], runtimes_backpropagated[reference_policy],
                                                                   runtimes_p2p[reference_policy], runtimes_segmented[reference_policy],
                                                                   complexity_backpropagated[reference_policy], complexity_p2p[reference_policy],
                                                                   complexity_segmented[reference_policy], local_accuracy[reference_policy])
                        if export_data:
                            exporters_dict[configuration].export_instance_and_explanations(idx, y_target,
                                                                                           features, configuration,
                                                                                           explanations_for_instance)
                    for reference_policy in measures_for_instance:
                        (r2s, runtimes_backpropagated,
                         runtimes_p2p, runtimes_segmented,
                         complexity_backpropagated, complexity_p2p,
                         complexity_segmented, local_accuracy) = measures_for_instance[reference_policy]

                        results_df_rp = copy.deepcopy(results_df)
                        results_df_rp['reference_policy'].append(reference_policy)
                        results_df_rp['complexity'] = to_sep_list(complexity_backpropagated)
                        results_df_rp['complexity-mean'] = np.mean(complexity_backpropagated)
                        results_df_rp['complexity-std'] = np.std(complexity_backpropagated)

                        results_df_rp['complexity-p2p'] = to_sep_list(complexity_p2p)
                        results_df_rp['complexity-p2p-mean'] = np.mean(complexity_p2p)
                        results_df_rp['complexity-p2p-std'] = np.std(complexity_p2p)

                        results_df_rp['runtimes-seconds'] = to_sep_list(runtimes_backpropagated)
                        results_df_rp['runtimes-mean'] = np.mean(runtimes_backpropagated)
                        results_df_rp['runtimes-std'] = np.std(runtimes_backpropagated)

                        results_df_rp['runtimes-p2p-seconds'] = to_sep_list(runtimes_p2p)
                        results_df_rp['runtimes-p2p-mean'] = np.mean(runtimes_p2p)
                        results_df_rp['runtimes-p2p-std'] = np.std(runtimes_p2p)

                        results_df_rp['runtimes-segmented-seconds'] = to_sep_list(runtimes_segmented)
                        results_df_rp['runtimes-segmented-mean'] = np.mean(runtimes_segmented)
                        results_df_rp['runtimes-segmented-std'] = np.std(runtimes_segmented)

                        results_df_rp['r2s'] = to_sep_list(r2s)
                        results_df_rp['r2s-mean'] = np.mean(r2s)
                        results_df_rp['r2s-std'] = np.std(r2s)

                        results_df_rp['local_accuracy'] = local_accuracy / len(r2s)
                        pd.DataFrame(results_df).to_csv(OUTPUT_FILE, mode='a', index=False, header=False)






    
    
    
    
