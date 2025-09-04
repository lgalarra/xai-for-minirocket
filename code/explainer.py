from functools import partial

import numpy as np
import shap
import inspect

from seobject import kwargs

from minirocket_multivariate_variable import back_propagate_attribution
from reference import centroid_time_series, medoid_time_series_idx, centroid_per_class, medoid_ids_per_class
import minirocket_multivariate_variable as mmv

class ExtremeFeatureCoalitions:
    def __init__(self, clf_fn, x_reference):
        self.clf_fn = clf_fn
        self.x_reference = x_reference

    def explain(self, x_target) -> np.ndarray:
        # Extreme Feature Coalitions
        single_coalition = self.x_reference.copy()
        almost_full_coalition = x_target.copy()
        y_t = int(np.argmax(self.clf_fn(x_target)[0]))
        fsc = np.zeros(x_target.shape)
        fafc = np.zeros(self.x_reference.shape)
        for x_target_j, x_reference_j, sc, afc, vsc, vafc in np.nditer([x_target, self.x_reference,
                                                                        single_coalition, almost_full_coalition, fsc, fafc],
                           op_flags=[['readonly'], ['readonly'], ['readwrite'], ['readwrite'], ['readwrite'], ['readwrite']] ):

            almost_full_coalition[...] = x_reference_j
            single_coalition[...] = x_target_j
            fafc[...] = self.clf_fn(almost_full_coalition)[:,y_t]
            fsc[...] = self.clf_fn(single_coalition)[:,y_t]
            almost_full_coalition[...] = x_target_j
            single_coalition[...] = x_reference_j

        return 0.5 * fsc - 0.5 * fafc

class Explanation:
    def __init__(self):
        self.instances = list()

    def add_instance(self, explanation: dict):
        self.instances.append(explanation)

    def check_explanations_local_accuracy(self, tol=1e-10):
        checks = []
        for instance_exp in self.instances:
            cake1 = instance_exp['coefficients'].sum()
            cake2 = instance_exp['minirocket_coefficients'].sum()
            checks.append(np.abs(cake1 - cake2) <= tol)
        return np.array(checks)

    def get_references(self):
        X = list()
        for instance_exp in self.instances:
            X.append(instance_exp['reference'])
        return np.array(X)

    def get_attributions(self):
        X = list()
        for instance_exp in self.instances:
            X.append(instance_exp['coefficients'].reshape(-1))
        return np.array(X)

def get_minirocket_classifier_explainer(classifier_explainer, classifier_fn, background=None, target=None):
    if type(classifier_explainer) == str:
        if classifier_explainer == "shap":
            return shap.KernelExplainer(classifier_fn, background).explain
        elif classifier_explainer == 'stratoshap-k1':
            if target is None:
                raise ValueError("Target original time series must be provided when using stratoshap-k1.")
            if len(target.shape) == 1:
                budget = 2 * target.shape[0]
            else:
                budget = 2 * target.shape[0] * target.shape[1]
            return partial(shap.KernelExplainer(classifier_fn, background).explain,
                           kwargs=dict(nsamples=budget))
        elif classifier_explainer == 'extreme_feature_coalitions':
            return ExtremeFeatureCoalitions(classifier_fn, background).explain
    elif inspect.isfunction(classifier_explainer):
        return classifier_explainer
    elif inspect.isclass(classifier_explainer):
        return classifier_explainer
    else:
        raise ValueError("classifier_explainer must be a string or a function or a class object with an 'explain' method.")


class MinirocketExplainer:
    def __init__(self, X, y, minirocket_classifier, minirocket_params):
        self.minirocket_params = minirocket_params
        self.minirocket_classifier = minirocket_classifier
        self._X = X
        self._y = y
        self.global_medoid = X[medoid_time_series_idx(X)]
        self.global_centroid = centroid_time_series(X)
        self.centroids_per_class = centroid_per_class(X, y)
        self.medoids_per_class = medoid_ids_per_class(X, y)


    def _explain_single_instance(self, x_target: np.ndarray, y_label, classifier_explainer_fn,
                                 reference_policy, reference=None) -> dict:
        """

        :param x_target: An array of shape (C, L) representing a multivariate time series
        :param y: The class label
        :return: a dictionary with the attributions as arrays (C, L) and other information
        """
        if reference_policy == 'custom':
            reference = reference
        else:
            reference = self.get_reference(x_target, y_label, reference_policy)

        is_multichannel = x_target.shape[1] > 1
        out_x = mmv.transform_prime(x_target, parameters=self.minirocket_params)
        reference_mr = mmv.transform_prime(reference, parameters=self.minirocket_params)
        classifier_explainer_fn = get_minirocket_classifier_explainer(classifier_explainer_fn,
                                                                      lambda x : self.minirocket_classifier.predict_proba(x)[:,y_label],
                                                                      background=np.array(reference_mr['phi']),
                                                                      target=out_x['phi'])
        alphas = classifier_explainer_fn(out_x['phi'])
        beta = back_propagate_attribution(alphas, out_x["traces"], x_target, reference,
                                          per_channel=is_multichannel)
        return {'coefficients': beta, 'minirocket_coefficients': alphas,
                'instance': x_target, 'instance_transformed': out_x['phi'],
                'traces': out_x['traces'], 'reference': reference,
                'reference_prediction': self.minirocket_classifier.predict(reference_mr['phi']),
                'instance_prediction': self.minirocket_classifier.predict(out_x['phi']),
                'reference_logits': self.minirocket_classifier.predict_proba(reference_mr['phi']),
                'instance_logits': self.minirocket_classifier.predict_proba(out_x['phi'])
                }

    def get_reference(self, X, y, reference_policy) -> np.ndarray:
        if reference_policy == 'global_medoid':
            return self.global_medoid
        elif reference_policy == 'global_centroid':
            return self.global_centroid
        elif reference_policy == 'opposite_class_medoid':
            return self._X[self.medoids_per_class[1 - y]]
        elif reference_policy == 'opposite_class_centroid':
            return self.centroids_per_class[1 - y]
        else:
            ##TODO: Add support for farthest instance policy
            raise ValueError(f"Unsupported reference policy: {reference_policy}")


    def explain_instances(self, X: np.ndarray, y=None, classifier_explainer='shap',
                          reference_policy = 'global_centroid', reference=None) -> Explanation:
        """
        :param X: A time series dataset (n, C, L) or a single instance (C, L)
        :param y: The class labels (n,) or a single label
        :return: An explanation object with the attributions per instance as arrays (n, C, L)
        """
        if reference_policy == 'custom' and reference is None:
            raise ValueError("Reference must be different from None if reference_policy is 'custom'.")

        explanation = Explanation()
        if y is None:
            y = self.minirocket_classifier.predict(mmv.transform_prime(X)['phi'])

        if len(X.shape) == 2:
            explanation.add_instance(self._explain_single_instance(X, y[0],
                                                                   classifier_explainer,
                                                                   reference_policy, reference))
        else:
            for idx, x in enumerate(X):
                explanation.add_instance(self._explain_single_instance(x, y[idx],
                                                                       classifier_explainer,
                                                                       reference_policy, reference))

        return explanation

