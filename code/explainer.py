from functools import partial
from typing import Generator

import numpy as np
import shap
import inspect

from minirocket_multivariate_variable import back_propagate_attribution
from reference import centroid_time_series, medoid_time_series_idx, centroid_per_class, medoid_ids_per_class
import minirocket_multivariate_variable as mmv

class ExtremeFeatureCoalitions:
    def __init__(self, clf_fn, x_reference: np.ndarray):
        """

        :param clf_fn: A classifier function that takes an array of shape (1, ...) as input
        :param x_reference: An (1, ...) array representing a single instance
        """
        self.clf_fn = clf_fn
        self.x_reference = x_reference

    def explain_instance(self, x_target: np.ndarray) -> np.ndarray:
        print(x_target.shape, self.x_reference.shape)
        if x_target.shape[0] != 1:
            x_target = np.array([x_target])
        # Extreme Feature Coalitions
        fsc = np.zeros(x_target.shape)
        fafc = np.zeros(self.x_reference.shape)
        for idx in np.ndindex(x_target.shape):
            x_target_j = x_target[idx]
            x_reference_j = self.x_reference[idx]
            x_target[idx] = x_reference_j
            self.x_reference[idx] = x_target_j

            fafc[idx] = self.clf_fn(x_target)
            fsc[idx] = self.clf_fn(self.x_reference)

            x_target[idx] = x_target_j
            self.x_reference[idx] = x_reference_j

        return 0.5 * fsc - 0.5 * fafc

class Explanation:
    def __init__(self, explanation: dict):
        self.explanation = explanation

    def check_explanation_local_accuracy(self, tol=1e-10) -> bool:
        cake1 = self.explanation['coefficients'].sum()
        cake2 = self.explanation['minirocket_coefficients'].sum()
        return np.abs(cake1 - cake2) <= tol

    def get_reference(self) -> np.ndarray:
        return self.explanation['reference']

    def get_attributions(self) -> np.ndarray:
        return self.explanation['coefficients'].reshape(-1)

def get_minirocket_classifier_explainer(classifier_explainer, classifier_fn, X_background=None, target=None):
    if type(classifier_explainer) == str:
        if classifier_explainer == "shap":
            return shap.KernelExplainer(classifier_fn, X_background).explain
        elif classifier_explainer == 'stratoshap-k1':
            if target is None:
                raise ValueError("Target original time series must be provided when using stratoshap-k1.")
            if len(target.shape) == 1:
                budget = 2 * target.shape[0]
            else:
                budget = 2 * target.shape[0] * target.shape[1]
            return partial(shap.KernelExplainer(classifier_fn, X_background).explain,
                           kwargs=dict(nsamples=budget))
        elif classifier_explainer == 'extreme_feature_coalitions':
            ## Extreme Feature Coalitions directly works with a single background instance
            return ExtremeFeatureCoalitions(classifier_fn, X_background).explain_instance
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
                                 reference_policy, reference=None, alpha_mask=None) -> dict:
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
                                                                      X_background=np.array([reference_mr['phi'][0]]),
                                                                      target=out_x['phi'][0])
        alphas = classifier_explainer_fn(out_x['phi'])
        if alphas.shape[0] == 1:
            alphas = alphas[0]
        if alpha_mask is not None:
            alphas = alphas * alpha_mask
        beta = back_propagate_attribution(alphas, out_x["traces"], x_target, reference,
                                          per_channel=is_multichannel)
        return {'coefficients': beta, 'minirocket_coefficients': alphas,
                'instance': x_target, 'instance_transformed': out_x['phi'][0],
                'traces': out_x['traces'][0], 'reference': reference,
                'reference_prediction': self.minirocket_classifier.predict(reference_mr['phi'][0].reshape(1, -1)),
                'instance_prediction': self.minirocket_classifier.predict(out_x['phi'][0].reshape(1, -1)),
                'reference_logits': self.minirocket_classifier.predict_proba(reference_mr['phi'][0].reshape(1, -1)),
                'instance_logits': self.minirocket_classifier.predict_proba(out_x['phi'][0].reshape(1, -1))
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
                          reference_policy = 'global_centroid', reference=None, alpha_mask=None) -> Generator:
        """
        :param X: A time series dataset (n, C, L) or a single instance (C, L)
        :param y: The class labels (n,) or a single label
        :return: A generator of Explanation objects
        """
        if reference_policy == 'custom' and reference is None:
            raise ValueError("Reference must be different from None if reference_policy is 'custom'.")

        if y is None:
            y = self.minirocket_classifier.predict(mmv.transform_prime(X)['phi'])

        if len(X.shape) == 2:
            yield self._explain_single_instance(X, y, alpha_mask=alpha_mask)
        else:
            for idx, x in enumerate(X):
                yield self._explain_single_instance(x, y[idx],
                                                    classifier_explainer, reference_policy,
                                                    reference, alpha_mask=alpha_mask)
