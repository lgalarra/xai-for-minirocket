import inspect
import time
from typing import Generator

import numpy as np
import shap
import numdifftools as nd


import minirocket_multivariate_variable as mmv
from minirocket_multivariate_variable import back_propagate_attribution
from reference import centroid_time_series, medoid_time_series_idx, centroid_per_class, medoid_ids_per_class, \
    farthest_series_euclidean, closest_series_euclidean
from stratoshap.StratoShap import SHAPStratum


class Game:
    def __init__(self, model, x, x0):
        self.model = model
        self.x = x
        self.x0 = x0
        self.feature_count = len(x)

    def compute_value(self, coalition):
        m = np.zeros_like(self.x, dtype=float)
        m[list(coalition)] = 1.0
        z = self.x0 * (1 - m) + self.x * m
        return self.model(z.reshape(1, -1))[0]  # vector de probabilidades

class ExtremeFeatureCoalitions:
    def __init__(self, clf_fn, x_reference: np.ndarray):
        """

        :param clf_fn: A classifier function that takes an array of shape (1, ...) as input
        :param x_reference: An (1, ...) array representing a single instance
        """
        self.clf_fn = clf_fn
        self.x_reference = x_reference

    def explain_instance(self, x_target: np.ndarray) -> np.ndarray:
        if x_target.shape[0] != 1:
            x_target = np.array([x_target])
        # Extreme Feature Coalitions
        fx0 = self.clf_fn(self.x_reference)
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
        sum_fafc = np.sum(fafc)
        sum_fsc = np.sum(fsc)
        w = (self.clf_fn(x_target) + sum_fafc)/(sum_fsc + sum_fafc)
        return (w * fsc - (1-w) * fafc) - np.full(self.x_reference.shape, fx0 / self.x_reference.shape[-1])

class Explanation:
    def __init__(self, explanation: dict):
        self.explanation = explanation

    def check_explanation_local_accuracy_wrt_minirocket(self, tol=1e-10) -> (bool, float):
        cake1 = self.explanation['coefficients'].sum()
        cake2 = self.explanation['minirocket_coefficients'].sum()
        delta = np.abs(cake1 - cake2)
        return (delta <= tol, delta)

    def check_mr_explanation_local_accuracy_wrt_classifier(self, tol=1e-10) -> (bool, float):
        cake1 = self.explanation['instance_logit'] - self.explanation['reference_logit']
        cake2 = self.explanation['minirocket_coefficients'].sum()
        delta = np.abs(cake1 - cake2)
        return (delta <= tol, delta)

    def get_instance(self) -> np.ndarray:
        return self.explanation['instance']

    def get_reference(self) -> np.ndarray:
        return self.explanation['reference']

    def get_runtime(self):
        return self.explanation['time_elapsed']

    def get_attributions_as_single_vector(self) -> np.ndarray:
        return self.explanation['coefficients'].reshape(-1)

    def get_attributions_in_original_dimensions(self):
        return self.explanation['coefficients']

    def distribute_attributions_for_channel(self, channel):
        n_segments = self.explanation['coefficients'].shape[-1]
        standard_segment_size = channel.shape[-1] // self.explanation['coefficients'].shape[-1]
        results = []
        for idx in range(n_segments):
            s = idx * standard_segment_size
            e = min(self.explanation['instance'].shape[-1], s + standard_segment_size)
            actual_segment_size = e - s
            unfolded_attributions = np.tile(channel[idx] / actual_segment_size,
                                            actual_segment_size)
            results.append(unfolded_attributions)
        return np.concatenate(results)

    def get_distributed_explanations_in_original_space(self):
        print(self.explanation['coefficients'].shape, self.explanation['instance'].shape)
        if self.explanation['coefficients'].shape[-1] < self.explanation['instance'].shape[-1]:
            result = list()
            for channel in self.explanation['instance']:
                unfolded_attributions = self.distribute_attributions_for_channel(channel)
                result.append(unfolded_attributions)
            return np.array(result)
        else:
            raise ValueError("There is a problem: there cannot be more attributions than features.")

def get_classifier_explainer(classifier_explainer, classifier_fn, X_background=None, target=None):
    if type(classifier_explainer) == str:
        if classifier_explainer == "shap":
            return shap.KernelExplainer(classifier_fn, X_background).explain
        elif classifier_explainer == 'stratoshap-k1':
            x_flat = target.reshape(-1)
            x_bar = X_background[0].reshape(-1)
            strato = SHAPStratum()
            strato.game = Game(classifier_fn, x_flat, x_bar)
            strato.n = len(x_flat)
            strato.dim = 1
            strato.idx_dims = 0
            strato.budget = 1
            return lambda _x : strato.approximate_shapley_values()[0]
        elif classifier_explainer == 'extreme_feature_coalitions':
            ## Extreme Feature Coalitions directly works with a single background instance
            return ExtremeFeatureCoalitions(classifier_fn, X_background).explain_instance
        elif classifier_explainer == 'gradients':
            def f(x_flat):
                # Handle both single input and batched inputs (for vectorization)
                if len(x_flat.shape) == 1:
                    # Single input case
                    return classifier_fn(np.array([x_flat.reshape(X_background[0].shape)]))[0]
                else:
                    # Batched inputs case (each row is a perturbed input)
                    reshaped_inputs = np.array([xi.reshape(X_background[0].shape) for xi in x_flat])
                    results = classifier_fn(reshaped_inputs)
                    # Return results for each input
                    return results

            gradient_fn = nd.Gradient(f, method='central', step=1e-5)
            return gradient_fn
        else:
            raise ValueError(f"classifier_explainer '{classifier_explainer}' not recognized.")
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
        self.subsets = {}
        for label in np.unique(y):
            self.subsets[label] =  X[y == label]


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

        start = time.perf_counter()
        is_multichannel = x_target.shape[1] > 1
        out_x = mmv.transform_prime(x_target, parameters=self.minirocket_params)
        reference_mr = mmv.transform_prime(reference, parameters=self.minirocket_params)

        classifier_explainer_fn = get_classifier_explainer(classifier_explainer_fn,
                                                           lambda x : self.minirocket_classifier.predict_proba(x)[:,y_label],
                                                           X_background=np.array([reference_mr['phi'][0]]),
                                                           target=out_x['phi'][0])
        alphas = classifier_explainer_fn(out_x['phi'])
        if alphas.shape[0] == 1:
            alphas = alphas[0]
        if alpha_mask is not None:
            alphas = alphas * alpha_mask
        beta = back_propagate_attribution(alphas, out_x["traces"], x_target, reference,
                                          per_channel=is_multichannel, params=self.minirocket_params)
        if beta.shape[0] > 1:
            beta = beta.T

        return {'coefficients': beta, 'minirocket_coefficients': alphas,
                'instance': x_target, 'instance_transformed': out_x['phi'][0],
                'traces': out_x['traces'][0], 'reference': reference,
                'instance_label': y_label,
                'reference_prediction': self.minirocket_classifier.predict(reference_mr['phi'][0].reshape(1, -1))[0],
                'instance_prediction': self.minirocket_classifier.predict(out_x['phi'][0].reshape(1, -1))[0],
                'reference_logit': self.minirocket_classifier.predict_proba(reference_mr['phi'][0].reshape(1, -1))[0][y_label],
                'instance_logit': self.minirocket_classifier.predict_proba(out_x['phi'][0].reshape(1, -1))[0][y_label],
                'time_elapsed': time.perf_counter() - start, 'reference_policy': reference_policy
                }

    def get_reference(self, x_target, y, reference_policy) -> np.ndarray:
        if reference_policy == 'global_medoid':
            return self.global_medoid
        elif reference_policy == 'global_centroid':
            return self.global_centroid
        elif reference_policy == 'opposite_class_medoid':
            return self._X[self.medoids_per_class[1 - y]]
        elif reference_policy == 'opposite_class_centroid':
            return self.centroids_per_class[1 - y]
        elif reference_policy == 'opposite_class_farthest_instance':
            return (farthest_series_euclidean(x_target, self.subsets[1 - y]))[1]
        elif reference_policy == 'opposite_class_closest_instance':
            return (closest_series_euclidean(x_target, self.subsets[1 - y]))[1]
        else:
            raise ValueError(f"reference_policy '{reference_policy}' not recognized.")

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
            yield Explanation(self._explain_single_instance(X, y, classifier_explainer,
                                                            reference_policy,
                                                            reference,
                                                            alpha_mask=alpha_mask))
        else:
            for idx, x in enumerate(X):
                yield Explanation(self._explain_single_instance(x, y[idx],
                                                    classifier_explainer, reference_policy,
                                                    reference, alpha_mask=alpha_mask))

class SegmentedMinirocketExplainer(MinirocketExplainer):

    def __init__(self, X, y, minirocket_classifier, minirocket_params, num_segments=10):
        self.num_segments = num_segments
        super().__init__(X, y, minirocket_classifier, minirocket_params)

