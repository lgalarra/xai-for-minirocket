import time

import numpy as np
import shap

import minirocket_multivariate_variable as mmv
from explainer import MinirocketExplainer, get_classifier_explainer, Explanation

class MinirocketClassifier:
    """
    A wrapper for a classifier that uses MiniRocket to transform the input data before fitting.
    """

    def __init__(self, minirocket_features_classifier):
        self.classifier = minirocket_features_classifier
        self.minirocket_params = None
        self.traces_obj = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y, **minirocketargs):
        """

        :param X: A time series dataset of size (n, C, L) or (n, L)
        :param y: A vector of class labels of size (n,)
        :param minirocketargs: Additional parameters for MiniRocket, check fit(...) in minirocket_multivariate_variable.py
        :return:
        """
        self.minirocket_params = mmv.fit_minirocket_parameters(X, **minirocketargs)
        self.traces_obj = mmv.transform_prime(X, parameters=self.minirocket_params)
        self.classifier.fit(self.traces_obj["phi"], y)
        self._X_train = X
        self._y_train = y

    def minirocket_transform(self, X) -> dict:
        return mmv.transform_prime(X, parameters=self.minirocket_params)

    def get_minirocket_representation(self):
        return self.traces_obj["phi"]

    def predict(self, X):
        out = mmv.transform_prime(X, parameters=self.minirocket_params)
        return self.classifier.predict(out["phi"])

    def predict_proba(self, X):
        out = mmv.transform_prime(X, parameters=self.minirocket_params)
        return self.classifier.predict_proba(out["phi"])

    def get_explainer(self, X=None, y=None) -> MinirocketExplainer:
        """
        Get an explainer object for the classifier.
        :param X: The training data. If None, use the training data used in fit(...).
        :param y: The training labels. If None, use the training labels used in fit(...).
        :param reference_policy: opposite_class_medoid | global_medoid | global_centroid | opposite_class_centroid |
        opposite_predicted_class_medoid | opposite_predicted_class_centroid | farthest_instance | custom
        :param classifier_explainer: The explainer to use for the Minirocket classifier.
        :param reference: the reference instance to use for explanation. It is used only if the reference_policy is 'custom'.
        Otherwise it is ignored
        :return: An explainer object.
        """
        if X is None or y is None:
            X = self._X_train
            y = self._y_train

        return MinirocketExplainer(X, y, minirocket_classifier=self.classifier,
                                   minirocket_params=self.minirocket_params)


    def explain_instance_on_original_space(self, x_target: np.ndarray, reference: np.ndarray, explainer='shap'):
        start = time.perf_counter()
        y_label = self.classifier.predict(self.minirocket_transform(x_target)['phi'])[0]


        classifier_explainer_fn = get_classifier_explainer(explainer,
                                                           lambda x: self.predict_proba(np.array([xi.reshape(x_target.shape) for xi in x]))[:,y_label],
                                                           X_background=np.array([reference.reshape(-1)]),
                                                           target=x_target)
        alphas = classifier_explainer_fn(np.array([x_target.reshape(-1)]))

        return {'coefficients': alphas, 'instance': x_target, 'reference': reference,
                'instance_prediction': y_label,
                'instance_logits': self.predict_proba(x_target.reshape(1, -1))[:,y_label],
                'time_elapsed': time.perf_counter() - start
                }

    def explain_instances(self, X: np.ndarray, X_reference: np.ndarray, explainer='shap'):
        explanations = []
        if len(X.shape) == 2:
            return Explanation(self.explain_instance_on_original_space(X, X_reference, explainer))
        else:
            for idx, x in enumerate(X):
                explanations.append(Explanation(self.explain_instance_on_original_space(x, X_reference[idx], explainer)))

        return explanations

class MinirocketSegmentedClassifier(MinirocketClassifier):
    def __init__(self, minirocket_features_classifier, target_instance,
                 reference_instance, num_segments=10):
        super().__init__(minirocket_features_classifier)
        self.target_instance = target_instance
        self.reference_instance = reference_instance
        self.num_segments = num_segments

    def predict(self, X):
        X_undiscretized = self._undiscretize(X)
        return super().predict(X_undiscretized)

    def predict_proba(self, X):
        X_undiscretized = self._undiscretize(X)
        return super().predict_proba(X_undiscretized)

    def _undiscretize(self, X: np.ndarray):
        X_d = list()
        for x in X:
            X_d.append(self._undiscretize_instance(x))
        return np.array(X_d)

    def _undiscretize_instance(self, x: np.ndarray):
        n_ones = np.count_nonzero(x)
        recipient = None
        sender = None
        segment_size = self.reference_instance.shape[-1] // self.num_segments
        if n_ones > x.shape[0] // 2:
            recipient = self.target_instance
            sender = self.reference_instance
        else:
            recipient = self.reference_instance
            sender = self.target_instance

        z = recipient.copy()
        segments_sender = np.array_split(sender, self.num_segments, axis=-1)
        for idx, segment in enumerate(segments_sender):
            z[..., idx*segment_size:(idx+1)*segment_size] = segment

        return z

    def explain_instance_on_original_space(self, x_target: np.ndarray, reference: np.ndarray, explainer='shap'):
        start = time.perf_counter()
        y_label = self.predict(self.minirocket_transform(x_target)['phi'])[0]


        classifier_explainer_fn = get_classifier_explainer(explainer,
                                                           lambda x: self.predict_proba(x)[:,y_label],
                                                           X_background=np.array([reference]),
                                                           target=x_target)
        alphas = classifier_explainer_fn(np.array([x_target]))

        return {'coefficients': alphas, 'instance': x_target, 'reference': reference,
                'instance_prediction': y_label,
                'instance_logits': self.predict_proba(x_target.reshape(1, -1))[:,y_label],
                'time_elapsed': time.perf_counter() - start
                }
