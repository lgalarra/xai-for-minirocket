import minirocket_multivariate_variable as mmv




class MinirocketClassifier:
    """
    A wrapper for a classifier that uses MiniRocket to transform the input data before fitting.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.minirocket_params = None
        self.traces_obj = None
        self.explainers = {{}}
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

    def minirocket_transform(self, X):
        pass

    def get_minirocket_representation(self):
        ##TODO: Verify that the thing is fit
        return self.traces_obj["phi"]

    def predict(self, X):
        out = mmv.transform_prime(X, parameters=self.minirocket_params)
        return self.classifier.predict(out["phi"])

    def explain_instances(self, X, reference_policy='opposite_class_medoid', classifier_explainer='shap',
                          reference=None):
        """
        :param X:
        :param reference_policy: opposite_class_medoid | global_medoid | global_centroid | opposite_class_centroid |
        opposite_predicted_class_medoid | opposite_predicted_class_centroid | farthest_instance | custom
        :return:
        """
        pass

    def explain_instances_from_explainer(self, X, explainerfn=None, reference=None):
        if explainerfn is None or reference is None:
            return self.explain_instances(X, reference=reference)
        return explainerfn(X)

