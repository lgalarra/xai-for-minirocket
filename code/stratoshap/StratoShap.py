import math
import numpy as np
from stratoshap.ShapleyEstimator import ShapleyEstimator
import stratoshap.utils.utils as stratoutils

class SHAPStratum(ShapleyEstimator):
    """Uses the SHAP Stratum method to compute feature importances of any trained model."""

    def __init__(self):
        self.stratum = None
        self.grand = None
        self.null = None
        self.shapley_values = None
        self.total_stratum = None
        self.idx_dims = None

    def approximate_shapley_values(self):
        """ Calculation of the feature attribution score.

        Args:
            stratum (int): desired stratum.

        Returns:
            Explanation: the coefficient of the features w.r.t the formula used to compute explanations.
        """
        stratum = stratoutils.budget_to_stratum(self.budget - 2, self.n)
        self.stratum = max(1, stratum)
        final_budget = stratoutils.stratum_instance_number(self.stratum, self.n) + 2

        self.total_stratum = int(np.ceil((self.n - 1) / 2.0))
        self.stratum = max(1, min(self.stratum, self.total_stratum))

        full = list(range(self.n))
        empty = []
        self.null = self.game.compute_value(empty)
        self.grand = self.game.compute_value(full)

        # print(f"self.grand = {self.grand} , self.grand.shape = {self.grand.shape}, type(self.grand) = {type(self.grand)}")

        if isinstance(self.grand, np.ndarray) and len(self.grand.shape) > 1:

            num_classes = self.grand.shape[1]

            if num_classes > 10:  # if too much classes --> ImageNet
                top_k = min(1, num_classes)
                top_classes = np.argsort(-self.grand[0])[:top_k]

                self.dim = top_k
                self.grand = self.grand[:, top_classes]
                self.null = self.null[:, top_classes]

                self.idx_dims = top_classes

            else:
                self.dim = num_classes  # classification of tabular data
                self.idx_dims = np.arange(self.dim)  # Indices inchangés

        else:
            self.dim = 1
            self.grand = np.array([self.grand]).reshape(-1, 1)
            self.null = np.array([self.null]).reshape(-1, 1)
            self.idx_dims = np.array([0])

            # print(f"grand = {self.grand} , null = {self.null}")

        coalitions = stratoutils.stratum_coalitions(self.stratum, self.n)

        phi = self.shapley_values = np.zeros((self.dim, self.n))

        for d in range(self.dim):
            phi_formula = self.solve(coalitions, d)
            phi[d] = phi_formula

        self.shapley_values = phi

        return self.get_estimates(), final_budget

    def solve(self, coalitions, d):
        """Solve the equation to calculate the feature attribution values."""

        r = self.rho_strat()
        phi = np.zeros(self.n)
        ey = np.zeros((len(coalitions), self.dim))

        if self.dim == 1:
            for i in range(len(coalitions)):
                ey[i, :] = self.game.compute_value(list(coalitions[i]))
        else:
            for i in range(len(coalitions)):
                ey[i, :] = self.game.compute_value(list(coalitions[i]))[:, self.idx_dims]

        if self.dim == 1:
            ey = ey.reshape(-1, 1)

        w1_values = np.array([self.w1(coal) for coal in coalitions])
        w2_values = np.array([self.w2(coal) for coal in coalitions])

        for j in range(self.n):
            contains_j_mask = np.array([j in coal for coal in coalitions])
            not_contains_j_mask = ~contains_j_mask

            contains_j = np.sum(w1_values[contains_j_mask] * ey[contains_j_mask, d])
            not_contains_j = np.sum(w2_values[not_contains_j_mask] * ey[not_contains_j_mask, d])

            phi[j] = (1 / self.n) * (self.grand[0, d] - self.null[0, d]) + r * (contains_j - not_contains_j)

            if np.abs(phi[j]) < 1e-10:
                phi[j] = 0

        return phi

    def get_estimates(self):
        """
        Returns the estimated Shapley values, normalized if required.
        """
        return self.shapley_values

    def get_name(self):
        """
        Returns the name of the method, indicating if normalization is applied.
        """
        return "SHAPStratum"

    def rho_strat(self):
        if (self.stratum == int(self.n / 2)):
            return 1
        else:
            return (self.n - 1) / (2 * self.stratum)

    def w1(self, S):
        """Calculating the weight associated with coalitions containing the feature whose contribution is being calculated.
        Args:
            S: binary vector representing a coalition.
        Returns:
            float: ((|S|-1)!(n-|S|)!)/n!
        """
        S = list(S)

        num = math.factorial(len(S) - 1) * math.factorial(self.n - len(S))
        denom = math.factorial(self.n)

        return num / denom

    def w2(self, S):
        """Calculating the weight associated with coalitions not containing the feature whose contribution is being calculated.
        Args:
            S: binary vector representing a coalition.
        Returns:
            float: ((|S|!(n-|S|-1)!)/n!
        """
        S = list(S)

        num = math.factorial(len(S)) * math.factorial(self.n - len(S) - 1)
        denom = math.factorial(self.n)

        return num / denom

    @staticmethod
    def to_numpy_array(vector):
        """Converts any vector to a numpy array.

        Args:
            vector: list, tuple, or numpy array.

        Returns:
            numpy.array: The input vector as a numpy array.
        """
        return np.array(vector)


