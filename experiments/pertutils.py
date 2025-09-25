import numpy as np


def get_gaussian_perturbation(x_target: np.ndarray, explanation: np.ndarray, budget, **kwargs):
    X = []
    for i in range(budget):
        X.append(x_target.copy())
    X = np.array(X)
    X_perturb = np.random.normal(0, kwargs['sigma'], size=X.shape)

    return X + X_perturb


def apply_explanation_mask(xto: np.ndarray, xfrom: np.ndarray, percentile_mask: np.ndarray, noise_ratio: float) -> np.ndarray:
    delta = noise_ratio * (xfrom - xto) * percentile_mask
    return xto + delta

def get_perturbation(xfrom, xto, explanation, filter_explanation_fn, **kwargs):
    first_mask = np.vectorize(filter_explanation_fn)
    masked_explanation = first_mask(explanation)
    percentile = kwargs['percentile_cut']
    percentile_mask = np.vectorize(lambda x: x > np.percentile(masked_explanation, percentile))
    return apply_explanation_mask(xto, xfrom, percentile_mask, kwargs['noise_ratio'])

def get_perturbations(x_target, x_reference, explanation, budget=100, policy='gaussian', **args):
    if policy == 'gaussian':
        return get_gaussian_perturbation(x_target=x_target, explanation=explanation, budget=budget, **args)
    elif policy == 'instance_to_reference':
        return np.array([get_perturbation(xfrom=x_reference, xto=x_target, explanation=explanation, lambda x : x>0, **args)])
    elif policy == 'reference_to_instance':
        return np.array([get_perturbations(xfrom=x_target, xto=x_reference, explanation=explanation, lambda x: x<0, **args)])
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")