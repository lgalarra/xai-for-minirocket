import numpy as np


def get_gaussian_perturbation(X_target: np.ndarray, X_to: np.ndarray, explanation: np.ndarray, **kwargs):
    budget = kwargs['budget']
    new_shape = list(explanation.shape)
    new_shape[0] = new_shape[0] * budget
    if X_target.shape[0] == 1:
        std = (X_target - X_to).std()
    else:
        std = (X_target - X_to).std(axis=0)
    X_perturb = np.random.normal(0.0, kwargs['sigma'] * std, size=new_shape)
    threshold = np.percentile(explanation, kwargs['percentile_cut'])
    percentile_mask = np.vectorize(lambda x: x if x > max(threshold, 0.0) else 0.0)
    explanation_mask = percentile_mask(explanation)
    return np.repeat(X_target, budget, axis=0) + np.repeat(explanation_mask, budget, axis=0) * X_perturb


def apply_explanation_mask(xto: np.ndarray, xfrom: np.ndarray,
                           percentile_vector: np.ndarray, interpolation_level: float) -> np.ndarray:
    delta = xfrom - xto
    delta = percentile_vector * delta
    delta = interpolation_level * delta
    return xto + delta

def get_reference_perturbation(xfrom, xto, explanation, filter_explanation_fn, **kwargs):
    first_mask = np.vectorize(filter_explanation_fn)
    masked_explanation = first_mask(explanation)
    percentile = kwargs['percentile_cut']
    percentile_value = np.percentile(np.abs(masked_explanation), percentile)
    percentile_mask = np.vectorize(lambda x: x if np.abs(x) > percentile_value else 0.0)
    percentile_vector = percentile_mask(masked_explanation)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])

def get_perturbations(X_target, X_references, X_explanations, policy='gaussian', **args):
    if policy == 'gaussian':
        return get_gaussian_perturbation(X_target=X_target, X_to=X_references, explanation=X_explanations, **args)
    elif policy == 'instance_to_reference':
        return get_reference_perturbation(xfrom=X_target, xto=X_references, explanation=X_explanations,
                                          filter_explanation_fn=lambda x : x if x>0.0 else 0.0, **args)
    elif policy == 'reference_to_instance':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x<0.0 else 0.0, **args)
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")