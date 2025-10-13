import numpy as np


def get_gaussian_perturbation(X_target: np.ndarray, explanation: np.ndarray, **kwargs):
    new_shape = list(X_target.shape)
    budget = kwargs['budget']
    new_shape[0] = new_shape[0] * budget
    X_perturb = np.random.normal(0, kwargs['sigma'], size=new_shape)
    threshold = np.percentile(np.abs(explanation), kwargs['percentile_cut'])
    percentile_mask = np.vectorize(lambda x: x if np.abs(x) > threshold else 0)
    return X_target + percentile_mask(X_perturb)


def apply_explanation_mask(xto: np.ndarray, xfrom: np.ndarray,
                           percentile_mask: callable, noise_level: float) -> np.ndarray:
    delta = noise_level * percentile_mask(xfrom - xto)
    return xto + delta

def get_reference_perturbation(xfrom, xto, explanation, filter_explanation_fn, **kwargs):
    first_mask = np.vectorize(filter_explanation_fn)
    masked_explanation = first_mask(explanation)
    percentile = kwargs['percentile_cut']
    percentile_mask = np.vectorize(lambda x: x if np.abs(x) > np.percentile(np.abs(masked_explanation), percentile) else 0)
    return apply_explanation_mask(xto, xfrom, percentile_mask, kwargs['noise_level'])

def get_perturbations(X_target, X_references, X_explanations, policy='gaussian', **args):
    if policy == 'gaussian':
        return get_gaussian_perturbation(X_target=X_target, explanation=X_explanations, **args)
    elif policy == 'instance_to_reference':
        return get_reference_perturbation(xfrom=X_target, xto=X_references, explanation=X_explanations,
                                          filter_explanation_fn=lambda x : x if x>0 else 0, **args)
    elif policy == 'reference_to_instance':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x<0 else 0, **args)
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")