from typing import Callable

import numpy as np

def zero_out_random_ones(arr, x, rng=None):
    arr = arr.copy()
    rng = np.random.default_rng(rng)

    flat_indices = np.flatnonzero(arr)
    if x > flat_indices.size:
        raise ValueError("x is larger than the number of 1s")

    chosen = rng.choice(flat_indices, size=x, replace=False)
    coords = np.unravel_index(chosen, arr.shape)

    arr[coords] = 0
    return arr


def get_gaussian_perturbation(X_target: np.ndarray, X_to: np.ndarray, explanation: np.ndarray,
                              filter_explanation_fn: Callable,
                              **kwargs):
    budget = kwargs['budget']
    padded_explanation = explanation
    if explanation.shape[-1] < X_target.shape[-1]:
        padded_explanation = np.pad(explanation, pad_width=((0, 0), (0, 0), (0, X_target.shape[-1] - explanation.shape[-1])),
                                   mode="edge")
    new_shape = list(padded_explanation.shape)
    new_shape[0] = new_shape[0] * budget
    if X_target.shape[0] == 1:
        std = (X_to - X_target).std()
        avg = (X_to - X_target).mean()
    else:
        std = (X_to - X_target).std(axis=0)
        avg = (X_to - X_target).mean(axis=0)
    X_perturb = np.random.normal(avg, kwargs['sigma'] * std, size=new_shape)
    percentile_mask = np.vectorize(filter_explanation_fn)
    explanation_mask = percentile_mask(padded_explanation)
    explanation_size = np.count_nonzero(explanation_mask)
    #print('Original explanation size:', explanation_size)
    #if 'n_perturbed_points' in kwargs and kwargs['n_perturbed_points'] < explanation_size:
    #    n_zeros =  explanation_size - kwargs['n_perturbed_points']
    #    explanation_mask = zero_out_random_ones(explanation_mask, n_zeros, rng=np.random.default_rng())
    #    explanation_size = np.count_nonzero(explanation_mask)
    #    print('New explanation size:', explanation_size)

    return np.repeat(X_target, budget, axis=0) + np.repeat(explanation_mask, budget, axis=0) * X_perturb, explanation_size


def apply_explanation_mask(xto: np.ndarray, xfrom: np.ndarray,
                           percentile_vector: np.ndarray, interpolation_level: float) -> np.ndarray:
    delta = xfrom - xto
    if percentile_vector.shape[-1] < delta.shape[-1]:
        percentile_vector = np.pad(percentile_vector,
                                pad_width=((0, 0), (0, 0), (0, delta.shape[-1] - percentile_vector.shape[-1])),
                                mode="edge")
    delta = percentile_vector * delta
    delta = interpolation_level * delta
    return xto + delta, np.count_nonzero(percentile_vector)

def get_reference_perturbation(xfrom, xto, explanation, filter_explanation_fn, **kwargs):
    first_mask = np.vectorize(filter_explanation_fn)
    masked_explanation = first_mask(explanation)
    percentile = kwargs['percentile_cut']
    percentile_value = np.percentile(np.abs(masked_explanation), percentile)
    percentile_mask = np.vectorize(lambda x: x if np.abs(x) > percentile_value else 0.0)
    percentile_vector = percentile_mask(masked_explanation)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])

def get_perturbations(X_target, X_references, X_explanations, explainer_method, policy='gaussian', **args):
    if policy == 'gaussian':
        if explainer_method == 'gradients' and 'y' in args:
            ## Here check that we (a) make a distinction between our explanations and the  explanations
            ## Get the right label to cut the gradient explanation
            y_factor = (2 * args['y'] - 1)[:, None, None] # shape (no_of_instances, 1, 1)
            X_adjusted_explanations = X_explanations * y_factor
            threshold = np.percentile(X_adjusted_explanations, args['percentile_cut'])
            X_e = X_adjusted_explanations
        else:
            threshold = np.percentile(X_explanations, args['percentile_cut'])
            X_e = X_explanations
        percentile_fn = lambda x: 1.0 if x > max(threshold, 0.0) else 0.0
        return get_gaussian_perturbation(X_target=X_target, X_to=X_references, explanation=X_e,
                                             filter_explanation_fn=percentile_fn,
                                             **args)
    elif policy == 'instance_to_reference':
        return get_reference_perturbation(xfrom=X_target, xto=X_references, explanation=X_explanations,
                                          filter_explanation_fn=lambda x : x if x>0.0 else 0.0, **args)
    elif policy == 'reference_to_instance':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x<0.0 else 0.0, **args)
    elif policy == 'reference_to_instance_positive':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x>0.0 else 0.0, **args)
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")
