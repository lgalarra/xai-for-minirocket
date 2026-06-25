from typing import Callable

import numpy as np


def _pad_last_axis(X: np.ndarray, target_length: int) -> np.ndarray:
    if X.shape[-1] >= target_length:
        return X
    return np.pad(X, pad_width=((0, 0), (0, 0), (0, target_length - X.shape[-1])), mode="edge")


def _top_positive_mask(explanation: np.ndarray, percentile_cut: float) -> np.ndarray:
    masked_explanation = np.where(explanation > 0.0, explanation, 0.0)
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(masked_explanation):
        percentile_value = np.percentile(np.abs(explanation_i), percentile_cut)
        mask[idx] = np.abs(explanation_i) > percentile_value
    return mask


def _random_positive_mask(explanation: np.ndarray, percentile_cut: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    top_mask = _top_positive_mask(explanation, percentile_cut)
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        top_count = np.count_nonzero(top_mask[idx])
        positive_indices = np.flatnonzero(explanation_i > 0.0)
        if top_count == 0 or positive_indices.size == 0:
            continue

        n_selected = min(top_count, positive_indices.size)
        selected = rng.choice(positive_indices, size=n_selected, replace=False)
        mask_i = np.zeros(explanation_i.size, dtype=bool)
        mask_i[selected] = True
        mask[idx] = mask_i.reshape(explanation_i.shape)
    return mask


def _bottom_mask(explanation: np.ndarray, percentile_cut: float) -> np.ndarray:
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        threshold = np.percentile(explanation_i, 100 - percentile_cut)
        top_count = np.count_nonzero(explanation_i < threshold)
        if top_count == 0:
            continue

        flat_explanation = explanation_i.ravel()
        n_selected = min(top_count, flat_explanation.size)
        selected = np.argpartition(flat_explanation, n_selected - 1)
        selected = selected[:n_selected]
        mask_i = np.zeros(flat_explanation.size, dtype=bool)
        mask_i[selected] = True
        mask[idx] = mask_i.reshape(explanation_i.shape)
    return mask


def _random_mask(explanation: np.ndarray, percentile_cut: float, rng=None) -> np.ndarray:
    top_count = np.count_nonzero(explanation > max(np.percentile(explanation, percentile_cut), 0.0))
    if top_count == 0:
        return np.zeros(explanation.shape, dtype=bool)

    rng = np.random.default_rng(rng)
    n_selected = min(top_count, explanation.size)
    selected = rng.choice(explanation.size, size=n_selected, replace=False)
    mask = np.zeros(explanation.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(explanation.shape)


def _random_unconstrained_mask(explanation: np.ndarray, percentile_cut: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        threshold = np.percentile(explanation_i, percentile_cut)
        top_count = np.count_nonzero(explanation_i > threshold)
        if top_count == 0:
            continue

        n_selected = min(top_count, explanation_i.size)
        selected = rng.choice(explanation_i.size, size=n_selected, replace=False)
        mask_i = np.zeros(explanation_i.size, dtype=bool)
        mask_i[selected] = True
        mask[idx] = mask_i.reshape(explanation_i.shape)
    return mask

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
    padded_explanation = _pad_last_axis(explanation, X_target.shape[-1])
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


def get_gaussian_perturbation_on_mask(X_target: np.ndarray, X_to: np.ndarray, explanation_mask: np.ndarray,
                                      **kwargs):
    budget = kwargs['budget']
    explanation_mask = _pad_last_axis(explanation_mask, X_target.shape[-1]).astype(float)
    if X_target.shape[0] == 1:
        std = (X_to - X_target).std()
        avg = (X_to - X_target).mean()
    else:
        std = (X_to - X_target).std(axis=0)
        avg = (X_to - X_target).mean(axis=0)

    if explanation_mask.shape[0] == X_target.shape[0]:
        repeated_mask = np.repeat(explanation_mask, budget, axis=0)
    elif explanation_mask.shape[0] == X_target.shape[0] * budget:
        repeated_mask = explanation_mask
    else:
        raise ValueError("explanation_mask must have one row per target instance or per budgeted perturbation")

    X_perturb = np.random.normal(avg, kwargs['sigma'] * std, size=repeated_mask.shape)
    explanation_size = np.count_nonzero(repeated_mask) / budget
    return np.repeat(X_target, budget, axis=0) + repeated_mask * X_perturb, explanation_size


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


def apply_explanation_masks(xto: np.ndarray, xfrom: np.ndarray,
                            percentile_vectors: np.ndarray, interpolation_level: float) -> np.ndarray:
    percentile_vectors = _pad_last_axis(percentile_vectors, xfrom.shape[-1])
    budget = int(percentile_vectors.shape[0] / xfrom.shape[0])
    xfrom_repeated = np.repeat(xfrom, budget, axis=0)
    xto_repeated = np.repeat(xto, budget, axis=0)
    delta = percentile_vectors * (xfrom_repeated - xto_repeated)
    delta = interpolation_level * delta
    return xto_repeated + delta, np.count_nonzero(percentile_vectors) / budget

def get_reference_perturbation(xfrom, xto, explanation, filter_explanation_fn, **kwargs):
    first_mask = np.vectorize(filter_explanation_fn)
    masked_explanation = first_mask(explanation)
    percentile = kwargs['percentile_cut']
    percentile_value = np.percentile(np.abs(masked_explanation), percentile)
    percentile_mask = np.vectorize(lambda x: x if np.abs(x) > percentile_value else 0.0)
    percentile_vector = percentile_mask(masked_explanation)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])


def get_reference_perturbation_on_mask(xfrom, xto, explanation_mask, **kwargs):
    percentile_vector = np.where(explanation_mask, 1.0, 0.0)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])


def get_random_reference_perturbation(xfrom, xto, explanation, unconstrained=False, **kwargs):
    mask_fn = _random_unconstrained_mask if unconstrained else _random_positive_mask
    masks = [
        np.where(mask_fn(explanation, kwargs['percentile_cut']), 1.0, 0.0)
        for _ in range(kwargs['budget'])
    ]
    percentile_vectors = np.stack(masks, axis=1).reshape(
        explanation.shape[0] * kwargs['budget'], *explanation.shape[1:]
    )
    return apply_explanation_masks(xto, xfrom, percentile_vectors, kwargs['interpolation'])

def ensure_consistency(X: np.ndarray, X1: np.ndarray, X2: np.ndarray):
    lengths = set([len(X[i][0]) for i in range(len(X)) if hasattr(X[i][0], "__len__")])
    max_length = max(lengths)
    print(f'Ensuring consistency, all series must have size {max_length}')
    indices_to_remove = set([i for i in range(len(X)) if hasattr(X[i][0], "__len__") and len(X[i][0]) != max_length])
    print(indices_to_remove, lengths)
    X = np.array([x for x in X if hasattr(x[0], "__len__") and len(x[0]) == max_length])
    X1 = np.array([x for idx, x in enumerate(X1) if idx not in indices_to_remove])
    X2 = np.array([x for idx, x in enumerate(X2) if idx not in indices_to_remove])
    return X, X1, X2

def get_perturbations(X_target, X_references, X_explanations, explainer_method, policy='gaussian', **args):
    if policy in ('gaussian', 'gaussian_bottom', 'gaussian_random', 'gaussian random',
                  'gaussian_random_no_positive'):
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
        if policy == 'gaussian':
            percentile_fn = lambda x: 1.0 if x > max(threshold, 0.0) else 0.0
            return get_gaussian_perturbation(X_target=X_target, X_to=X_references, explanation=X_e,
                                             filter_explanation_fn=percentile_fn,
                                             **args)
        elif policy == 'gaussian_bottom':
            explanation_mask = _bottom_mask(X_e, args['percentile_cut'])
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
        elif policy in ('gaussian_random', 'gaussian random'):
            masks = [
                _random_mask(X_e, args['percentile_cut']) for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_e.shape[0] * args['budget'], *X_e.shape[1:]
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
        else:
            masks = [
                _random_unconstrained_mask(X_e, args['percentile_cut']) for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_e.shape[0] * args['budget'], *X_e.shape[1:]
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
    elif policy == 'instance_to_reference':
        return get_reference_perturbation(xfrom=X_target, xto=X_references, explanation=X_explanations,
                                          filter_explanation_fn=lambda x : x if x>0.0 else 0.0, **args)
    elif policy == 'instance_to_reference_bottom':
        explanation_mask = _bottom_mask(X_explanations, args['percentile_cut'])
        return get_reference_perturbation_on_mask(xfrom=X_target, xto=X_references,
                                                  explanation_mask=explanation_mask, **args)
    elif policy == 'instance_to_reference_random':
        return get_random_reference_perturbation(xfrom=X_target, xto=X_references,
                                                explanation=X_explanations, **args)
    elif policy == 'instance_to_reference_random_no_positive':
        return get_random_reference_perturbation(xfrom=X_target, xto=X_references,
                                                explanation=X_explanations, unconstrained=True, **args)
    elif policy == 'reference_to_instance':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x<0.0 else 0.0, **args)
    elif policy == 'reference_to_instance_positive':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_explanations,
                                          filter_explanation_fn=lambda x: x if x>0.0 else 0.0, **args)
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")
