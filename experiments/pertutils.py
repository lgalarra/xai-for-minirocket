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


def _random_positive_count_mask(explanation: np.ndarray, N, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        n_points = _n_perturbed_points_for_index(N, idx)
        if n_points <= 0:
            continue

        positive_indices = np.flatnonzero(explanation_i > 0.0)
        if positive_indices.size == 0:
            continue

        n_selected = min(n_points, positive_indices.size)
        selected = rng.choice(positive_indices, size=n_selected, replace=False)
        mask_i = np.zeros(explanation_i.size, dtype=bool)
        mask_i[selected] = True
        mask[idx] = mask_i.reshape(explanation_i.shape)
    return mask


def _reference_positive_mask(explanation: np.ndarray, percentile_cut: float) -> np.ndarray:
    masked_explanation = np.where(explanation > 0.0, explanation, 0.0)
    percentile_value = np.percentile(np.abs(masked_explanation), percentile_cut)
    return np.abs(masked_explanation) > percentile_value


def _selection_scores(explanation: np.ndarray, explainer_method: str) -> np.ndarray:
    if explainer_method == 'gradients':
        return np.abs(explanation)
    return explanation


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


def _bottom_unsigned_mask(explanation: np.ndarray, percentile_cut: float, n_perturbed_points=None) -> np.ndarray:
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        if n_perturbed_points is None:
            n_points = int(np.ceil(explanation_i.size * (100.0 - percentile_cut) / 100.0))
        else:
            n_points = _n_perturbed_points_for_index(n_perturbed_points, idx)
        if n_points <= 0:
            continue

        flat_scores = np.abs(explanation_i).ravel()
        n_selected = min(n_points, flat_scores.size)
        selected = np.argpartition(flat_scores, n_selected - 1)[:n_selected]
        mask_i = np.zeros(flat_scores.size, dtype=bool)
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


def _n_perturbed_points_for_index(n_perturbed_points, idx: int) -> int:
    counts = np.asarray(n_perturbed_points)
    if counts.ndim == 0:
        return int(counts)

    counts = counts.ravel()
    if counts.size == 0:
        return 0
    if counts.size == 1:
        return int(counts[0])
    if idx >= counts.size:
        raise ValueError(
            "n_perturbed_points must be a scalar or have one entry per instance "
            f"(got {counts.size} entries, needed index {idx})"
        )
    return int(counts[idx])


def _random_unconstrained_mask(explanation: np.ndarray, N, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    mask = np.zeros(explanation.shape, dtype=bool)
    for idx, explanation_i in enumerate(explanation):
        n_points = _n_perturbed_points_for_index(N, idx)
        if n_points <= 0:
            continue

        n_selected = min(n_points, explanation_i.size)
        selected = rng.choice(explanation_i.size, size=n_selected, replace=False)
        mask_i = np.zeros(explanation_i.size, dtype=bool)
        mask_i[selected] = True
        mask[idx] = mask_i.reshape(explanation_i.shape)
    return mask


def _mask_counts(mask: np.ndarray) -> np.ndarray:
    if mask.ndim <= 1:
        return np.array([np.count_nonzero(mask)])
    return np.count_nonzero(mask.reshape(mask.shape[0], -1), axis=1)


def _limit_mask(mask: np.ndarray, scores: np.ndarray, n_perturbed_points=None, rng=None,
                random_select: bool = False) -> np.ndarray:
    if n_perturbed_points is None:
        return mask

    if mask.ndim <= 1:
        masks = [np.asarray(mask)]
        scores_per_instance = [np.asarray(scores)]
        output_shape = mask.shape
    else:
        masks = mask
        scores_per_instance = scores
        output_shape = mask.shape

    if random_select:
        rng = np.random.default_rng(rng)

    limited_mask = np.zeros(output_shape, dtype=bool)
    for idx, mask_i in enumerate(masks):
        instance_n_perturbed_points = _n_perturbed_points_for_index(n_perturbed_points, idx)
        if instance_n_perturbed_points <= 0:
            if mask.ndim <= 1:
                return np.zeros(output_shape, dtype=bool)
            continue

        flat_mask = np.asarray(mask_i).ravel()
        selected_indices = np.flatnonzero(flat_mask)
        effective_n_perturbed_points = min(instance_n_perturbed_points, selected_indices.size)
        #print(
        #    f'Effective perturbed points: {effective_n_perturbed_points} per instance)'
        #    f'(requested: {n_perturbed_points}, available after percentile cut: {selected_indices.size})'
        #)
        if effective_n_perturbed_points == selected_indices.size:
            limited_mask_i = flat_mask
        elif random_select:
            kept_indices = rng.choice(selected_indices, size=effective_n_perturbed_points, replace=False)
            kept_indices.sort()
            limited_mask_i = np.zeros(flat_mask.size, dtype=bool)
            limited_mask_i[kept_indices] = True
        else:
            flat_scores = np.asarray(scores_per_instance[idx]).ravel()[selected_indices]
            score_order = np.argpartition(flat_scores, -effective_n_perturbed_points)[-effective_n_perturbed_points:]
            score_order.sort()
            kept_indices = selected_indices[score_order]
            limited_mask_i = np.zeros(flat_mask.size, dtype=bool)
            limited_mask_i[kept_indices] = True

        if mask.ndim <= 1:
            return limited_mask_i.reshape(output_shape)
        limited_mask[idx] = limited_mask_i.reshape(mask_i.shape)

    return limited_mask


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


def _rho_std_from_args(kwargs):
    if 'rho_std' in kwargs:
        return float(kwargs['rho_std'])

    sigma = float(kwargs.get('sigma', 0.25))
    if sigma > 1.0:
        # Preserve the legacy sigma grid: 3.0 corresponds to the proposed rho_std=0.25.
        return sigma / 12.0
    return sigma


def _sample_interpolation_coefficients(shape, kwargs):
    rho_mean = float(kwargs.get('rho_mean', kwargs.get('interpolation', 0.5)))
    rho_std = _rho_std_from_args(kwargs)
    rho = np.random.normal(rho_mean, rho_std, size=shape)
    return np.clip(rho, 0.0, 1.0)


def get_gaussian_perturbation(X_target: np.ndarray, X_to: np.ndarray, explanation: np.ndarray,
                              filter_explanation_fn: Callable,
                              **kwargs):
    padded_explanation = _pad_last_axis(explanation, X_target.shape[-1])
    percentile_mask = np.vectorize(filter_explanation_fn)
    explanation_mask = percentile_mask(padded_explanation)
    explanation_mask = _limit_mask(
        explanation_mask.astype(bool),
        padded_explanation,
        n_perturbed_points=kwargs.get('n_perturbed_points'),
        random_select=False
    ).astype(float)
    return get_gaussian_perturbation_on_mask(
        X_target=X_target,
        X_to=X_to,
        explanation_mask=explanation_mask,
        **kwargs
    )


def get_gaussian_perturbation_on_mask(X_target: np.ndarray, X_to: np.ndarray, explanation_mask: np.ndarray,
                                      **kwargs):
    budget = kwargs['budget']
    explanation_mask = _pad_last_axis(explanation_mask, X_target.shape[-1]).astype(float)

    if explanation_mask.shape[0] == X_target.shape[0]:
        repeated_mask = np.repeat(explanation_mask, budget, axis=0)
    elif explanation_mask.shape[0] == X_target.shape[0] * budget:
        repeated_mask = explanation_mask
    else:
        raise ValueError("explanation_mask must have one row per target instance or per budgeted perturbation")

    X_target_repeated = np.repeat(X_target, budget, axis=0)
    X_to_repeated = np.repeat(X_to, budget, axis=0)
    delta = X_to_repeated - X_target_repeated
    rho = _sample_interpolation_coefficients(repeated_mask.shape, kwargs)
    explanation_size = np.count_nonzero(repeated_mask) / budget
    return X_target_repeated + repeated_mask * rho * delta, explanation_size


def get_signed_gaussian_perturbation_on_mask(X_target: np.ndarray, explanation: np.ndarray,
                                             explanation_mask: np.ndarray, **kwargs):
    budget = kwargs['budget']
    signed_explanation = _pad_last_axis(explanation, X_target.shape[-1])
    explanation_mask = _pad_last_axis(explanation_mask, X_target.shape[-1]).astype(float)

    if explanation_mask.shape[0] == X_target.shape[0]:
        repeated_mask = np.repeat(explanation_mask, budget, axis=0)
        repeated_explanation = np.repeat(signed_explanation, budget, axis=0)
    elif explanation_mask.shape[0] == X_target.shape[0] * budget:
        repeated_mask = explanation_mask
        repeated_explanation = np.repeat(signed_explanation, budget, axis=0)
    else:
        raise ValueError("explanation_mask must have one row per target instance or per budgeted perturbation")

    X_target_repeated = np.repeat(X_target, budget, axis=0)
    sigma_multiplier = float(kwargs.get('signed_sigma', kwargs.get('sigma', 1.0)))
    observation_std = np.std(X_target, axis=tuple(range(1, X_target.ndim)), keepdims=True)
    perturbation_scale = np.repeat(observation_std, budget, axis=0) * sigma_multiplier
    perturbation = np.abs(np.random.normal(0.0, perturbation_scale, size=repeated_mask.shape))
    perturbation = np.sign(repeated_explanation) * perturbation
    explanation_size = np.count_nonzero(repeated_mask) / budget
    return X_target_repeated + repeated_mask * perturbation, explanation_size


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
    percentile_vector = _limit_mask(
        percentile_vector != 0.0,
        np.abs(percentile_vector),
        n_perturbed_points=kwargs.get('n_perturbed_points')
    ).astype(float)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])


def get_reference_perturbation_on_mask(xfrom, xto, explanation_mask, **kwargs):
    percentile_vector = np.where(explanation_mask, 1.0, 0.0)
    return apply_explanation_mask(xto, xfrom, percentile_vector, kwargs['interpolation'])


def get_random_reference_perturbation(xfrom, xto, explanation, unconstrained=False, **kwargs):
    attribution_mask = _reference_positive_mask(explanation, kwargs['percentile_cut'])
    attribution_mask = _limit_mask(
        attribution_mask,
        np.abs(explanation),
        n_perturbed_points=kwargs.get('n_perturbed_points')
    )
    matched_n_perturbed_points = _mask_counts(attribution_mask)
    mask_fn = _random_unconstrained_mask if unconstrained else _random_positive_count_mask
    masks = [
        np.where(
            _limit_mask(
                mask_fn(explanation, matched_n_perturbed_points),
                np.abs(explanation),
                n_perturbed_points=matched_n_perturbed_points,
                random_select=True
            ),
            1.0,
            0.0
        )
        for _ in range(kwargs['budget'])
    ]
    percentile_vectors = np.stack(masks, axis=1).reshape(
        explanation.shape[0] * kwargs['budget'], *explanation.shape[1:]
    )
    return apply_explanation_masks(xto, xfrom, percentile_vectors, kwargs['interpolation'])

def ensure_consistency(X: np.ndarray, X1: np.ndarray, X2: np.ndarray, return_kept_indices: bool = False):
    def row_length(row):
        row_array = np.asarray(row, dtype=object)
        if row_array.size == 0 or any(value is None for value in row_array.flat):
            return None
        nested_lengths = [
            len(value) for value in row_array.flat
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes))
        ]
        if nested_lengths:
            return nested_lengths[0]
        if row_array.ndim == 0:
            return None
        return row_array.shape[-1]

    row_lengths = [row_length(X[i]) for i in range(len(X))]
    lengths = set(length for length in row_lengths if length is not None)
    if not lengths:
        raise ValueError(
            "No valid explanation series found. Check that the selected reference policy has "
            "non-empty reference_* and beta_*_attributions entries in metadata.csv."
        )

    max_length = max(lengths)
    print(f'Ensuring consistency, all series must have size {max_length}')
    indices_to_remove = set([
        i for i, length in enumerate(row_lengths)
        if length is None or length != max_length
    ])
    print(indices_to_remove, lengths)
    kept_indices = np.array([idx for idx in range(len(X)) if idx not in indices_to_remove], dtype=int)
    X = np.array([x for idx, x in enumerate(X) if idx not in indices_to_remove])
    X1 = np.array([x for idx, x in enumerate(X1) if idx not in indices_to_remove])
    X2 = np.array([x for idx, x in enumerate(X2) if idx not in indices_to_remove])
    if return_kept_indices:
        return X, X1, X2, kept_indices
    return X, X1, X2

def get_perturbations(X_target, X_references, X_explanations, explainer_method, policy='gaussian', **args):
    X_scores = _selection_scores(X_explanations, explainer_method)
    if policy in ('gaussian', 'gaussian_bottom', 'gaussian_bottom_unsigned', 'gaussian_random',
                  'gaussian random', 'gaussian_random_no_positive'):
        threshold = np.percentile(X_scores, args['percentile_cut'])
        if policy == 'gaussian':
            percentile_fn = lambda x: 1.0 if x > max(threshold, 0.0) else 0.0
            return get_gaussian_perturbation(X_target=X_target, X_to=X_references, explanation=X_scores,
                                             filter_explanation_fn=percentile_fn,
                                             **args)
        elif policy == 'gaussian_bottom_unsigned':
            explanation_mask = _bottom_unsigned_mask(
                X_scores,
                args['percentile_cut'],
                args.get('n_perturbed_points')
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
        elif policy == 'gaussian_bottom':
            explanation_mask = _bottom_mask(X_scores, args['percentile_cut'])
            explanation_mask = _limit_mask(
                explanation_mask,
                -X_scores,
                n_perturbed_points=args.get('n_perturbed_points'),
                random_select=False
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
        elif policy in ('gaussian_random', 'gaussian random'):
            masks = [
                _limit_mask(
                    _random_mask(X_scores, args['percentile_cut']),
                    X_scores,
                    n_perturbed_points=args.get('n_perturbed_points'),
                    random_select=True
                )
                for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_scores.shape[0] * args['budget'], *X_scores.shape[1:]
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
        else:
            ## gaussian_random_no_positive
            top_positive_mask = (X_scores > max(threshold, 0.0))
            top_positive_mask = _limit_mask(
                top_positive_mask,
                X_scores,
                n_perturbed_points=args.get('n_perturbed_points'),
                random_select=False
            )
            matched_n_perturbed_points = _mask_counts(top_positive_mask)
            masks = [
                _limit_mask(
                    _random_unconstrained_mask(X_scores, matched_n_perturbed_points),
                    np.abs(X_scores),
                    n_perturbed_points=matched_n_perturbed_points,
                    random_select=True
                )
                for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_scores.shape[0] * args['budget'], *X_scores.shape[1:]
            )
            return get_gaussian_perturbation_on_mask(X_target=X_target, X_to=X_references,
                                                     explanation_mask=explanation_mask, **args)
    elif policy in ('gradient_gaussian', 'gradient_gaussian_bottom', 'gradient_gaussian_random',
                    'gradient_gaussian_random_no_positive'):
        threshold = np.percentile(X_scores, args['percentile_cut'])
        if policy == 'gradient_gaussian':
            explanation_mask = (X_scores > max(threshold, 0.0))
            explanation_mask = _limit_mask(
                explanation_mask,
                X_scores,
                n_perturbed_points=args.get('n_perturbed_points'),
                random_select=False
            )
        elif policy == 'gradient_gaussian_bottom':
            explanation_mask = _bottom_mask(X_scores, args['percentile_cut'])
            explanation_mask = _limit_mask(
                explanation_mask,
                -X_scores,
                n_perturbed_points=args.get('n_perturbed_points'),
                random_select=False
            )
        elif policy == 'gradient_gaussian_random':
            masks = [
                _limit_mask(
                    _random_mask(X_scores, args['percentile_cut']),
                    X_scores,
                    n_perturbed_points=args.get('n_perturbed_points'),
                    random_select=True
                )
                for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_scores.shape[0] * args['budget'], *X_scores.shape[1:]
            )
        else:
            top_attribution_mask = (X_scores > max(threshold, 0.0))
            top_attribution_mask = _limit_mask(
                top_attribution_mask,
                X_scores,
                n_perturbed_points=args.get('n_perturbed_points'),
                random_select=False
            )
            matched_n_perturbed_points = _mask_counts(top_attribution_mask)
            masks = [
                _limit_mask(
                    _random_unconstrained_mask(X_scores, matched_n_perturbed_points),
                    np.abs(X_scores),
                    n_perturbed_points=matched_n_perturbed_points,
                    random_select=True
                )
                for _ in range(args['budget'])
            ]
            explanation_mask = np.stack(masks, axis=1).reshape(
                X_scores.shape[0] * args['budget'], *X_scores.shape[1:]
            )
        return get_signed_gaussian_perturbation_on_mask(
            X_target=X_target,
            explanation=X_explanations,
            explanation_mask=explanation_mask,
            **args
        )
    elif policy == 'reference_to_instance':
        return get_reference_perturbation(xfrom=X_target, xto=X_references, explanation=X_scores,
                                          filter_explanation_fn=lambda x : x if x>0.0 else 0.0, **args)
    elif policy == 'reference_to_instance_unsigned':
        explanation_mask = _bottom_unsigned_mask(
            X_scores,
            args['percentile_cut'],
            args.get('n_perturbed_points')
        )
        return get_reference_perturbation_on_mask(xfrom=X_target, xto=X_references,
                                                  explanation_mask=explanation_mask, **args)
    elif policy == 'reference_to_instance_bottom':
        explanation_mask = _bottom_mask(X_scores, args['percentile_cut'])
        explanation_mask = _limit_mask(
            explanation_mask,
            -X_scores,
            n_perturbed_points=args.get('n_perturbed_points')
        )
        return get_reference_perturbation_on_mask(xfrom=X_target, xto=X_references,
                                                  explanation_mask=explanation_mask, **args)
    elif policy == 'reference_to_instance_random':
        return get_random_reference_perturbation(xfrom=X_target, xto=X_references,
                                                explanation=X_scores, **args)
    elif policy == 'reference_to_instance_random_no_positive':
        return get_random_reference_perturbation(xfrom=X_target, xto=X_references,
                                                explanation=X_scores, unconstrained=True, **args)
    elif policy == 'instance_to_reference':
        return get_reference_perturbation(xfrom=X_references, xto=X_target, explanation=X_scores,
                                          filter_explanation_fn=lambda x: x if x>0.0 else 0.0, **args)
    elif policy == 'instance_to_reference_bottom_unsigned':
        explanation_mask = _bottom_unsigned_mask(
            X_scores,
            args['percentile_cut'],
            args.get('n_perturbed_points')
        )
        return get_reference_perturbation_on_mask(xfrom=X_references, xto=X_target,
                                                  explanation_mask=explanation_mask, **args)
    elif policy == 'instance_to_reference_bottom':
        explanation_mask = _bottom_mask(X_scores, args['percentile_cut'])
        explanation_mask = _limit_mask(
            explanation_mask,
            -X_scores,
            n_perturbed_points=args.get('n_perturbed_points')
        )
        return get_reference_perturbation_on_mask(xfrom=X_references, xto=X_target,
                                                  explanation_mask=explanation_mask, **args)
    elif policy == 'instance_to_reference_random':
        return get_random_reference_perturbation(xfrom=X_references, xto=X_target,
                                                explanation=X_scores, **args)
    elif policy == 'instance_to_reference_random_no_positive':
        return get_random_reference_perturbation(xfrom=X_references, xto=X_target,
                                                explanation=X_scores, unconstrained=True, **args)
    else:
        raise ValueError(f"Unknown perturbation policy {policy}")
