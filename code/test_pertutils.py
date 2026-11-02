from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from pertutils import (
    _matched_bottom_unsigned_mask,
    get_perturbations,
    get_signed_gaussian_perturbation_on_mask,
)


def test_signed_gaussian_perturbation_follows_gradient_by_default():
    def fake_normal(loc, scale, size):
        return np.ones(size)

    x = np.array([[[10.0, 10.0, 10.0, 10.0]]])
    gradients = np.array([[[2.0, -3.0, 0.0, 4.0]]])
    mask = np.array([[[True, True, True, False]]])

    with patch.object(np.random, "normal", fake_normal):
        perturbed, n_points = get_signed_gaussian_perturbation_on_mask(
            x,
            gradients,
            mask,
            budget=1,
            sigma=1.0,
        )

    np.testing.assert_allclose(perturbed, np.array([[[11.0, 9.0, 10.0, 10.0]]]))
    assert n_points == 3.0


def test_signed_gaussian_perturbation_can_move_against_gradient():
    def fake_normal(loc, scale, size):
        return np.ones(size)

    x = np.array([[[10.0, 10.0, 10.0, 10.0]]])
    gradients = np.array([[[2.0, -3.0, 0.0, 4.0]]])
    mask = np.array([[[True, True, True, False]]])

    with patch.object(np.random, "normal", fake_normal):
        perturbed, n_points = get_signed_gaussian_perturbation_on_mask(
            x,
            gradients,
            mask,
            budget=1,
            sigma=1.0,
            opposite_gradient=True,
        )

    np.testing.assert_allclose(perturbed, np.array([[[9.0, 11.0, 10.0, 10.0]]]))
    assert n_points == 3.0


def test_gaussian_bottom_unsigned_matches_top_positive_mask_size():
    target = np.zeros((2, 1, 10))
    reference = np.ones((2, 1, 10))
    explanations = np.array(
        [
            [[2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]],
            [[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0]],
        ]
    )
    args = {
        "percentile_cut": 10,
        "n_perturbed_points": np.array([9, 9]),
        "budget": 1,
        "sigma": 3.0,
    }
    bottom_mask = _matched_bottom_unsigned_mask(
        explanations,
        args["percentile_cut"],
        args["n_perturbed_points"],
    )

    _, top_count = get_perturbations(
        target,
        reference,
        explanations,
        explainer_method="shap",
        policy="gaussian",
        **args,
    )
    _, bottom_count = get_perturbations(
        target,
        reference,
        explanations,
        explainer_method="shap",
        policy="gaussian_bottom_unsigned",
        **args,
    )

    np.testing.assert_array_equal(bottom_mask.sum(axis=(1, 2)), [2, 8])
    assert top_count == 10
    assert bottom_count == top_count


def test_reference_bottom_unsigned_policies_match_top_positive_mask_size():
    target = np.zeros((2, 1, 10))
    reference = np.ones((2, 1, 10))
    explanations = np.array(
        [
            [[2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]],
            [[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0]],
        ]
    )
    args = {
        "percentile_cut": 10,
        "n_perturbed_points": np.array([9, 9]),
        "budget": 1,
        "interpolation": 1.0,
    }

    policy_pairs = (
        ("instance_to_reference", "instance_to_reference_bottom_unsigned"),
        ("reference_to_instance", "reference_to_instance_unsigned"),
    )
    for top_policy, bottom_policy in policy_pairs:
        _, top_count = get_perturbations(
            target,
            reference,
            explanations,
            explainer_method="shap",
            policy=top_policy,
            **args,
        )
        _, bottom_count = get_perturbations(
            target,
            reference,
            explanations,
            explainer_method="shap",
            policy=bottom_policy,
            **args,
        )

        assert top_count == 10
        assert bottom_count == top_count


if __name__ == "__main__":
    test_signed_gaussian_perturbation_follows_gradient_by_default()
    test_signed_gaussian_perturbation_can_move_against_gradient()
    test_gaussian_bottom_unsigned_matches_top_positive_mask_size()
    test_reference_bottom_unsigned_policies_match_top_positive_mask_size()
