from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from pertutils import get_signed_gaussian_perturbation_on_mask


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


if __name__ == "__main__":
    test_signed_gaussian_perturbation_follows_gradient_by_default()
    test_signed_gaussian_perturbation_can_move_against_gradient()
