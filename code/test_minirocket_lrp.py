from __future__ import annotations

import numpy as np
import pytest

from minirocket_multivariate_variable import (
    _minirocket_lrp_denominator,
    _minirocket_lrp_reconstruct_conv_at,
    _minirocket_lrp_source_index,
    back_propagate_attribution_lrp,
    transform_prime,
)


def _trace(conv_sum, bias_b, dilation=1, channels=(0,), kernel=None):
    if kernel is None:
        kernel = np.array([-1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0])
    return {
        "conv_sum": np.asarray(conv_sum, dtype=np.float64),
        "bias_b": float(bias_b),
        "dilation": int(dilation),
        "channels": list(channels),
        "kernel": np.asarray(kernel, dtype=np.float64),
    }


def _manual_lrp_feature(x_tc, tr, alpha, epsilon=0.0, stabilizer="paper"):
    x_tc = np.asarray(x_tc, dtype=np.float64)
    T, C = x_tc.shape
    out = np.zeros((T, C), dtype=np.float64)
    active = np.flatnonzero(np.asarray(tr["conv_sum"]) > float(tr["bias_b"]))
    if alpha == 0.0 or active.size == 0:
        return out

    kappa = np.asarray(tr["kernel"], dtype=np.float64)
    share = float(alpha) / active.size
    for j in active:
        terms = []
        z = 0.0
        for m, tap in enumerate(kappa):
            src = _minirocket_lrp_source_index(j, m, tr["dilation"], T)
            if src is None:
                continue
            for ch in tr["channels"]:
                value = x_tc[src, ch] * tap
                terms.append((src, ch, value))
                z += value
        denominator = _minirocket_lrp_denominator(z, epsilon, stabilizer)
        if denominator == 0.0:
            continue
        for src, ch, value in terms:
            out[src, ch] += value / denominator * share
    return out


def test_single_feature_single_active_convolution_manual():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    tr = _trace(
        conv_sum=[0, 0, 0, 0, 5, 0, 0, 0, 0],
        bias_b=1.0,
        kernel=np.ones(9, dtype=np.float64),
    )
    alpha = 3.0

    beta = back_propagate_attribution_lrp(
        np.array([alpha]), [tr], x, epsilon=0.0, per_channel=True, n_jobs=1
    )

    expected = _manual_lrp_feature(x, tr, alpha).T
    np.testing.assert_allclose(beta, expected)
    np.testing.assert_allclose(beta.sum(), alpha)


def test_ppv_redistribution_is_equal_over_active_positions():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    kernel = np.ones(9, dtype=np.float64)
    tr_one = _trace(conv_sum=[0, 2, 0, 0, 100, 0, 0, 0, 0], bias_b=1.0, kernel=kernel)
    tr_two = _trace(conv_sum=[0, 100, 0, 0, 2, 0, 0, 0, 0], bias_b=1.0, kernel=kernel)
    alpha = 6.0

    beta_one = back_propagate_attribution_lrp(
        np.array([alpha]), [tr_one], x, epsilon=0.0, per_channel=True, n_jobs=1
    )
    beta_two = back_propagate_attribution_lrp(
        np.array([alpha]), [tr_two], x, epsilon=0.0, per_channel=True, n_jobs=1
    )

    np.testing.assert_allclose(beta_one, _manual_lrp_feature(x, tr_one, alpha).T)
    np.testing.assert_allclose(beta_two, _manual_lrp_feature(x, tr_two, alpha).T)
    np.testing.assert_allclose(beta_one, beta_two)
    np.testing.assert_allclose(beta_one.sum(), alpha)


def test_dilation_uses_only_nine_dilated_tap_locations():
    x = np.ones((21, 1), dtype=np.float64)
    tr = _trace(
        conv_sum=np.r_[np.zeros(10), 1.0, np.zeros(10)],
        bias_b=0.5,
        dilation=2,
        kernel=np.ones(9, dtype=np.float64),
    )

    beta = back_propagate_attribution_lrp(
        np.array([9.0]), [tr], x, epsilon=0.0, per_channel=True, n_jobs=1
    )[0]

    expected_indices = np.arange(2, 19, 2)
    assert set(np.flatnonzero(beta)) == set(expected_indices)
    np.testing.assert_allclose(beta[expected_indices], np.ones(9))


def test_multivariate_channels_use_signed_contributions_not_equal_split():
    x = np.zeros((9, 3), dtype=np.float64)
    x[4, 0] = 1.0
    x[4, 2] = 3.0
    tr = _trace(
        conv_sum=[0, 0, 0, 0, 1, 0, 0, 0, 0],
        bias_b=0.0,
        channels=(0, 2),
        kernel=np.r_[np.zeros(4), 2.0, np.zeros(4)],
    )

    beta = back_propagate_attribution_lrp(
        np.array([8.0]), [tr], x, epsilon=0.0, per_channel=True, n_jobs=1
    )

    assert np.all(beta[1] == 0.0)
    np.testing.assert_allclose(beta[0, 4], 2.0)
    np.testing.assert_allclose(beta[2, 4], 6.0)


def test_signed_negative_contributions_are_preserved():
    x = np.zeros((9, 1), dtype=np.float64)
    x[4, 0] = 2.0
    x[5, 0] = 1.0
    tr = _trace(
        conv_sum=[0, 0, 0, 0, 1, 0, 0, 0, 0],
        bias_b=0.0,
        kernel=np.r_[np.zeros(4), 2.0, -1.0, np.zeros(3)],
    )

    beta = back_propagate_attribution_lrp(
        np.array([3.0]), [tr], x, epsilon=0.0, per_channel=True, n_jobs=1
    )

    assert beta[0, 4] > 0.0
    assert beta[0, 5] < 0.0
    np.testing.assert_allclose(beta.sum(), 3.0)


def test_conservation_excludes_zero_active_and_zero_alpha_features():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    active = _trace(
        conv_sum=[0, 0, 0, 0, 2, 0, 0, 0, 0],
        bias_b=1.0,
        kernel=np.ones(9, dtype=np.float64),
    )
    zero_alpha = _trace(conv_sum=[0, 2, 0, 0, 0, 0, 0, 0, 0], bias_b=1.0)
    zero_active = _trace(conv_sum=np.zeros(9), bias_b=1.0)
    alphas = np.array([4.0, 0.0, 5.0])

    beta = back_propagate_attribution_lrp(
        alphas, [active, zero_alpha, zero_active], x, epsilon=0.0, n_jobs=1
    )

    assert not np.isnan(beta).any()
    np.testing.assert_allclose(beta.sum(), 4.0)


def test_boundary_padding_omits_invalid_taps():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    tr = _trace(
        conv_sum=[1, 0, 0, 0, 0, 0, 0, 0, 0],
        bias_b=0.0,
        kernel=np.ones(9, dtype=np.float64),
    )

    beta = back_propagate_attribution_lrp(
        np.array([1.0]), [tr], x, epsilon=0.0, per_channel=True, n_jobs=1
    )[0]

    assert set(np.flatnonzero(beta)) == {0, 1, 2, 3, 4}
    np.testing.assert_allclose(beta[5:], 0.0)
    np.testing.assert_allclose(beta.sum(), 1.0)


def test_paper_stabilizer_uses_z_plus_epsilon():
    x = np.zeros((9, 1), dtype=np.float64)
    x[4, 0] = 2.0
    tr = _trace(
        conv_sum=[0, 0, 0, 0, 1, 0, 0, 0, 0],
        bias_b=0.0,
        kernel=np.r_[np.zeros(4), 3.0, np.zeros(4)],
    )

    beta = back_propagate_attribution_lrp(
        np.array([10.0]), [tr], x, epsilon=2.0, stabilizer="paper", per_channel=True, n_jobs=1
    )

    np.testing.assert_allclose(beta[0, 4], 6.0 / (6.0 + 2.0) * 10.0)


def test_signed_stabilizer_positive_negative_and_zero_denominators():
    assert _minirocket_lrp_denominator(2.0, 0.5, "signed") == 2.5
    assert _minirocket_lrp_denominator(-2.0, 0.5, "signed") == -2.5
    assert _minirocket_lrp_denominator(0.0, 0.5, "signed") == 0.5


def test_negative_denominator_distinguishes_paper_and_signed():
    x = np.zeros((9, 1), dtype=np.float64)
    x[4, 0] = 2.0
    tr = _trace(
        conv_sum=[0, 0, 0, 0, 1, 0, 0, 0, 0],
        bias_b=0.0,
        kernel=np.r_[np.zeros(4), -3.0, np.zeros(4)],
    )

    beta_paper = back_propagate_attribution_lrp(
        np.array([10.0]), [tr], x, epsilon=1.0, stabilizer="paper", per_channel=True, n_jobs=1
    )
    beta_signed = back_propagate_attribution_lrp(
        np.array([10.0]), [tr], x, epsilon=1.0, stabilizer="signed", per_channel=True, n_jobs=1
    )

    np.testing.assert_allclose(beta_paper[0, 4], -6.0 / (-6.0 + 1.0) * 10.0)
    np.testing.assert_allclose(beta_signed[0, 4], -6.0 / (-6.0 - 1.0) * 10.0)
    assert beta_paper[0, 4] != beta_signed[0, 4]


def test_paper_and_signed_are_equivalent_when_epsilon_is_zero():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    tr = _trace(conv_sum=[0, 0, 0, 0, 3, 0, 0, 0, 0], bias_b=1.0)

    paper = back_propagate_attribution_lrp(
        np.array([2.0]), [tr], x, epsilon=0.0, stabilizer="paper", per_channel=True, n_jobs=1
    )
    signed = back_propagate_attribution_lrp(
        np.array([2.0]), [tr], x, epsilon=0.0, stabilizer="signed", per_channel=True, n_jobs=1
    )

    np.testing.assert_allclose(paper, signed)


def test_invalid_stabilizer_raises_clear_value_error():
    x = np.arange(1.0, 10.0).reshape(-1, 1)
    tr = _trace(conv_sum=[0, 0, 0, 0, 3, 0, 0, 0, 0], bias_b=1.0)

    with pytest.raises(ValueError, match="Unsupported stabilizer"):
        back_propagate_attribution_lrp(np.array([1.0]), [tr], x, stabilizer="bad", n_jobs=1)


def test_reconstructed_convolution_matches_transform_prime_trace():
    x = np.array(
        [[[1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0],
          [2.0, 1.0, -1.0, 0.5, 3.0, -2.0, 4.0]]],
        dtype=np.float32,
    )
    num_channels_per_combination = np.full(84, 2, dtype=np.int32)
    channel_indices = np.tile(np.array([0, 1], dtype=np.int32), 84)
    parameters = (
        num_channels_per_combination,
        channel_indices,
        np.array([2], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.zeros(84, dtype=np.float32),
        False,
    )

    out = transform_prime(x, parameters=parameters)
    tr = out["traces"][0]
    x_tc = x[0].T

    reconstructed = np.array([
        _minirocket_lrp_reconstruct_conv_at(x_tc, tr, j)
        for j in range(x_tc.shape[0])
    ])

    np.testing.assert_allclose(reconstructed, tr["conv_sum"], rtol=1e-6, atol=1e-6)
