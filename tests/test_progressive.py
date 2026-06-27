import numpy as np

from padic_quant.progressive import (
    build_progressive_model,
    reconstruct_layer,
)


def test_monna_prefixes_are_nested_and_error_decreases():
    weights = [np.array([[-1.0, -0.37, 0.0, 0.26, 0.91]])]
    model = build_progressive_model(weights, [np.zeros(1)], max_bits=8)
    errors = []
    for bits in range(2, 9):
        reconstructed = reconstruct_layer(
            model.layers[0], method="monna", bits=bits
        )
        errors.append(np.max(np.abs(reconstructed - weights[0])))
    assert all(right <= left + 1e-12 for left, right in zip(errors, errors[1:]))


def test_residual_binary_prefix_reconstruction_is_shared():
    rng = np.random.default_rng(4)
    weights = [rng.normal(size=(4, 5))]
    model = build_progressive_model(weights, [np.zeros(4)], max_bits=6)
    layer = model.layers[0]
    previous = np.zeros_like(weights[0])
    for bits in range(1, 7):
        current = reconstruct_layer(layer, method="residual_binary", bits=bits)
        expected = previous + layer.residual_scales[bits - 1] * layer.residual_planes[
            bits - 1
        ]
        np.testing.assert_allclose(current, expected)
        previous = current


def test_independent_one_bit_is_binary_scaled_sign():
    weights = [np.array([[-2.0, -1.0, 0.5, 1.5]])]
    model = build_progressive_model(weights, [np.zeros(1)], max_bits=2)
    reconstructed = reconstruct_layer(
        model.layers[0], method="independent", bits=1
    )
    assert set(np.unique(reconstructed)) == {-1.25, 1.25}

