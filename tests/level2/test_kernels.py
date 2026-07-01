import torch
import torch.nn.functional as F

from dyadic_quant.level1 import encode_tensor_per_output_channel
from dyadic_quant.level2 import (
    dyadic_activation_matmul,
    dyadic_conv2d,
    dyadic_embedding,
    dyadic_gemm,
    dyadic_gemv,
    dyadic_linear,
    dyadic_output_projection,
)


def test_level2_linear_matches_level1_materialized_baseline():
    torch.manual_seed(1)
    weight = torch.randn(5, 7)
    bias = torch.randn(5)
    inputs = torch.randn(2, 3, 7)
    encoded = encode_tensor_per_output_channel(
        weight,
        max_bits=8,
        optimize_prefix_bits=(4, 6, 8),
        group_size=4,
    )

    actual = dyadic_linear(inputs, encoded, bias=bias, bits=6)
    expected = F.linear(inputs, encoded.decode(6), bias)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_gemv_and_gemm_match_level1_materialized_baseline():
    torch.manual_seed(2)
    weight = torch.randn(4, 6)
    bias = torch.randn(4)
    vector = torch.randn(6)
    matrix = torch.randn(3, 6)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=3)

    torch.testing.assert_close(
        dyadic_gemv(vector, encoded, bias=bias, bits=5),
        F.linear(vector, encoded.decode(5), bias),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        dyadic_gemm(matrix, encoded, bias=bias, bits=5),
        F.linear(matrix, encoded.decode(5), bias),
        rtol=1e-6,
        atol=1e-6,
    )


def test_level2_embedding_matches_level1_materialized_baseline():
    torch.manual_seed(3)
    weight = torch.randn(11, 6)
    indices = torch.tensor([[0, 3, 5], [10, 2, 3]])
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=4)

    actual = dyadic_embedding(indices, encoded, bits=4)
    expected = F.embedding(indices, encoded.decode(4))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_output_projection_matches_level1_materialized_baseline():
    torch.manual_seed(33)
    hidden = torch.randn(2, 4)
    lm_head = torch.randn(9, 4)
    encoded = encode_tensor_per_output_channel(lm_head, max_bits=8, group_size=2)

    actual = dyadic_output_projection(hidden, encoded, bits=6)
    expected = F.linear(hidden, encoded.decode(6))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_conv2d_matches_level1_materialized_baseline():
    torch.manual_seed(4)
    inputs = torch.randn(2, 3, 8, 7)
    weight = torch.randn(5, 3, 3, 2)
    bias = torch.randn(5)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=9)

    actual = dyadic_conv2d(
        inputs,
        encoded,
        bias=bias,
        bits=6,
        stride=(2, 1),
        padding=(1, 0),
    )
    expected = F.conv2d(
        inputs,
        encoded.decode(6),
        bias=bias,
        stride=(2, 1),
        padding=(1, 0),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_grouped_conv2d_matches_level1_materialized_baseline():
    torch.manual_seed(5)
    inputs = torch.randn(1, 4, 7, 7)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=6)

    actual = dyadic_conv2d(
        inputs,
        encoded,
        bias=bias,
        bits=5,
        padding=1,
        groups=2,
    )
    expected = F.conv2d(
        inputs,
        encoded.decode(5),
        bias=bias,
        padding=1,
        groups=2,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_activation_matmul_matches_level1_materialized_baseline():
    torch.manual_seed(6)
    left = torch.randn(3, 5)
    right = torch.randn(5, 4)
    encoded_left = encode_tensor_per_output_channel(
        left,
        max_bits=8,
        optimize_prefix_bits=(5, 7),
        group_size=3,
    )
    encoded_right_t = encode_tensor_per_output_channel(
        right.T,
        max_bits=8,
        optimize_prefix_bits=(5, 7),
        group_size=3,
    )

    actual = dyadic_activation_matmul(
        encoded_left,
        encoded_right_t,
        left_bits=5,
        right_bits=7,
    )
    expected = encoded_left.decode(5) @ encoded_right_t.decode(7).T

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
