import torch
import torch.nn.functional as F

from dyadic_quant.level1 import encode_tensor_per_output_channel
from dyadic_quant.level2.native import (
    build_native_cpu,
    dyadic_qwen_mlp_stack_plan_native_cpu,
    dyadic_qwen_qkv_plan_native_cpu,
    pack_native_cpu_weight,
    pack_qwen_mlp_stack_native_cpu,
    pack_qwen_qkv_native_cpu,
)


def _pack(weight: torch.Tensor, bits: int):
    encoded = encode_tensor_per_output_channel(weight, max_bits=8)
    return (
        pack_native_cpu_weight(
            encoded.signs,
            encoded.magnitude_code,
            encoded.exponents,
            encoded.max_bits,
            encoded.group_size,
            bits,
        ),
        encoded.decode(bits),
    )


def test_planned_qwen_mlp_stack_matches_decoded_reference():
    build_native_cpu(force=True)
    torch.manual_seed(101)
    bits = 6
    inputs = torch.randn(3, 8)
    packed_blocks = []
    decoded_blocks = []
    for _ in range(2):
        gate_packed, gate_decoded = _pack(torch.randn(12, 8), bits)
        up_packed, up_decoded = _pack(torch.randn(12, 8), bits)
        down_packed, down_decoded = _pack(torch.randn(8, 12), bits)
        gate_bias = torch.randn(12)
        up_bias = torch.randn(12)
        down_bias = torch.randn(8)
        packed_blocks.append(
            (gate_packed, up_packed, down_packed, gate_bias, up_bias, down_bias)
        )
        decoded_blocks.append(
            (gate_decoded, up_decoded, down_decoded, gate_bias, up_bias, down_bias)
        )

    expected = inputs
    for gate, up, down, gate_bias, up_bias, down_bias in decoded_blocks:
        expected = F.linear(
            F.silu(F.linear(expected, gate, gate_bias))
            * F.linear(expected, up, up_bias),
            down,
            down_bias,
        )

    plan = pack_qwen_mlp_stack_native_cpu(packed_blocks)
    actual = dyadic_qwen_mlp_stack_plan_native_cpu(inputs, plan)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_planned_qwen_qkv_matches_decoded_reference():
    build_native_cpu(force=True)
    torch.manual_seed(102)
    bits = 6
    inputs = torch.randn(5, 8)
    q_packed, q_decoded = _pack(torch.randn(8, 8), bits)
    k_packed, k_decoded = _pack(torch.randn(4, 8), bits)
    v_packed, v_decoded = _pack(torch.randn(4, 8), bits)
    q_bias = torch.randn(8)
    k_bias = torch.randn(4)
    v_bias = torch.randn(4)

    plan = pack_qwen_qkv_native_cpu(
        q_packed,
        k_packed,
        v_packed,
        q_bias,
        k_bias,
        v_bias,
    )
    actual_q, actual_k, actual_v = dyadic_qwen_qkv_plan_native_cpu(inputs, plan)

    torch.testing.assert_close(
        actual_q, F.linear(inputs, q_decoded, q_bias), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        actual_k, F.linear(inputs, k_decoded, k_bias), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        actual_v, F.linear(inputs, v_decoded, v_bias), rtol=1e-5, atol=1e-5
    )
