from __future__ import annotations

import torch
import torch.nn.functional as F

from dyadic_quant.level1 import DyadicTensor


def dyadic_linear(
    inputs: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=inputs.dtype, device=inputs.device)
    return F.linear(inputs, weight, bias)


def dyadic_gemv(
    vector: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=vector.dtype, device=vector.device)
    return F.linear(vector, weight, bias)


def dyadic_gemm(
    matrix: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=matrix.dtype, device=matrix.device)
    return F.linear(matrix, weight, bias)


def dyadic_embedding(
    indices: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bits: int | None = None,
    padding_idx: int | None = None,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=torch.float32, device=indices.device)
    return F.embedding(indices, weight, padding_idx=padding_idx)


def dyadic_output_projection(
    hidden: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bits: int | None = None,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=hidden.dtype, device=hidden.device)
    return F.linear(hidden, weight)


def dyadic_conv2d(
    inputs: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
) -> torch.Tensor:
    weight = _dyop_weight(encoded, bits, dtype=inputs.dtype, device=inputs.device)
    return F.conv2d(inputs, weight, bias, stride=stride, padding=padding, groups=groups)


def dyadic_activation_matmul(
    encoded_left: DyadicTensor,
    encoded_right_t: DyadicTensor,
    *,
    left_bits: int,
    right_bits: int,
) -> torch.Tensor:
    left = _dyop_weight(
        encoded_left,
        left_bits,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    right = _dyop_weight(
        encoded_right_t,
        right_bits,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return left @ right.T


def native_add_relu_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from dyadic_quant.level2.native import native_add_relu_cpu as native_fn

    return native_fn(a, b)


def native_add_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from dyadic_quant.level2.native import native_add_cpu as native_fn

    return native_fn(a, b)


def native_relu_cpu(a: torch.Tensor) -> torch.Tensor:
    from dyadic_quant.level2.native import native_relu_cpu as native_fn

    return native_fn(a)


def native_max_pool2d_cpu(
    inputs: torch.Tensor,
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    if not (
        _same_pair(kernel_size)
        and _same_pair(stride)
        and _same_pair(padding)
        and _same_pair(dilation)
        and _first(dilation) == 1
        and not ceil_mode
    ):
        raise NotImplementedError(
            "native MaxPool2d supports only square kernel/stride/padding, "
            "dilation=1, and ceil_mode=False"
        )
    from dyadic_quant.level2.native import native_max_pool2d_cpu as native_fn

    return native_fn(inputs, _first(kernel_size), _first(stride), _first(padding))


def native_adaptive_avg_pool2d_cpu(
    inputs: torch.Tensor,
    output_size: int | tuple[int, int],
) -> torch.Tensor:
    if not _same_pair(output_size):
        raise NotImplementedError(
            "native AdaptiveAvgPool2d supports only square output_size"
        )
    from dyadic_quant.level2.native import native_adaptive_avg_pool2d_cpu as native_fn

    return native_fn(inputs, _first(output_size))


# Native CPU convenience wrappers — these call the compiled C++ extension
# via pack_native_cpu_weight + packed kernel.  For the "scalar" backend the
# pure Python dyadic_linear / dyadic_embedding / dyadic_conv2d are used.


def _require_native():
    from dyadic_quant.level2.native import build_native_cpu
    build_native_cpu()
    from dyadic_quant.level2.native import (
        pack_native_cpu_weight,
        dyadic_linear_packed_native_cpu,
        dyadic_embedding_packed_native_cpu,
        dyadic_conv2d_packed_native_cpu,
    )
    return (
        pack_native_cpu_weight,
        dyadic_linear_packed_native_cpu,
        dyadic_embedding_packed_native_cpu,
        dyadic_conv2d_packed_native_cpu,
    )


def dyadic_linear_native_cpu(
    inputs: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
    packed_weight=None,
) -> torch.Tensor:
    p, fn, _, _ = _require_native()
    weight = packed_weight if packed_weight is not None else _pack_native_weight(p, encoded, bits)
    if inputs.ndim <= 2:
        return fn(inputs, weight, bias)
    original_shape = inputs.shape[:-1]
    output = fn(inputs.reshape(-1, inputs.shape[-1]), weight, bias)
    return output.reshape(*original_shape, output.shape[-1])


def dyadic_embedding_native_cpu(
    indices: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bits: int | None = None,
    padding_idx: int | None = None,
    packed_weight=None,
) -> torch.Tensor:
    p, _, fn, _ = _require_native()
    weight = packed_weight if packed_weight is not None else _pack_native_weight(p, encoded, bits)
    return fn(indices, weight)


def dyadic_conv2d_native_cpu(
    inputs: torch.Tensor,
    encoded: DyadicTensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
    packed_weight=None,
) -> torch.Tensor:
    weight_shape = tuple(int(dim) for dim in encoded.signs.shape)
    _validate_native_conv2d_shape(weight_shape, stride=stride, padding=padding, groups=groups)
    p, _, _, fn = _require_native()
    return fn(
        inputs,
        packed_weight if packed_weight is not None else _pack_native_weight(p, encoded, bits),
        bias,
        _first(stride),
        _first(padding),
        weight_shape[2],
        weight_shape[3],
    )


def _first(value: int | tuple[int, int]) -> int:
    return value if isinstance(value, int) else value[0]


def _same_pair(value: int | tuple[int, int]) -> bool:
    return isinstance(value, int) or value[0] == value[1]


def _pack_native_weight(pack_fn, encoded: DyadicTensor, bits: int | None):
    return pack_fn(
        encoded.signs,
        encoded.magnitude_code,
        encoded.exponents,
        encoded.max_bits,
        encoded.group_size,
        _resolved_bits(encoded, bits),
    )


def pack_native_weight(encoded: DyadicTensor, bits: int | None):
    pack_fn, _, _, _ = _require_native()
    return _pack_native_weight(pack_fn, encoded, bits)


def _validate_native_conv2d(
    weight: torch.Tensor,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    groups: int,
) -> None:
    _validate_native_conv2d_shape(
        tuple(int(dim) for dim in weight.shape),
        stride=stride,
        padding=padding,
        groups=groups,
    )


def _validate_native_conv2d_shape(
    weight_shape: tuple[int, ...],
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    groups: int,
) -> None:
    if groups != 1:
        raise NotImplementedError("native Conv2d currently supports groups=1 only")
    if not _same_pair(stride) or not _same_pair(padding):
        raise NotImplementedError(
            "native Conv2d currently supports square stride and padding only"
        )
    if len(weight_shape) != 4:
        raise NotImplementedError("native Conv2d requires a 4D weight tensor")
    if weight_shape[2:] not in ((1, 1), (3, 3)):
        raise NotImplementedError(
            "native Conv2d currently supports only 1x1 and 3x3 kernels"
        )


def _dyop_weight(
    encoded: DyadicTensor,
    bits: int | None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_bits = _resolved_bits(encoded, bits)
    prefix_magnitude_bits = resolved_bits - 1
    shift = encoded.magnitude_bits - prefix_magnitude_bits
    prefix_code = torch.bitwise_right_shift(encoded.magnitude_code, shift)
    exponents = _expand_exponents(encoded, dtype=dtype)
    prefix_step = torch.pow(torch.tensor(2.0, dtype=dtype), exponents + shift)
    magnitude = prefix_code.to(dtype) + 0.5
    weight = encoded.signs.to(dtype) * prefix_step * magnitude
    return weight if device is None else weight.to(device)


def _resolved_bits(encoded: DyadicTensor, bits: int | None) -> int:
    resolved = encoded.max_bits if bits is None else bits
    if resolved < 2 or resolved > encoded.max_bits:
        raise ValueError(f"bits must be in [2, {encoded.max_bits}]")
    return resolved


def _expand_exponents(encoded: DyadicTensor, *, dtype: torch.dtype) -> torch.Tensor:
    out_channels = encoded.signs.shape[0]
    elements = 1
    for size in encoded.signs.shape[1:]:
        elements *= size
    per_weight = encoded.exponents.repeat_interleave(encoded.group_size, dim=1)
    per_weight = per_weight[:, :elements]
    return per_weight.reshape(out_channels, *encoded.signs.shape[1:]).to(dtype)
