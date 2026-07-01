from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from dyadic_quant.level1 import DyadicTensor, EncodedModel
from dyadic_quant.level2.dyops import (
    dyadic_conv2d,
    dyadic_conv2d_native_cpu,
    dyadic_embedding,
    dyadic_embedding_native_cpu,
    dyadic_linear,
    dyadic_linear_native_cpu,
    native_adaptive_avg_pool2d_cpu,
    native_max_pool2d_cpu,
    native_relu_cpu,
)


@dataclass
class NativeCPUReplacement:
    replaced_modules: tuple[str, ...] = ()
    shared_weight_modules: tuple[str, ...] = ()


class DyadicLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        encoded: DyadicTensor,
        *,
        bits: int,
        dtype: torch.dtype = torch.float32,
        linear_backend: str = "scalar",
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits
        self.encoded = encoded
        self.linear_backend = _validate_backend(linear_backend, "linear")
        bias = source.bias
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fn = (
            dyadic_linear_native_cpu
            if self.linear_backend == "native-cpu"
            else dyadic_linear
        )
        return fn(
            inputs,
            self.encoded,
            bias=getattr(self, "bias", None),
            bits=self.bits,
        )


class DyadicEmbedding(nn.Module):
    def __init__(
        self,
        source: nn.Embedding,
        encoded: DyadicTensor,
        *,
        bits: int,
        dtype: torch.dtype = torch.float32,
        embedding_backend: str = "scalar",
    ) -> None:
        super().__init__()
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.bits = bits
        self.encoded = encoded
        self.embedding_backend = _validate_backend(embedding_backend, "embedding")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        fn = (
            dyadic_embedding_native_cpu
            if self.embedding_backend == "native-cpu"
            else dyadic_embedding
        )
        return fn(
            indices,
            self.encoded,
            bits=self.bits,
            padding_idx=self.padding_idx,
        )


class DyadicConv2d(nn.Module):
    def __init__(
        self,
        source: nn.Conv2d,
        encoded: DyadicTensor,
        *,
        bits: int,
        dtype: torch.dtype = torch.float32,
        conv_backend: str = "scalar",
    ) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.groups = source.groups
        self.bits = bits
        self.encoded = encoded
        self.conv_backend = _validate_backend(conv_backend, "conv")
        bias = source.bias
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fn = (
            dyadic_conv2d_native_cpu
            if self.conv_backend == "native-cpu"
            else dyadic_conv2d
        )
        return fn(
            inputs,
            self.encoded,
            bias=getattr(self, "bias", None),
            bits=self.bits,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


class NativeReLU(nn.Module):
    def __init__(self, source: nn.ReLU) -> None:
        super().__init__()
        self.inplace = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return native_relu_cpu(inputs)


class NativeMaxPool2d(nn.Module):
    def __init__(self, source: nn.MaxPool2d) -> None:
        super().__init__()
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.ceil_mode = source.ceil_mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return native_max_pool2d_cpu(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )


class NativeAdaptiveAvgPool2d(nn.Module):
    def __init__(self, source: nn.AdaptiveAvgPool2d) -> None:
        super().__init__()
        self.output_size = source.output_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return native_adaptive_avg_pool2d_cpu(inputs, self.output_size)


_MODULE_TYPE_MAP: dict[type, type] = {
    nn.Linear: DyadicLinear,
    nn.Embedding: DyadicEmbedding,
    nn.Conv2d: DyadicConv2d,
}

_SPATIAL_TYPE_MAP: dict[type, type[nn.Module]] = {
    nn.ReLU: NativeReLU,
    nn.MaxPool2d: NativeMaxPool2d,
    nn.AdaptiveAvgPool2d: NativeAdaptiveAvgPool2d,
}


def build_level2_model(
    base_model: nn.Module,
    encoded: EncodedModel,
    *,
    bits: int,
    dtype: torch.dtype = torch.float32,
    linear_backend: str = "scalar",
    conv_backend: str = "scalar",
    embedding_backend: str = "scalar",
    spatial_backend: str = "torch",
    overrides: dict[str, int] | None = None,
) -> tuple[nn.Module, NativeCPUReplacement]:
    overrides = overrides or {}
    _validate_backend(linear_backend, "linear")
    _validate_backend(conv_backend, "conv")
    _validate_backend(embedding_backend, "embedding")
    if spatial_backend not in ("torch", "native-cpu"):
        raise ValueError("spatial_backend must be 'torch' or 'native-cpu'")
    encoded_map: dict[str, DyadicTensor] = {
        item.name: item.tensor for item in encoded.modules
    }
    replaced_modules: list[str] = []
    shared_weight_modules: list[str] = []

    candidate = copy.deepcopy(base_model)
    all_modules = dict(candidate.named_modules())

    # Map weight parameter ids to encoded tensors to handle tied/shared weights.
    weight_id_to_tensor: dict[int, tuple[DyadicTensor, int]] = {}
    for name in list(encoded_map):
        module = all_modules.get(name)
        if module is not None and hasattr(module, "weight"):
            tid = id(module.weight)
            if tid not in weight_id_to_tensor:
                weight_id_to_tensor[tid] = (encoded_map[name], overrides.get(name, bits))

    for name, module in all_modules.items():
        if (
            spatial_backend == "native-cpu"
            and type(module) in _SPATIAL_TYPE_MAP
        ):
            spatial_module = _SPATIAL_TYPE_MAP[type(module)](module)
            _replace_module(candidate, name, spatial_module)
            replaced_modules.append(name)
            continue
        if not isinstance(module, tuple(_MODULE_TYPE_MAP)):
            continue
        module_bits = overrides.get(name, bits)
        encoded_tensor: DyadicTensor | None = encoded_map.get(name)
        # Check for tied weights: if the module's weight tensor was already
        # encoded under a different name, reuse that encoded representation.
        if encoded_tensor is None and hasattr(module, "weight"):
            entry = weight_id_to_tensor.get(id(module.weight))
            if entry is not None:
                encoded_tensor = entry[0]
                module_bits = overrides.get(name, entry[1])
                shared_weight_modules.append(name)
        if encoded_tensor is None:
            continue
        dyadic_cls = _MODULE_TYPE_MAP.get(type(module))
        if dyadic_cls is None:
            continue
        kwargs: dict[str, str] = {}
        if issubclass(dyadic_cls, DyadicLinear):
            kwargs["linear_backend"] = linear_backend
        elif issubclass(dyadic_cls, DyadicEmbedding):
            kwargs["embedding_backend"] = embedding_backend
        elif issubclass(dyadic_cls, DyadicConv2d):
            kwargs["conv_backend"] = conv_backend
        dyadic_module = dyadic_cls(
            module, encoded_tensor, bits=module_bits, dtype=dtype, **kwargs
        )
        _replace_module(candidate, name, dyadic_module)
        replaced_modules.append(name)

    replacement = NativeCPUReplacement(
        replaced_modules=tuple(replaced_modules),
        shared_weight_modules=tuple(shared_weight_modules),
    )
    return candidate, replacement


def _replace_module(
    model: nn.Module,
    target_name: str,
    new_module: nn.Module,
) -> None:
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _validate_backend(backend: str, label: str) -> str:
    if backend not in ("scalar", "native-cpu"):
        raise ValueError(f"{label}_backend must be 'scalar' or 'native-cpu'")
    return backend
