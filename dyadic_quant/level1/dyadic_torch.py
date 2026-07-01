from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn


@dataclass
class DyadicTensor:
    """Nested sign-magnitude dyadic encoding with one power-of-two scale per group.

    The magnitude planes are stored MSB-first per weight, so any prefix width is
    obtained by dropping low planes. The exponent that sets the dyadic step is
    shared by a contiguous group of ``group_size`` weights within each output
    row; ``group_size == K`` (a full row) recovers per-channel scaling.
    """

    signs: torch.Tensor
    magnitude_code: torch.Tensor
    exponents: torch.Tensor  # shape [out_channels, num_blocks]
    max_bits: int
    group_size: int

    @property
    def magnitude_bits(self) -> int:
        return self.max_bits - 1

    def _expand_exponents(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Broadcast per-block exponents to one exponent per weight."""
        out_channels = self.signs.shape[0]
        rest = self.signs.shape[1:]
        elements = 1
        for size in rest:
            elements *= size
        per_weight = self.exponents.repeat_interleave(self.group_size, dim=1)
        per_weight = per_weight[:, :elements]
        return per_weight.reshape(out_channels, *rest).to(dtype)

    def decode(self, bits: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if bits < 2 or bits > self.max_bits:
            raise ValueError(f"bits must be in [2, {self.max_bits}]")
        prefix_magnitude_bits = bits - 1
        shift = self.magnitude_bits - prefix_magnitude_bits
        prefix_code = torch.bitwise_right_shift(self.magnitude_code, shift)
        exponents = self._expand_exponents(dtype=dtype)
        # A prefix identifies a dyadic interval. Decode at its midpoint rather
        # than its lower edge to avoid a systematic shrinkage bias across deep
        # networks. Refinement replaces that interval by one of its two halves.
        prefix_step = torch.pow(torch.tensor(2.0, dtype=dtype), exponents + shift)
        magnitude = prefix_code.to(dtype) + 0.5
        return self.signs.to(dtype) * prefix_step * magnitude


@dataclass
class EncodedModule:
    name: str
    tensor: DyadicTensor
    weight_count: int
    output_channels: int


@dataclass
class EncodedModel:
    modules: list[EncodedModule]
    max_bits: int
    conversion_ms: float
    quantized_weight_count: int
    exponent_count: int
    group_size: int


def save_encoded_model(encoded: EncodedModel, path: Path) -> None:
    """Serialize one packed maximum-depth code that supports every prefix."""
    if encoded.max_bits > 8:
        raise ValueError("the packed dyadic artifact currently supports at most 8 bits")
    path.parent.mkdir(parents=True, exist_ok=True)
    modules: list[dict[str, object]] = []
    sign_shift = encoded.max_bits - 1
    for item in encoded.modules:
        sign_bit = (item.tensor.signs < 0).to(torch.uint8)
        packed = torch.bitwise_or(
            item.tensor.magnitude_code.to(torch.uint8),
            torch.bitwise_left_shift(sign_bit, sign_shift),
        )
        modules.append(
            {
                "name": item.name,
                "shape": tuple(item.tensor.signs.shape),
                "packed_sign_magnitude": packed,
                "exponents": item.tensor.exponents,
                "group_size": item.tensor.group_size,
                "weight_count": item.weight_count,
                "output_channels": item.output_channels,
            }
        )
    payload = {
        "format": "progressive_dyadic_sign_magnitude_v1",
        "max_bits": encoded.max_bits,
        "group_size": encoded.group_size,
        "quantized_weight_count": encoded.quantized_weight_count,
        "exponent_count": encoded.exponent_count,
        "modules": modules,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_encoded_model(path: Path) -> EncodedModel:
    """Load a packed maximum-depth dyadic code saved by ``save_encoded_model``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "progressive_dyadic_sign_magnitude_v1":
        raise ValueError(f"unsupported dyadic format: {payload.get('format')!r}")
    max_bits = int(payload["max_bits"])
    if max_bits < 2 or max_bits > 8:
        raise ValueError("the packed dyadic artifact currently supports 2-8 bits")
    sign_shift = max_bits - 1
    magnitude_mask = (1 << sign_shift) - 1
    modules: list[EncodedModule] = []
    for raw in payload["modules"]:
        packed = raw["packed_sign_magnitude"].to(torch.uint8)
        shape = tuple(int(size) for size in raw["shape"])
        if tuple(packed.shape) != shape:
            raise ValueError(f"packed tensor shape mismatch for {raw['name']!r}")
        sign_bit = torch.bitwise_right_shift(packed, sign_shift)
        signs = torch.where(sign_bit > 0, -1, 1).to(torch.int8)
        magnitude_code = torch.bitwise_and(packed, magnitude_mask).to(torch.int32)
        if "group_size" in raw:
            group_size = int(raw["group_size"])
        else:
            group_size = math.prod(shape[1:])
        exponents = raw["exponents"].to(torch.int16)
        if exponents.ndim == 1:
            exponents = exponents.unsqueeze(1)
        tensor = DyadicTensor(
            signs=signs.reshape(shape),
            magnitude_code=magnitude_code.reshape(shape),
            exponents=exponents,
            max_bits=max_bits,
            group_size=group_size,
        )
        modules.append(
            EncodedModule(
                name=str(raw["name"]),
                tensor=tensor,
                weight_count=int(raw["weight_count"]),
                output_channels=int(raw["output_channels"]),
            )
        )
    return EncodedModel(
        modules=modules,
        max_bits=max_bits,
        conversion_ms=0.0,
        quantized_weight_count=int(payload["quantized_weight_count"]),
        exponent_count=int(payload["exponent_count"]),
        group_size=int(payload.get("group_size", 0)),
    )


def _blocks_for(elements: int, group_size: int | None) -> tuple[int, int]:
    """Return the effective group size and block count for a row of weights."""
    if group_size is None or group_size >= elements or group_size <= 0:
        return elements, 1
    num_blocks = math.ceil(elements / group_size)
    return group_size, num_blocks


def encode_tensor_per_output_channel(
    weight: torch.Tensor,
    *,
    max_bits: int,
    optimize_prefix_bits: tuple[int, ...] | None = None,
    group_size: int | None = None,
) -> DyadicTensor:
    if max_bits < 2 or max_bits > 16:
        raise ValueError("max_bits must be between 2 and 16")
    if weight.ndim < 2:
        raise ValueError("weight must have an output-channel dimension")

    source = weight.detach().to(device="cpu", dtype=torch.float32)
    out_channels = source.shape[0]
    flat = source.reshape(out_channels, -1)
    elements = flat.shape[1]
    effective_group, num_blocks = _blocks_for(elements, group_size)
    padded_elements = effective_group * num_blocks
    if padded_elements != elements:
        padding = torch.zeros(
            out_channels, padded_elements - elements, dtype=flat.dtype
        )
        padded = torch.cat([flat, padding], dim=1)
    else:
        padded = flat
    # [out, num_blocks, group]: the group axis is reduced for scale selection.
    grouped = padded.reshape(out_channels, num_blocks, effective_group)

    if optimize_prefix_bits is None:
        optimize_prefix_bits = tuple(range(max(2, min(4, max_bits)), max_bits + 1))
    if not optimize_prefix_bits:
        raise ValueError("at least one prefix width is required")
    if any(bits < 2 or bits > max_bits for bits in optimize_prefix_bits):
        raise ValueError("optimized prefix widths must be between 2 and max_bits")

    magnitude_bits = max_bits - 1
    levels = 2**magnitude_bits
    maximum_code = levels - 1
    maxima = grouped.abs().amax(dim=-1)  # [out, num_blocks]
    safe_maxima = torch.where(maxima > 0, maxima, torch.ones_like(maxima))
    coverage_exponents = torch.ceil(torch.log2(safe_maxima / levels))

    signed_grouped = torch.where(grouped < 0, -1.0, 1.0)

    # Select a power-of-two base step by minimizing the average reconstruction
    # error of all requested prefixes within each group. Lower candidates
    # intentionally permit clipping a few outliers; this is essential at 4-6
    # bits and is now decided independently per group rather than per channel.
    candidates: list[torch.Tensor] = []
    errors_by_candidate: list[list[torch.Tensor]] = []
    for delta in range(-6, 4):
        candidate_exponents = coverage_exponents + delta  # [out, num_blocks]
        steps = torch.pow(2.0, candidate_exponents).unsqueeze(-1)
        codes = torch.clamp(
            torch.floor(grouped.abs() / steps), min=0, max=maximum_code
        ).to(torch.int32)
        prefix_errors: list[torch.Tensor] = []
        for bits in optimize_prefix_bits:
            shift = max_bits - bits
            prefix_code = torch.bitwise_right_shift(codes, shift)
            prefix_step = steps * float(2**shift)
            reconstruction = (
                signed_grouped * prefix_step * (prefix_code.to(torch.float32) + 0.5)
            )
            prefix_errors.append(
                torch.mean(torch.square(reconstruction - grouped), dim=-1)
            )
        candidates.append(candidate_exponents)
        errors_by_candidate.append(prefix_errors)

    # Raw low-bit MSE dominates a direct sum, so normalize each prefix by its
    # independently best candidate and minimize total relative regret.
    best_per_prefix = [
        torch.stack(
            [candidate[prefix_index] for candidate in errors_by_candidate]
        ).amin(dim=0)
        for prefix_index in range(len(optimize_prefix_bits))
    ]
    best_error = torch.full_like(maxima, torch.inf)
    best_exponents = coverage_exponents.clone()
    for candidate_exponents, prefix_errors in zip(
        candidates, errors_by_candidate, strict=True
    ):
        score = torch.zeros_like(maxima)
        for error, best in zip(prefix_errors, best_per_prefix, strict=True):
            score += error / torch.clamp(best, min=1e-20)
        better = score < best_error
        best_error = torch.where(better, score, best_error)
        best_exponents = torch.where(better, candidate_exponents, best_exponents)

    exponents = best_exponents.to(torch.int16)  # [out, num_blocks]
    steps = torch.pow(2.0, best_exponents).unsqueeze(-1)

    # Floor is intentional: dropping low planes must equal taking an earlier
    # prefix of the one stored maximum-depth code.
    grouped_codes = torch.clamp(
        torch.floor(grouped.abs() / steps), min=0, max=maximum_code
    ).to(torch.int32)
    grouped_signs = torch.where(grouped < 0, -1, 1).to(torch.int8)

    codes = grouped_codes.reshape(out_channels, padded_elements)[:, :elements]
    signs = grouped_signs.reshape(out_channels, padded_elements)[:, :elements]
    return DyadicTensor(
        signs=signs.reshape(source.shape),
        magnitude_code=codes.reshape(source.shape),
        exponents=exponents,
        max_bits=max_bits,
        group_size=effective_group,
    )


def encode_model(
    model: nn.Module,
    *,
    max_bits: int = 8,
    optimize_prefix_bits: tuple[int, ...] | None = None,
    exclude_names: set[str] | None = None,
    group_size: int | None = None,
) -> EncodedModel:
    start = perf_counter()
    exclude_names = exclude_names or set()
    encoded: list[EncodedModule] = []
    encoded_parameter_ids: set[int] = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding))
            and name not in exclude_names
            and id(module.weight) not in encoded_parameter_ids
        ):
            tensor = encode_tensor_per_output_channel(
                module.weight,
                max_bits=max_bits,
                optimize_prefix_bits=optimize_prefix_bits,
                group_size=group_size,
            )
            encoded.append(
                EncodedModule(
                    name=name,
                    tensor=tensor,
                    weight_count=module.weight.numel(),
                    output_channels=module.weight.shape[0],
                )
            )
            encoded_parameter_ids.add(id(module.weight))
    return EncodedModel(
        modules=encoded,
        max_bits=max_bits,
        conversion_ms=(perf_counter() - start) * 1000,
        quantized_weight_count=sum(item.weight_count for item in encoded),
        exponent_count=sum(item.tensor.exponents.numel() for item in encoded),
        group_size=group_size if group_size is not None else 0,
    )


def materialize_prefix(
    model: nn.Module,
    encoded: EncodedModel,
    *,
    bits: int,
    dtype: torch.dtype = torch.float32,
    overrides: dict[str, int] | None = None,
) -> float:
    """Write the requested prefix into the model weights.

    ``overrides`` maps a module name to a deeper (or shallower) prefix width,
    allowing mixed-precision materialization (e.g. keeping the embedding at a
    higher width) from the single stored maximum-depth code.
    """
    overrides = overrides or {}
    modules = dict(model.named_modules())
    start = perf_counter()
    with torch.no_grad():
        for item in encoded.modules:
            module = modules[item.name]
            module_bits = overrides.get(item.name, bits)
            module.weight.copy_(item.tensor.decode(module_bits, dtype=dtype))
    return (perf_counter() - start) * 1000


def storage_bytes(
    model: nn.Module,
    encoded: EncodedModel,
    *,
    bits: int,
    nonquantized_dtype_bytes: int = 2,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    overrides = overrides or {}
    modules = dict(model.named_modules())
    quantized_parameter_ids = {
        id(modules[item.name].weight) for item in encoded.modules
    }
    nonquantized_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in quantized_parameter_ids
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    payload_bits = sum(
        item.weight_count * overrides.get(item.name, bits)
        for item in encoded.modules
    )
    payload_bytes = math.ceil(payload_bits / 8)
    # One signed byte is enough for the measured per-group exponents.
    exponent_bytes = encoded.exponent_count
    other_bytes = (nonquantized_count + buffer_count) * nonquantized_dtype_bytes
    return {
        "weight_payload_bytes": payload_bytes,
        "exponent_bytes": exponent_bytes,
        "other_model_bytes": other_bytes,
        "total_model_bytes": payload_bytes + exponent_bytes + other_bytes,
        "incremental_plane_bytes": math.ceil(
            encoded.quantized_weight_count / 8
        ),
    }
