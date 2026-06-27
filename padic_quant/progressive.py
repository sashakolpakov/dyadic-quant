from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

Method = Literal["independent", "monna", "residual_binary"]


@dataclass
class ProgressiveLayer:
    original: np.ndarray
    bias: np.ndarray
    monna_scale: float
    monna_sign: np.ndarray
    monna_planes: list[np.ndarray]
    residual_planes: list[np.ndarray]
    residual_scales: list[float]


@dataclass
class ProgressiveModel:
    layers: list[ProgressiveLayer]
    max_bits: int
    conversion_ms: float
    monna_conversion_ms: float
    residual_conversion_ms: float
    weight_count: int
    bias_count: int


def _mse_uniform_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be positive")
    if bits == 1:
        scale = float(np.mean(np.abs(values)))
        return np.where(values >= 0, scale, -scale)

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return np.zeros_like(values)

    best = np.zeros_like(values)
    best_error = math.inf
    for fraction in np.geomspace(0.05, 1.0, num=100):
        scale = qmax / (maximum * fraction)
        codes = np.clip(np.rint(values * scale), qmin, qmax)
        reconstructed = codes / scale
        error = float(np.mean(np.square(reconstructed - values)))
        if error < best_error:
            best_error = error
            best = reconstructed
    return best


def _build_monna_planes(
    weights: np.ndarray, magnitude_bits: int
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """Encode magnitudes by binary fractions, most significant digit first."""
    scale = float(np.max(np.abs(weights)))
    signs = np.where(weights < 0, -1.0, 1.0)
    if scale == 0:
        return 1.0, signs, [np.zeros_like(weights, dtype=np.uint8)] * magnitude_bits

    normalized = np.clip(np.abs(weights) / scale, 0.0, 1.0)
    levels = 2**magnitude_bits
    codes = np.minimum(np.floor(normalized * levels), levels - 1).astype(np.uint64)
    planes = [
        ((codes >> shift) & 1).astype(np.uint8)
        for shift in range(magnitude_bits - 1, -1, -1)
    ]
    return scale, signs, planes


def _build_residual_binary_planes(
    weights: np.ndarray, bits: int
) -> tuple[list[np.ndarray], list[float]]:
    """Greedy nested binary expansion with one bipolar plane per refinement."""
    residual = weights.astype(np.float64).copy()
    planes: list[np.ndarray] = []
    scales: list[float] = []
    for _ in range(bits):
        scale = float(np.mean(np.abs(residual)))
        if scale == 0:
            plane = np.ones_like(residual, dtype=np.int8)
        else:
            plane = np.where(residual >= 0, 1, -1).astype(np.int8)
        planes.append(plane)
        scales.append(scale)
        residual = residual - scale * plane
    return planes, scales


def build_progressive_model(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    *,
    max_bits: int,
) -> ProgressiveModel:
    if max_bits < 2:
        raise ValueError("max_bits must be at least 2")
    total_start = perf_counter()
    monna_conversion_ms = 0.0
    residual_conversion_ms = 0.0
    layers: list[ProgressiveLayer] = []
    for weight, bias in zip(weights, biases, strict=True):
        start = perf_counter()
        monna_scale, monna_sign, monna_planes = _build_monna_planes(
            weight, max_bits - 1
        )
        monna_conversion_ms += (perf_counter() - start) * 1000
        start = perf_counter()
        residual_planes, residual_scales = _build_residual_binary_planes(
            weight, max_bits
        )
        residual_conversion_ms += (perf_counter() - start) * 1000
        layers.append(
            ProgressiveLayer(
                original=weight,
                bias=bias,
                monna_scale=monna_scale,
                monna_sign=monna_sign,
                monna_planes=monna_planes,
                residual_planes=residual_planes,
                residual_scales=residual_scales,
            )
        )
    conversion_ms = (perf_counter() - total_start) * 1000
    return ProgressiveModel(
        layers=layers,
        max_bits=max_bits,
        conversion_ms=conversion_ms,
        monna_conversion_ms=monna_conversion_ms,
        residual_conversion_ms=residual_conversion_ms,
        weight_count=sum(weight.size for weight in weights),
        bias_count=sum(bias.size for bias in biases),
    )


def reconstruct_layer(
    layer: ProgressiveLayer,
    *,
    method: Method,
    bits: int,
) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be positive")
    if method == "independent":
        return _mse_uniform_quantize(layer.original, bits)
    if method == "monna":
        if bits < 2:
            return np.zeros_like(layer.original)
        magnitude = np.zeros_like(layer.original)
        for index, plane in enumerate(layer.monna_planes[: bits - 1], start=1):
            magnitude += plane * (2.0**-index)
        return layer.monna_scale * layer.monna_sign * magnitude
    if method == "residual_binary":
        reconstructed = np.zeros_like(layer.original)
        for plane, scale in zip(
            layer.residual_planes[:bits],
            layer.residual_scales[:bits],
            strict=True,
        ):
            reconstructed += scale * plane
        return reconstructed
    raise ValueError(f"unknown method: {method}")


def reconstruct_model(
    model: ProgressiveModel, *, method: Method, bits: int
) -> list[np.ndarray]:
    return [
        reconstruct_layer(layer, method=method, bits=bits) for layer in model.layers
    ]


def forward_weights(
    weights: list[np.ndarray], biases: list[np.ndarray], inputs: np.ndarray
) -> np.ndarray:
    current = inputs.astype(np.float64)
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        current = current @ weight.T + bias
        if index < len(weights) - 1:
            current = np.maximum(current, 0.0)
    return current


def evaluate_representation(
    model: ProgressiveModel,
    inputs: np.ndarray,
    targets: np.ndarray,
    float_logits: np.ndarray,
    *,
    method: Method,
    bits: int,
    timing_repeats: int,
) -> dict[str, float | int]:
    start = perf_counter()
    reconstructed = reconstruct_model(model, method=method, bits=bits)
    reconstruction_ms = (perf_counter() - start) * 1000
    biases = [layer.bias for layer in model.layers]
    logits = forward_weights(reconstructed, biases, inputs)

    start = perf_counter()
    for _ in range(timing_repeats):
        forward_weights(reconstructed, biases, inputs)
    runtime_ms = (perf_counter() - start) * 1000 / timing_repeats

    errors = np.concatenate(
        [
            (reconstructed_weight - layer.original).ravel()
            for reconstructed_weight, layer in zip(
                reconstructed, model.layers, strict=True
            )
        ]
    )
    predictions = np.argmax(logits, axis=1)
    float_predictions = np.argmax(float_logits, axis=1)

    if method == "monna":
        payload_bits = model.weight_count * bits
        metadata_floats = len(model.layers)
        active_fraction = float(
            np.mean(
                np.concatenate(
                    [
                        np.stack(layer.monna_planes[: max(bits - 1, 0)]).ravel()
                        for layer in model.layers
                        if bits >= 2
                    ]
                )
            )
        ) if bits >= 2 else 0.0
        primitive_weight_ops = round(
            model.weight_count * max(bits - 1, 0) * active_fraction
        )
        incremental_plane_ops = (
            sum(
                int(np.count_nonzero(layer.monna_planes[bits - 2]))
                for layer in model.layers
            )
            if bits >= 2
            else 0
        )
    elif method == "residual_binary":
        payload_bits = model.weight_count * bits
        metadata_floats = len(model.layers) * bits
        active_fraction = 1.0
        primitive_weight_ops = model.weight_count * bits
        incremental_plane_ops = model.weight_count
    else:
        payload_bits = model.weight_count * bits
        metadata_floats = len(model.layers)
        active_fraction = 1.0
        primitive_weight_ops = model.weight_count
        incremental_plane_ops = model.weight_count

    metadata_bits = metadata_floats * 32 + model.bias_count * 32
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "float_agreement": float(np.mean(predictions == float_predictions)),
        "logit_mae": float(np.mean(np.abs(logits - float_logits))),
        "weight_mae": float(np.mean(np.abs(errors))),
        "weight_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "weight_max_error": float(np.max(np.abs(errors))),
        "runtime_ms": runtime_ms,
        "reconstruction_ms": reconstruction_ms,
        "payload_bits": payload_bits,
        "metadata_bits": metadata_bits,
        "total_bytes": math.ceil((payload_bits + metadata_bits) / 8),
        "active_plane_fraction": active_fraction,
        "primitive_weight_ops_per_sample": primitive_weight_ops,
        "incremental_plane_ops_per_sample": incremental_plane_ops,
    }
