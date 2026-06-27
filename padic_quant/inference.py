from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from .arithmetic import centered_mod, crt2_centered

Arithmetic = Literal["wide", "modular", "rns"]


@dataclass
class LayerQuantization:
    weight_codes: np.ndarray
    bias_codes: np.ndarray
    input_scale: float
    weight_scale: float
    weight_clip_rate: float


@dataclass
class QuantizedMLP:
    layers: list[LayerQuantization]
    operand_bits: int
    parameter_count: int


@dataclass
class Evaluation:
    logits: np.ndarray
    accuracy: float
    agreement: float
    logit_mae: float
    wrap_rate: float
    wrapped_values: int
    accumulator_values: int
    max_abs_accumulator: int
    runtime_ms: float
    multiply_accumulates: int
    input_clip_rate: float
    weight_clip_rate: float


def _max_symmetric_scale(values: np.ndarray, qmax: int) -> float:
    maximum = float(np.max(np.abs(values)))
    return qmax / maximum if maximum > 0 else 1.0


def _mse_symmetric_scale(values: np.ndarray, qmin: int, qmax: int) -> float:
    """Choose a clipping threshold that minimizes calibration-tensor MSE."""
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return 1.0

    # A geometric grid resolves aggressive clipping while retaining max-range
    # candidates. This is deterministic and cheap for the small calibration sets.
    fractions = np.geomspace(0.05, 1.0, num=80)
    best_scale = qmax / maximum
    best_error = math.inf
    for fraction in fractions:
        scale = qmax / (maximum * fraction)
        codes = np.clip(np.rint(values * scale), qmin, qmax)
        error = float(np.mean(np.square(codes / scale - values)))
        if error < best_error:
            best_error = error
            best_scale = scale
    return best_scale


def _quantize(
    values: np.ndarray, scale: float, qmin: int, qmax: int
) -> tuple[np.ndarray, float]:
    rounded = np.rint(values * scale)
    clipped = np.clip(rounded, qmin, qmax)
    clip_rate = float(np.mean(rounded != clipped))
    return clipped.astype(np.int64), clip_rate


def build_quantized_mlp(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    calibration_inputs: list[np.ndarray],
    operand_bits: int,
) -> QuantizedMLP:
    if operand_bits < 2:
        raise ValueError("operand_bits must be at least 2")
    qmin = -(2 ** (operand_bits - 1))
    qmax = 2 ** (operand_bits - 1) - 1
    layers: list[LayerQuantization] = []
    for weight, bias, layer_inputs in zip(
        weights, biases, calibration_inputs, strict=True
    ):
        input_scale = _mse_symmetric_scale(layer_inputs, qmin, qmax)
        weight_scale = _max_symmetric_scale(weight, qmax)
        weight_codes, weight_clip_rate = _quantize(weight, weight_scale, qmin, qmax)
        bias_codes = np.rint(bias * input_scale * weight_scale).astype(np.int64)
        layers.append(
            LayerQuantization(
                weight_codes=weight_codes,
                bias_codes=bias_codes,
                input_scale=input_scale,
                weight_scale=weight_scale,
                weight_clip_rate=weight_clip_rate,
            )
        )
    parameter_count = sum(
        layer.weight_codes.size + layer.bias_codes.size for layer in layers
    )
    return QuantizedMLP(
        layers=layers,
        operand_bits=operand_bits,
        parameter_count=parameter_count,
    )


def _matmul_rns(
    input_codes: np.ndarray,
    weight_codes: np.ndarray,
    bias_codes: np.ndarray,
    moduli: tuple[int, int],
) -> np.ndarray:
    modulus_a, modulus_b = moduli
    result_a = (
        np.mod(input_codes, modulus_a) @ np.mod(weight_codes.T, modulus_a)
        + np.mod(bias_codes, modulus_a)
    ) % modulus_a
    result_b = (
        np.mod(input_codes, modulus_b) @ np.mod(weight_codes.T, modulus_b)
        + np.mod(bias_codes, modulus_b)
    ) % modulus_b
    return crt2_centered(result_a, result_b, modulus_a, modulus_b)


def _forward_once(
    model: QuantizedMLP,
    inputs: np.ndarray,
    arithmetic: Arithmetic,
    modulus: int | None,
    rns_moduli: tuple[int, int] | None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    qmin = -(2 ** (model.operand_bits - 1))
    qmax = 2 ** (model.operand_bits - 1) - 1
    current = inputs.astype(np.float64)
    wrapped_values = 0
    accumulator_values = 0
    max_abs_accumulator = 0
    multiply_accumulates = 0
    input_clipped = 0
    input_total = 0

    for layer_index, layer in enumerate(model.layers):
        input_codes, input_clip_rate = _quantize(
            current, layer.input_scale, qmin, qmax
        )
        input_clipped += round(input_clip_rate * input_codes.size)
        input_total += input_codes.size
        exact_accumulator = (
            input_codes @ layer.weight_codes.T + layer.bias_codes
        )
        accumulator_values += exact_accumulator.size
        if exact_accumulator.size:
            max_abs_accumulator = max(
                max_abs_accumulator, int(np.max(np.abs(exact_accumulator)))
            )
        multiply_accumulates += (
            input_codes.shape[0]
            * input_codes.shape[1]
            * layer.weight_codes.shape[0]
        )

        if arithmetic == "wide":
            accumulator = exact_accumulator
        elif arithmetic == "modular":
            if modulus is None:
                raise ValueError("modular arithmetic requires a modulus")
            accumulator = centered_mod(exact_accumulator, modulus)
            wrapped_values += int(np.count_nonzero(accumulator != exact_accumulator))
        elif arithmetic == "rns":
            if rns_moduli is None:
                raise ValueError("RNS arithmetic requires two moduli")
            accumulator = _matmul_rns(
                input_codes,
                layer.weight_codes,
                layer.bias_codes,
                rns_moduli,
            )
            wrapped_values += int(np.count_nonzero(accumulator != exact_accumulator))
        else:
            raise ValueError(f"unknown arithmetic: {arithmetic}")

        current = accumulator.astype(np.float64) / (
            layer.input_scale * layer.weight_scale
        )
        if layer_index < len(model.layers) - 1:
            current = np.maximum(current, 0.0)

    return current, {
        "wrapped_values": wrapped_values,
        "accumulator_values": accumulator_values,
        "max_abs_accumulator": max_abs_accumulator,
        "multiply_accumulates": multiply_accumulates,
        "input_clipped": input_clipped,
        "input_total": input_total,
    }


def evaluate_quantized(
    model: QuantizedMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    float_logits: np.ndarray,
    *,
    arithmetic: Arithmetic,
    modulus: int | None = None,
    rns_moduli: tuple[int, int] | None = None,
    timing_repeats: int = 10,
) -> Evaluation:
    logits, counters = _forward_once(
        model, inputs, arithmetic, modulus, rns_moduli
    )
    start = perf_counter()
    for _ in range(timing_repeats):
        _forward_once(model, inputs, arithmetic, modulus, rns_moduli)
    runtime_ms = (perf_counter() - start) * 1000 / timing_repeats

    predictions = np.argmax(logits, axis=1)
    float_predictions = np.argmax(float_logits, axis=1)
    accumulator_values = int(counters["accumulator_values"])
    wrapped_values = int(counters["wrapped_values"])
    return Evaluation(
        logits=logits,
        accuracy=float(np.mean(predictions == targets)),
        agreement=float(np.mean(predictions == float_predictions)),
        logit_mae=float(np.mean(np.abs(logits - float_logits))),
        wrap_rate=wrapped_values / accumulator_values if accumulator_values else 0.0,
        wrapped_values=wrapped_values,
        accumulator_values=accumulator_values,
        max_abs_accumulator=int(counters["max_abs_accumulator"]),
        runtime_ms=runtime_ms,
        multiply_accumulates=int(counters["multiply_accumulates"]),
        input_clip_rate=(
            int(counters["input_clipped"]) / int(counters["input_total"])
            if int(counters["input_total"])
            else 0.0
        ),
        weight_clip_rate=float(
            np.mean([layer.weight_clip_rate for layer in model.layers])
        ),
    )
