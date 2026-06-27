import numpy as np

from dyadic_quant.inference import build_quantized_mlp, evaluate_quantized


def test_large_modulus_matches_wide_accumulator():
    weights = [np.array([[0.5, -0.25], [-0.2, 0.4]])]
    biases = [np.array([0.1, -0.1])]
    inputs = np.array([[1.0, 0.5], [-0.5, 0.2]])
    labels = np.array([0, 1])
    float_logits = inputs @ weights[0].T + biases[0]
    model = build_quantized_mlp(weights, biases, [inputs], operand_bits=4)

    wide = evaluate_quantized(
        model, inputs, labels, float_logits, arithmetic="wide", timing_repeats=1
    )
    modular = evaluate_quantized(
        model,
        inputs,
        labels,
        float_logits,
        arithmetic="modular",
        modulus=2**20,
        timing_repeats=1,
    )
    np.testing.assert_allclose(modular.logits, wide.logits)
    assert modular.wrap_rate == 0


def test_rns_matches_single_product_modulus():
    weights = [np.array([[0.8, -0.3], [0.2, 0.7]])]
    biases = [np.array([0.05, -0.02])]
    inputs = np.array([[1.2, -0.4], [0.3, 0.9]])
    labels = np.array([0, 1])
    float_logits = inputs @ weights[0].T + biases[0]
    model = build_quantized_mlp(weights, biases, [inputs], operand_bits=4)

    modular = evaluate_quantized(
        model,
        inputs,
        labels,
        float_logits,
        arithmetic="modular",
        modulus=253,
        timing_repeats=1,
    )
    rns = evaluate_quantized(
        model,
        inputs,
        labels,
        float_logits,
        arithmetic="rns",
        rns_moduli=(11, 23),
        timing_repeats=1,
    )
    np.testing.assert_allclose(rns.logits, modular.logits)
    assert rns.wrap_rate == modular.wrap_rate

