# Experimental results

All values below are means over seeds 7, 19, and 31. The full per-seed data is
in `results/small_network_results.csv`; grouped results are in
`results/small_network_summary.csv`.

## Accuracy controls

| Dataset | Float | 4-bit, wide accumulator | 8-bit, wide accumulator |
|---|---:|---:|---:|
| Two moons | 96.73% | 94.87% | 96.60% |
| Digits | 97.48% | 96.81% | 97.48% |

The 4-bit and 8-bit controls use MSE-calibrated symmetric activations and
non-wrapping integer accumulators.

## Power-of-two finite rings

| Dataset | Operand bits | `2^8` | `2^12` | `2^16` | `2^18` |
|---|---:|---:|---:|---:|---:|
| Two moons | 4 | 94.87% | 94.87% | 94.87% | 94.87% |
| Digits | 4 | 86.30% | 96.81% | 96.81% | 96.81% |
| Two moons | 8 | 48.13% | 50.60% | 96.60% | 96.60% |
| Digits | 8 | 8.74% | 9.33% | 96.44% | 97.48% |

Corresponding mean wrap rates:

| Dataset | Operand bits | `2^8` | `2^12` | `2^16` | `2^18` |
|---|---:|---:|---:|---:|---:|
| Two moons | 4 | 0 | 0 | 0 | 0 |
| Digits | 4 | 0.566% | 0 | 0 | 0 |
| Two moons | 8 | 90.00% | 39.96% | 0.002% | 0 |
| Digits | 8 | 90.72% | 42.71% | 0.030% | 0 |

Even rare wraps can matter: on 8-bit digits, a 0.030% wrap rate at `2^16`
cost about 1.04 percentage points. `2^18` eliminated wraps and exactly matched
the wide-accumulator control. A 17-bit power-of-two modulus should be sufficient
for these observed accumulator ranges, but the sweep conservatively tested 16
and 18 bits.

Odd prime powers behaved according to modulus size, not prime identity. The
smallest tested zero-wrap modulus for both 8-bit tasks was `5^7 = 78,125`
(16.25 entropy bits, 17 physical bits). For 4-bit digits, `3^7 = 2,187`
(11.09 entropy bits, 12 physical bits) was sufficient.

## Residue number system

The 18-bit RNS used lanes modulo 503 and 509, whose product is 256,027.
It exactly matched wide-accumulator accuracy for all four task/precision pairs.
It performs two residue MAC streams per logical MAC.

Measured NumPy RNS inference was roughly 2.2–3.5 times slower than the wide
integer control. This is not a hardware prediction: NumPy pays conversion and
CRT reconstruction overhead and does not execute the two lanes in dedicated
parallel hardware.

## Memory

Packed model storage includes low-bit weights and accumulator-width biases:

| Dataset | Configuration | Bytes |
|---|---|---:|
| Two moons | 8-bit weights + 32-bit biases | 456 |
| Two moons | 8-bit weights + 17-bit ring biases | 393 |
| Digits | 8-bit weights + 32-bit biases | 2,536 |
| Digits | 8-bit weights + 17-bit ring biases | 2,458 |
| Digits | 4-bit weights + 32-bit biases | 1,352 |
| Digits | 4-bit weights + 12-bit ring biases | 1,247 |

The ring does not reduce weight storage beyond ordinary low-bit quantization.
The measured model-memory saving comes from narrower bias/accumulator state and
is modest when weights dominate.

## Practical verdict

Fast post-training conversion is practical:

1. collect activation ranges or calibration tensors;
2. choose per-layer weight/activation precision;
3. bound each layer's integer accumulator;
4. choose `p^N` or an RNS product large enough to avoid harmful wraps;
5. stream-convert one layer at a time.

This supports fitting a model to an available memory budget, but it is standard
mixed-precision quantization plus modular arithmetic. The experiment found no
accuracy benefit from the p-adic topology itself.

Nested p-adic places are also not automatically a progressive real-precision
code. At a fixed scale, adding places expands representable dynamic range; it
does not refine the quantization step. To use more memory for better accuracy,
the converter should recalibrate scales and bit widths for the new budget, or
train a successive-refinement/bit-plane code explicitly.

The most plausible distinct use cases are:

- bounded-width modular accumulators with formally controlled overflow;
- RNS hardware with genuinely parallel narrow lanes;
- streaming deployment where per-layer precision is selected from a memory
  budget and calibration statistics;
- models trained directly for ring arithmetic, rather than converted from a
  real-valued model.

