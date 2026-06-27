# Progressive dyadic quantization experiments

This repository contains experiments on low-bit neural-network representations
with two related threads:

1. finite-ring / residue arithmetic for small quantized MLPs;
2. progressive dyadic quantization, where one signed MSB-first bit-plane code
   can be executed at multiple prefix widths and refined by adding exactly one
   bit per encoded weight.

The small-network benchmark trains two float MLPs:

- noisy two-moons: `2 -> 16 -> 16 -> 2`
- handwritten digits: `64 -> 32 -> 10`

It then performs post-training symmetric quantization with 4-bit and 8-bit
weights/activations and compares:

1. ordinary integer inference with a wide, non-wrapping accumulator;
2. linear layers over `Z / p^N Z` for `p = 2, 3, 5`;
3. two-lane residue-number arithmetic, reconstructed by CRT.

ReLU is applied after taking the centered signed representative, so the modular
models are hybrid finite-ring networks with a lookup/comparison nonlinearity.
The modular matrix multiplication itself is exact in the stated ring.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 experiments/run_small_networks.py
```

Results are written to `results/small_network_results.csv`, including:

- accuracy and accuracy loss from float;
- agreement with float predictions and mean absolute logit error;
- exact modular-wrap count and rate;
- maximum exact accumulator magnitude;
- MAC count and measured NumPy runtime;
- weight/bias storage, modulus entropy, and physical RNS lane width;
- clipping rates.

See [`RESULTS.md`](RESULTS.md) for the measured three-seed results and practical
conclusions.

See [`EXP_LOG.md`](EXP_LOG.md) for the complete research log, including the
Monna-map/signed-digit analysis, controlled GGUF comparisons, and block-wise
dyadic exponent experiments.

See [`PAPER.md`](PAPER.md) for a self-contained short paper presenting the
progressive dyadic method, large-model experiments, conclusions, and
limitations.

For a quick smoke test:

```bash
python3 experiments/run_small_networks.py \
  --datasets moons --operand-bits 4 --accumulator-bits 8 12 \
  --epochs 10 --timing-repeats 2
```

Run the progressive-prefix experiment:

```bash
python3 experiments/run_progressive_planes.py
```

Large-model experiments require MPS and locally downloaded model/data artifacts:

```bash
python3 experiments/run_resnet18_dyadic.py --help
python3 experiments/run_qwen_dyadic.py --help
python3 experiments/run_ollama_llm.py --help
```

Measured Apple M5 results are recorded in [`EXP_LOG.md`](EXP_LOG.md):

- ResNet-18: 6-bit dyadic weights reached 65.86% Imagenette top-1 versus
  66.85% FP16, using 9.43 MB versus 23.37 MB estimated stored size; the
  8-bit prefix reached 66.78%.
- Qwen2.5 0.5B: group-32 block-wise dyadic at 6.25 effective bits/weight
  reached WikiText-2 perplexity 28.88, 87.0% next-token agreement, and 47%
  ARC-Easy conditional-likelihood accuracy. This is comparable to Q4_K_M at
  6.35 effective bits/weight (PPL 28.64, 87.9% agreement, 37% ARC) while
  preserving the progressive-prefix property.
- Textual comparison: on 20 ARC-Easy free-generation prompts, group-32 6-bit
  dyadic reached 100% meaning equivalence against the BF16 source under the
  local embedding + LLM-judge pipeline; group-32 8-bit had the highest mean
  embedding cosine among tested variants.
- Native Ollama BF16-source generation ran on `100% GPU` at 4,456 prompt
  tokens/s and 219 generated tokens/s.

The MPS dyadic benchmarks currently materialize prefixes as FP16 because
PyTorch MPS has no packed progressive-weight operator. Their latency is not a
packed-kernel speed claim.

## Repository contents

- `dyadic_quant/`: reusable arithmetic, inference, dyadic encoding, and text
  generation helpers.
- `experiments/`: experiment drivers for small MLPs, ResNet-18, Qwen2.5,
  controlled GGUF baselines, block-size sweeps, and textual comparison.
- `tests/`: unit tests for arithmetic, inference, progressive nesting,
  block-wise dyadic encoding, and text-generation metrics.
- `results/`: small generated CSV/JSON summaries used by the writeups.

Large model checkpoints, GGUF files, downloaded datasets, and packed dyadic
artifacts are intentionally ignored by git. Recreate or download them locally
before running the large-model scripts.

## Interpretation

The wide-accumulator row is the correct conventional-quantization control.
Comparing an `N`-bit finite ring directly with an `N`-bit quantizer otherwise
hides the fact that deployed integer inference normally uses a wider
accumulator.

For RNS, the product of lane moduli determines the exact same wrap interval as
a single modulus of that product. Its possible advantage is implementation:
independent narrow multiply/add lanes without cross-lane carries. The NumPy
runtime is not a hardware-performance prediction because it includes residue
conversion and CRT reconstruction.
