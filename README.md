# Progressive dyadic quantization experiments

This repository contains progressive dyadic quantization experiments split into
two separate code areas:

1. Level 1 representation and quality: encode weights once as signed MSB-first
   dyadic bit planes, materialize prefixes, and measure quality/storage.
2. Level 2 native dyop execution: run packed dyadic primitives through native
   kernel/module surfaces and compare them against Level 1 materialized
   baselines.

## Run

```bash
python3 -m pip install -r requirements.txt
pytest tests/level1
pytest tests/level2
```

See [`EXP_LOG.md`](EXP_LOG.md) for the complete research log, including the
Monna-map/signed-digit analysis, controlled GGUF comparisons, and block-wise
dyadic exponent experiments.

See [`PAPER.md`](PAPER.md) for a self-contained short paper presenting the
progressive dyadic method, large-model experiments, conclusions, and
limitations.

Level 1 repros:

```bash
python3 experiments/level1/run_qwen_dyadic.py --help
python3 experiments/level1/run_dyadic_group_sweep.py --help
python3 experiments/level1/run_textual_comparison.py --help
```

Level 2 repros:

```bash
python3 experiments/level2/run_native_dyop_smoke.py --help
python3 experiments/level2/run_native_dyop_prefix_sweep.py --help
python3 experiments/level2/benchmark_native_kernels.py --help
```

Control and orchestration scripts live at `experiments/`. Large-model runs
require locally downloaded model/data artifacts; Level 1 MPS runs require MPS,
while Level 2 native CPU runs force CPU float32 execution.

```bash
python3 experiments/run_ollama_llm.py --help
python3 experiments/run_qwen_textual_global_rerun.py --help
```

Measured Apple M5 results are recorded in [`EXP_LOG.md`](EXP_LOG.md):

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

- `dyadic_quant/level1/`: reusable dyadic representation, packed artifact,
  materialization, and text-generation metric helpers.
- `dyadic_quant/level2/`: native dyop execution wrappers, module replacement,
  backend sources, and primitive catalogs.
- `experiments/level1/`: materialized Level 1 repros and text-comparison tools.
- `experiments/level2/`: native dyop smoke tests, kernel gates, and Level 1/Level 2 comparisons.
- `tests/level1/`: representation, artifact, storage, and text-metric tests.
- `tests/level2/`: scalar/native dyop kernel, module replacement, native CPU,
  and experiment-wiring tests.
- `results/`: Level 1 summaries, plus `results/level2/` for native dyop outputs.

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
