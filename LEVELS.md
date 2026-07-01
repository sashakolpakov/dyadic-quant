# Dyadic Quantization Levels

## Level 1: Representation and quality layer

Level 1 encodes weights once into an MSB-first dyadic code, selects power-of-two
exponents per group via normalized regret, saves packed sign/magnitude payloads,
and materializes prefixes for baseline PyTorch/Transformers evaluation.

- Code: `dyadic_quant/level1/dyadic_torch.py`, `dyadic_quant/level1/textgen.py`
- Repro scripts: `experiments/level1/run_resnet18_dyadic.py`, `experiments/level1/run_qwen_dyadic.py`,
  `experiments/level1/run_dyadic_group_sweep.py`, `experiments/level1/run_textual_comparison.py`
- Tests: `tests/level1/test_dyadic_torch.py`, `tests/level1/test_textgen.py`
- Repro test command: `pytest tests/level1`

Level 1 may decode or materialize prefixes. It validates nested storage, quality,
memory estimates, and comparison against ordinary quantized baselines.

## Level 2: Native dyadic execution primitives

Level 2 is the execution-kernel layer. It operates directly on signs, magnitude
codes, and exponents without calling decoded-weight materialization.

- Code: `dyadic_quant/level2/` (AVX-512 reference in `dyop_primitives.cpp`,
  ARM64/NEON porting guide in `ARM64_NEON_PORT.md`, op-tree definitions in `op_trees.json`)
- Tests: `tests/level2/test_kernels.py`, `tests/level2/test_modules.py`,
  `tests/level2/test_native_cpu.py`
- Repro test command: `pytest tests/level2`
- Repro scripts: `experiments/level2/`
- Artifacts: `results/level2/`

Level 2 tests may compare against Level 1 materialized baselines, but Level 2
code must not call into Level 1 decode/materialization internally.

The x86 AVX-512 implementation (`dyop_primitives.cpp`) is the tested architecture
instance and has beaten materialized Torch tensor baselines by about 2x on
average, with some subkernels near 7x. That result is the Level 2 target:
native dyop kernels should win by executing packed dyadic primitives, not by
silently falling back to Level 1 decoded tensors. The ARM64/NEON porting target
is described in `ARM64_NEON_PORT.md` with recommended tile geometry (MR=4,
NR=8), microkernel structure, and scheduling rules. The op-tree definitions in
`op_trees.json` and `experiments/level2/` (dyop_primitives.json,
dyop_kernel_op_trees.json) provide the primitive catalog and architecture-level
execution trees.

### Subkernel speed gates

Each dyop subkernel must pass a materialized-tensor speed gate before full-model
runs are meaningful. The current gate results are in
`dyadic_quant/level2/x86_avx512_gate_results.csv` (x86),
`dyadic_quant/level2/fixed_arm64_neon_gates.csv` (ARM64 CPU), and
`dyadic_quant/level2/fixed_metal_gates.csv` (Metal GPU).

Run `experiments/level2/check_subkernel_speed_gates.py` to produce the combined
pass/fail report.

### ARM64/NEON backend status

Gate pass/fail on Apple M5:

| Subkernel | Gate (ms) | NEON (ms) | Pass? |
|---|---|---|---|
| outproj (8×151k×896) | 10.84 | 0.34 | ✓ |
| output projection | 10.84 | 6.30 | ✓ |
| embedding | 0.0155 | 0.0053 | ✓ |
| global pool | 0.0133 | 0.0032 | ✓ |
| GEMM (64×896×896) | 0.192 | 0.315 | ✗ |
| ResNet 3×3 conv family | 0.229-3.936 | 0.419-8.764 | ✗ |
| ResNet 1×1 downsample | 0.266 | 0.383 | ✗ |

Latest artifact:
`results/level2/subkernel_speed_gates_arm64_neon_latest.csv`.

Callable native CPU linear status after packed int8 KNR weights, 8x8 NEON row
tiling, and a persistent worker pool:

| Shape | Torch materialized (ms) | Native dyop (ms) | Speedup |
|---|---:|---:|---:|
| Qwen MLP GEMV | 0.200 | 0.111 | 1.80x |
| Qwen sequence GEMM (64×896×896) | 0.097 | 0.209 | 0.47x |
| Qwen output projection GEMV | 6.862 | 3.928 | 1.75x |
| Qwen output projection (8×896×151936) | 17.749 | 5.548 | 3.20x |

Artifact:
`results/level2/native_linear_kernels_pool_i8knr_8x8u4_o3_threads12_bits6_r15.csv`.

Focused sequence-GEMM worker sweep on this runtime exposes 10 CPUs. Best clean
focused setting is 10 native workers. Current exact-path artifact:
`results/level2/native_linear_gemm_qwen_seq_exact_clean_threads10_bits6_r100.csv`
(`0.105 ms` materialized Torch, `0.253 ms` native dyop, `0.41x`; clean isolated
runs vary around `0.225-0.260 ms` native dyop). GEMV and
output projection pass because packed dyop codes reduce memory traffic versus
large dense float weights. Sequence GEMM fails because the same dense
materialized weights are reused across 64 activation rows, letting Accelerate's
dense GEMM amortize weight traffic and dominate on arithmetic throughput. Taller
12x8, wider 4x16, and 16x4 NEON tiles were tested and removed because they
regressed the sequence GEMM and/or output projection paths.

The materialized Torch sequence-GEMM gate remains strict under library thread
caps. With `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)`,
the focused artifact
`results/level2/native_linear_gemm_qwen_seq_threadcaps_threads10_bits6_r35.csv`
measured `0.103 ms` materialized Torch versus `0.261 ms` native dyop. This
runtime's PyTorch build reports `BLAS_INFO=accelerate`, so the materialized gate
is an Accelerate-backed dense GEMM baseline.

Level 2 Qwen and ResNet metric scripts now require passing speed gates before
they run native metric collection. Current ARM64/NEON gates are not sufficient
to repeat Level 1 Qwen/ResNet claims with native dyop execution: Qwen still
needs a faster GEMM path, and ResNet still needs faster 3×3 and 1×1 convolution
paths.

### Metal GPU backend status

A tiled Metal GPU backend (`dyadic_quant/level2/dyop_primitives_metal.mm`)
uses threadgroup memory + double-buffering with TK=16 tile size. Results:

| Subkernel | Gate (ms) | Metal (ms) | Pass? |
|---|---|---|---|
| outproj (8×151k×896) | 10.84 | 6.15 | ✓ (1.76× headroom) |
| GEMM (64×896×896) | 0.19 | 1.01 | ✗  (5.3× gate) |
| embedding (8×896×136M) | 0.04 | 0.64 | ✗  (16× gate) |
| global pool (8×896×49) | 0.003 | 0.047 | ✗  (15.7× gate) |
| conv (ResNet shapes) | — | — | not yet moved to tiled kernels |

**Analysis**: The bottleneck for tiny gates is not DRAM bandwidth (outproj
achieves 178 GB/s vs 200 GB/s peak) but GPU dispatch overhead. Embedding and
pool operate on ~8 rows — the GPU launch latency dominates the sub-0.1ms gate.
GEMM (64×896×896) is large enough to amortize dispatch but still 5.3× slow;
profiling shows the tiled kernel is compute-bound with 30–40% ALU utilization.
TK=16 outperformed TK=32 and TK=64. Bank conflicts are present but padding
(17-wide) did not measurably help.

The standalone Metal GEMM source now keeps tiled weights in int16 packed-code
form instead of host-side float-packed code. Sandboxed runs cannot see the Metal
device, but an unsandboxed M5 run produced
`results/level2/metal_gate_results_gemm_i16.csv`: `0.6336 ms` native Metal
versus the `0.1924 ms` fixed gate (`0.30x`, correct, fail). Metal output
projection can pass, but this packed-code Metal GEMM is not a viable Qwen
sequence-GEMM gate yet.

### Conv2d status

The Conv2d backend is functionally native but not yet speed-superior over
materialized PyTorch. The current dispatch uses OC4 workers for stride-2 block
entry, OC8 workers for same-padding 3x3, and a tiled dyop-dot worker for 1x1
downsample. The main bottleneck is kernel layout, not dyadic arithmetic. The
Metal conv kernels still use naive 1-thread-per-element dispatches and need
tiled optimizations.

`experiments/level2/benchmark_resnet_speed_gate.py` is the ResNet speed gate.
`experiments/level2/benchmark_native_conv2d.py` focuses on individual conv shapes.

## Cross-level textual reruns

Use `experiments/run_qwen_textual_global_rerun.py` when Level 1 and Level 2
textual metrics must be comparable. It writes one run directory with separate
`level1_materialized/`, `level2_native_dyop/`, and `audit/` subtrees.

## AVX to ARM64/NEON translation

The starting point for NEON porting is `dyadic_quant/level2/dyop_primitives.cpp`,
the tested AVX-512 implementation. The NEON porting map in
`dyadic_quant/level2/ARM64_NEON_PORT.md` specifies:

- MR=4, NR=8 tile geometry
- `int16x8_t` weight load, sign-extend via `vmovl_s16`, convert via `vcvtq_f32_s32`
- `float32x4_t` accumulators with `vfmaq_n_f32` for activation broadcast
- Persistent thread pool for parallelism
- Per-shape conv2d dispatch (1x1 stride-2 gather, 3x3 indirect window tile)

Do not admit a primitive into a layer kernel merely because it is correct; it must
first beat the corresponding row in `fixed_arm64_neon_gates.csv` with packing
excluded and without materializing decoded weights.
