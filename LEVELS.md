# Dyadic Quantization Levels

## Level 1: Progressive dyadic quantization

Level 1 is the representation and quality layer.

- Code: `dyadic_quant/dyadic_torch.py`, `dyadic_quant/progressive.py`
- Repro scripts: existing scripts directly under `experiments/`
- Artifacts: existing CSV/JSON outputs under `results/`
- Scope: encode weights once, choose dyadic exponents, save packed sign/magnitude
  payloads, materialize prefixes for baseline PyTorch/Transformers evaluation.

Level 1 may decode or materialize prefixes. It exists to validate nested storage,
quality, memory estimates, and comparison against ordinary quantized baselines.

## Level 2: Native dyadic execution primitives

Level 2 is the execution-kernel layer.

- Code: `dyadic_quant/level2/`
- Tests: `tests/test_level2_kernels.py`
- Repro scripts: `experiments/level2/`
- Artifacts: `results/level2/`
- Scope: operate directly on signs, magnitude codes, and exponents without
  calling decoded-weight materialization, BLAS GEMM, PyTorch conv, or im2col as
  the implementation path.

Level 2 tests may compare against Level 1 materialized baselines, but Level 2
code must not use Level 1 decode/materialization internally.

The current Level 2 kernels are scalar reference kernels. They are deliberately
slow and explicit so the integer contract is clear before replacing them with
SIMD, Metal, or assembly backends.

The first compiled native backend lives under `dyadic_quant/level2/native/`.
It currently covers CPU float32 dense-input/dyadic-weight linear execution
(Linear, GEMV, GEMM, and output projection) and CPU int64-index dyadic
embedding lookup, plus CPU float32 NCHW dyadic Conv2d. It must be built
explicitly; it does not silently replace the scalar kernels.

The native CPU backend owns a persistent hot worker pool. Call
`warm_native_cpu_workers()` after setting `DYOP_CPU_THREADS` when timing kernels
or running a benchmark that should not include thread startup. Parallel dyop
kernels dispatch task slices into that pool instead of creating pthreads for
each Linear, Embedding, or Conv2d call.

Level 2 native also contains activation/spatial kernels needed to keep native
graphs from bouncing through Torch CPU: ReLU, add, fused add+ReLU, MaxPool2d,
and AdaptiveAvgPool2d.
ReLU is the first activation because it is a sign/zero check. Sigmoid and other
smooth activations should be added as dyadic approximations later; a base-2
sigmoid using `2^x` is the preferred direction over approximating `e^x`
directly.

The current Conv2d backend is functionally native but not yet a speed-superior
ResNet path. The focused benchmark is
`experiments/level2/benchmark_native_conv2d.py`, and its artifacts belong under
`results/level2/`. Do not treat ResNet native reruns as final until Conv2d
benchmarks beat the materialized PyTorch baseline on the target shapes.
The current Conv2d weakness is kernel layout, not dyadic arithmetic: Conv2d
must be moved toward tiled implicit-GEMM/dyop-dot workers and fused block
runners for Conv+ReLU and Conv+Add+ReLU.
The 1x1 ResNet downsample path now uses a native tiled dyop-dot worker, which
is materially faster than the previous direct NCHW worker but is still not
speed-superior to the materialized PyTorch baseline. ResNet 3x3 stride-2
block-entry convolutions use a native OC4 worker. ResNet 3x3 same-padding
convolutions use a native OC8 worker; a tiled 3x3 patch worker was tested and
removed from active dispatch because it was slower.
Conv2d dispatch caps active workers at 12 even when the hot pool is larger,
because local benchmarking showed the ResNet 3x3 and 1x1 shapes slow down past
that point on this CPU.

For ResNet Level 2 native runs, torchvision BasicBlock instances are wrapped so
the residual add plus final ReLU uses fused native add+ReLU. The current
balanced-200 quality/speed artifact with the stride-2 OC4 and same-padding OC8
Conv2d workers is
`results/level2/resnet_native_residual_balanced200_stride2oc4_same3x3oc8/`;
it is evidence for native quality/agreement and incremental speed progress,
not yet for speed supremacy.

`experiments/level2/benchmark_resnet_speed_gate.py` is the Level 2 ResNet
speed gate. It records materialized dyadic CPU latency, materialized dyadic MPS
latency when MPS is available, and native dyop CPU latency across isolated
worker-count subprocesses. Full ResNet quality runs should wait until both the
subkernel speed gates and the full-model native dyop rows pass. The current
local artifact is
`results/level2/resnet_speed_gate_bits6_r8.csv`: MPS is unavailable in this
runtime, and native dyops scale from one to twelve workers but still fail
against materialized CPU tensors. The current active-kernel rerun is
`results/level2/resnet_speed_gate_bits6_neon_dot_active_r8.csv`: 12 native
workers reach about 18.9 ms/image versus about 7.5 ms/image for materialized
dyadic CPU tensors in this runtime.

The native CPU packed-row dot path now has an ARM64 NEON implementation for
contiguous `float` by packed `int16` dyadic rows. That helps GEMV and
output-projection GEMV, but it does not solve the ResNet bottleneck by itself:
the remaining Conv2d work needs shape-specific 3x3/1x1 spatial tiling and
packed weight layouts that match the microkernel rather than per-output-plane
accumulation. Failed experimental microkernels have been removed from the
active native code path; their benchmark artifacts remain as evidence, but they
are not part of the current dispatch.

`experiments/level2/dyop_primitives.json` is the primitive catalog below layer
kernels. It defines architecture-specific dyop primitives for packing,
activation/window tiling, MRxNR microkernels, dyadic scale/bias, writeback,
elementwise, and reductions. `experiments/level2/dyop_kernel_op_trees.json`
then assembles those primitives into subkernel op-trees for ARM64/NEON and
future targets. `experiments/level2/check_subkernel_speed_gates.py` combines
focused benchmark CSVs into a pass/fail table against materialized tensor
baselines. The current cleaned report is
`results/level2/subkernel_speed_gates_cleaned_primitives.csv`: GEMV,
output-projection GEMV, ReLU, add, add+ReLU, and MaxPool2d pass; GEMM,
output-projection GEMM, embedding, all measured Conv2d shapes, and standalone
AdaptiveAvgPool2d fail.

`build_level2_model` replaces Level 1 encoded `Embedding`, `Linear`, and
`Conv2d` modules with Level 2 execution modules so small Level 1 models can be
replayed without writing decoded prefixes into their weights.

The Level 1 ResNet and Qwen experiment entrypoints now accept
`--execution-backend level2-native`. In that mode, prefix rows are executed by
Level 2 dyop modules instead of `materialize_prefix`; Level 1 materialization is
still available through the default `--execution-backend materialized` baseline.
When `--output-dir` is omitted, materialized runs write to `results/` and
Level 2 native runs write to `results/level2/`.
The ResNet script can route Level 2 `Linear` and `Conv2d` modules to
`native-cpu`; the Qwen script can route Level 2 `Embedding` and `Linear` modules
to `native-cpu`. Selecting any native CPU backend forces CPU float32 execution,
because the compiled kernels operate on CPU tensors and do not accept MPS or
FP16/BF16 inputs.

Packed Level 1 artifacts saved by `save_encoded_model` can be loaded back with
`load_encoded_model` and used as Level 2 dyop inputs. This keeps the storage
achievement and execution achievement connected through the serialized format
rather than only through in-memory Python objects.

## Cross-level textual reruns

Use `experiments/run_qwen_textual_global_rerun.py` when Level 1 and Level 2
textual metrics must be comparable. It writes one run directory with separate
subtrees:

- `level1_materialized/` for Level 1 full-tensor materialized generations and
  metrics.
- `level2_native_dyop/` for Level 2 native dyop generations and metrics.
- `audit/` for prompt/source equality checks and Level 1 vs Level 2 deltas.

The script requires an explicit `--judge-model`; do not use the Claude CLI
default for comparable reports. If fresh Level 1 generation is impossible
because MPS is unavailable, pass `--level1-generations-file` to rerun the metric
collection from an existing Level 1 generation artifact while keeping the new
metrics and metadata in the separated run tree.
