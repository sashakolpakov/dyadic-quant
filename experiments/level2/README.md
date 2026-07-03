# Level 2 Kernel Experiments

Put native dyadic execution benchmarks and repro scripts here.

These scripts should benchmark `dyadic_quant.level2` kernels or later native
backends. Do not write Level 1 quantization-quality sweeps here.

Current entrypoint:

- `build_native_cpu.py`: builds the compiled Level 2 CPU linear microkernel
  shared library under `dyadic_quant/level2/native/`.
- `run_native_dyop_smoke.py`: builds a tiny mixed embedding/linear/conv model,
  encodes it through Level 1, executes the encoded weights through Level 2
  modules, and writes a Level 2 artifact comparing against the Level 1
  materialized baseline. Use `--reload-packed` to force Level 2 execution from
  the serialized packed dyadic artifact.
- `run_native_dyop_prefix_sweep.py`: saves one packed Level 1 dyadic artifact,
  reloads it for each requested prefix, executes through Level 2 dyop modules,
  and writes CSV/metadata rows comparable to Level 1 result tables. Use
  `--linear-backend native-cpu` to execute Linear/GEMV/GEMM/output projection
  through the compiled CPU microkernel, and `--embedding-backend native-cpu` to
  execute embedding lookup through the compiled CPU microkernel. Use
  `--conv-backend native-cpu` to execute Conv2d through the compiled CPU
  microkernel.
- `benchmark_native_kernels.py`: measures native dyop embedding, dense linear
  family, and output projection kernels against materialized CPU Torch baselines
  on representative Qwen shapes. This is the first gate for dyop-kernel
  supremacy before expensive full-model metric reruns.
- `profile_qwen_depth.py`: measures complete Qwen forwards and native dyop
  module time by sequence length. This captures the depth cost that remains
  after wide AVX kernels pass their isolated gate.
- `benchmark_qwen_mlp_flow.py`: compares disjoint native dyop Linear calls
  against bundled Qwen-style MLP flow. The bundled plan stores packed dyadic
  weights in native C++ objects and executes a short MLP stack without decoding
  weights.
- `dyop_primitives.json`: primitive catalog below layer kernels. It defines
  dyop packing, tiling, microkernel, scale/bias, writeback, elementwise, and
  reduction primitives and maps them to architecture profiles such as
  ARM64/NEON.
- `dyop_kernel_op_trees.json`: architecture-level execution trees for native
  dyop subkernels. Each tree is assembled from the primitive catalog and must
  pass its materialized-tensor subkernel gate before full-model runs are useful.
- `check_subkernel_speed_gates.py`: combines the focused kernel benchmark CSVs
  into a single pass/fail report. Run this before any full-model speed or
  quality sweep; full-model runs are not meaningful while required subkernels
  fail their materialized-tensor speed gate.
- `run_qwen_textual_generation.py`: generates Qwen text through Level 2 native
  dyop modules into a Level 2 generations JSON so cosine and LLM-judge scoring
  can be repeated without mixing Level 1 materialized textual artifacts.

Current Level 2 LLM interpretation:

- The kernel CSV is the width gate: it asks whether representative AVX native
  dyop kernels beat materialized Torch tensors for Qwen shapes.
- The depth CSV is the full-network execution probe: it asks how much time is
  lost crossing Python/PyTorch module boundaries and running the 24-layer graph.
- The quality CSV is the compression/accuracy evidence: it asks whether each
  dyadic bit width preserves Qwen perplexity, next-token agreement, and ARC
  behavior.

The next performance target is not another isolated GEMM tile. It is a native
Qwen block runner that keeps hidden states and layer intermediates in C++ across
RMSNorm, attention projections, attention reductions, MLP projections, and
residual adds.

Current LLM artifacts:

- `results/level2/<run-id>/kernels/qwen_native_kernels.csv`: isolated AVX
  width-kernel evidence.
- `results/level2/<run-id>/qwen_native/qwen25_level2_native_cpu_results.csv`:
  full Qwen quality, memory, ARC, and Wikitext evidence.
- `results/level2/<run-id>/depth/qwen_depth_profile.csv`: full-forward and
  per-module depth timing.
- `results/level2/<run-id>/textual/qwen25_dyop_generations.json`: optional
  Level 2 native textual generations.
- `results/level2/<run-id>/evidence/native_evidence_audit.md`: combined audit
  summary.
