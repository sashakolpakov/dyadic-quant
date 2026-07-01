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
  family, output projection, and Conv2d kernels against materialized CPU Torch
  baselines on representative Qwen/ResNet-like shapes. This is the first gate
  for dyop-kernel supremacy before expensive full-model metric reruns.
- `benchmark_resnet_speed_gate.py`: runs the Level 2 ResNet stop/go speed gate.
  It compares materialized dyadic CPU tensors, materialized dyadic MPS tensors
  when MPS is available, and native dyop CPU execution over an isolated worker
  sweep. Use this before any expensive ResNet quality run; if the native dyop
  rows do not pass the speed gate, fix kernels first.
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

Current speed artifacts:

- `results/level2/resnet_speed_gate_bits6_neon_dot_active_r8.csv`: full ResNet
  synthetic latency gate for the current active kernels. On the current local
  runtime, MPS is unavailable and native dyops scale from one to twelve workers
  but still fail against materialized CPU tensors.
- `results/level2/native_kernels_cleaned_primitives_bits6_r30.csv`: focused
  kernel benchmark after removing failed experimental microkernels and
  restoring the active ARM64/NEON packed-row dot path. GEMV and
  output-projection GEMV pass; GEMM, embedding, and Conv2d still need primitive
  work.
- `results/level2/native_conv2d_cleaned_primitives_bits6_r20.csv`: Conv2d-only
  benchmark after cleanup.
- `results/level2/subkernel_speed_gates_cleaned_primitives.csv`: combined subkernel gate
  report. Current passing subkernels are GEMV, output-projection GEMV, ReLU,
  add, add+ReLU, and MaxPool2d. Current failing subkernels are GEMM,
  output-projection GEMM, embedding, all measured Conv2d shapes, and standalone
  AdaptiveAvgPool2d.
- `results/level2/metal_gate_results.csv`: Metal GPU gate pass/fail for
  outproj, GEMM, embedding, and global pool. Outproj passes (6.15ms vs 10.84ms
  gate); GEMM (1.01ms vs 0.19ms), embedding (0.64ms vs 0.04ms), and pool
  (0.047ms vs 0.003ms) fail.
- `results/level2/metal_shmoo_tk16.csv`, `tk32.csv`, `tk64.csv`: Metal kernel
  tile-size shmoo. TK=16 is best; TK=32 and TK=64 are 1.2× and 1.5× slower.
- `results/level2/metal_shmoo_padded_bank_conflict.csv`: bank-conflict
  mitigation test (padding to 17) — no measurable improvement vs unpadded.
