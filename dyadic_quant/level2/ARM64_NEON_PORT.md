# Exact ARM64/NEON porting map

The AVX-512 implementation is a tested architecture instance of the following portable tree. The Apple implementation should preserve the tree and change only tile geometry and intrinsics.

## Recommended starting tile

- `MR=4`
- `NR=8`
- packed weight layout: `[output_block][K][8]` as `int16`
- two `float32x4_t` accumulators per activation row (low/high output lanes)
- 8 accumulators total for an MR=4 tile

## Inner K step

For each `k`:

1. `int16x8_t q = vld1q_s16(w + k*8)`
2. `int32x4_t qlo32 = vmovl_s16(vget_low_s16(q))`
3. `int32x4_t qhi32 = vmovl_s16(vget_high_s16(q))`
4. `float32x4_t qlo = vcvtq_f32_s32(qlo32)`
5. `float32x4_t qhi = vcvtq_f32_s32(qhi32)`
6. For each activation row `m`:
   - `acc_lo[m] = vfmaq_n_f32(acc_lo[m], qlo, a[m*K+k])`
   - `acc_hi[m] = vfmaq_n_f32(acc_hi[m], qhi, a[m*K+k])`

After K:

- load two scale vectors and two bias vectors;
- `acc = vfmaq_f32(bias, acc, scale)`;
- store directly to the final output when contiguous;
- use a compact strided writeback only for NCHW convolution output.

## Required scheduling rules

- Weight packing occurs once, outside timed inference.
- GEMM partitions `(M-tile, N-tile)` over a persistent pthread pool.
- Conv3×3 partitions spatial MR tiles; each window tile is reused across its assigned output-channel blocks.
- Low-spatial/high-channel layers also split output-channel blocks to expose enough parallel tasks.
- 1×1 stride-2 bypasses generic window descriptors and gathers the four MR input points directly.
- Embedding keeps a row-major packed-code view and decodes selected rows directly.
- Global average pool uses a shape-specialized 49-value reduction and persistent workers; no generic dispatch or allocation.

## M5-specific findings

### SVE2 unavailable

`hw.optional.arm.FEAT_SVE=0` on this M5 generation. The SVE2 port
(`dyop_primitives_sve2.cpp`) hangs the process if executed. SVE2 code is kept
for future hardware but must not be compiled in by default.

### Gate results

Gate CSVs: `fixed_arm64_neon_gates.csv` (ARM) and `fixed_metal_gates.csv`
(Metal). Current pass/fail on M5:

| Subkernel | Gate (ms) | NEON (ms) | Metal (ms) | Pass? |
|---|---|---|---|---|
| outproj (8×151k×896) | 10.84 | 0.34 | 6.15 | ✓ both |
| embedding (8×896×136M) | 0.04 | 0.34 | 0.64 | ✗ both |
| global pool (8×896×49) | 0.003 | 0.010 | 0.047 | ✗ both |
| GEMM (64×896×896) | 0.19 | 0.33 | 1.01 | ✗ both |

Only outproj passes on either backend.

### Bottleneck analysis

- NEON GEMM is 1.7× above gate (0.33 ms vs 0.19 ms). At 224 GFLOPS theoretical
  peak (4-wide FMLA × 3.88 GHz), the 536 MFLOP output cannot be computed in
  0.19 ms on a single core. Multi-core dispatch (4 P-cores) may narrow the gap
  but the remaining factor-of-2 difference suggests packing overhead and
  int16→float conversion dominate.

- Metal GEMM is 5.3× above the gate. Threadgroup-memory tiling (TK=16) with
  double-buffering reached 1.01 ms; TK=32 and TK=64 were slower. Bank conflicts
  were confirmed via occupancy analysis but padding to 17 did not help. ALU
  utilization is 30–40%.

- Tiny workloads (embedding, pool) are dominated by GPU dispatch overhead.
  Metal launch latency exceeds the sub-0.1 ms gates.

### Recommended strategy

A hybrid dispatch routes tiny workloads through NEON CPU and large matmuls
through Metal GPU. GEMM throughput on both backends is still below the 0.19 ms
gate, so this is a direction, not a resolution.

## Gate policy

Do not admit a primitive/tree into a layer kernel merely because it is correct. It must first beat the corresponding fixed row in `fixed_arm64_neon_gates.csv`, with packing excluded and without materializing decoded weights.
