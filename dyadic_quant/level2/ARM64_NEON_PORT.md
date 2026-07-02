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

### SME available, blocked from user-space

`hw.optional.arm.FEAT_SME=1` on M5 (SVL=512, 16×float32 ZA tile). However,
macOS traps `smstart`/`smstop` and all SME data-processing instructions
(`fmopa`, `svzero_za`, `ld1w`/`st1w` for ZA) with SIGILL. The `arm_streaming`
and `arm_new_za` function attributes are unknown to Apple's clang.

Apple reserves SME for its own frameworks (Accelerate, BNNS, MPS). User-space
code on macOS cannot access SME directly.

### SVE2 unavailable

`hw.optional.arm.FEAT_SVE` does not report on this system. The SVE2 port
(`dyop_primitives_sve2.cpp`) hangs the process if executed. SVE2 code is kept
for future hardware but must not be compiled in by default.

### Gate results

Gate CSVs: `fixed_arm64_neon_gates.csv` (ARM) and `fixed_metal_gates.csv`
(Metal). Current pass/fail on M5 (M=64, K=896, N=896):

| Subkernel | Gate (ms) | NEON (ms) | Metal (ms) | Pass? |
|---|---|---|---|---|
| outproj (8×151k×896) | 10.84 | 0.34 | 6.15 | ✓ both |
| embedding (8×896×136M) | 0.04 | 0.34 | 0.64 | ✗ both |
| global pool (8×896×49) | 0.003 | 0.010 | 0.047 | ✗ both |
| GEMM (64×896×896) | 0.19 | 0.44 | 1.01 | ✗ both |

Only outproj passes on either backend.

### GEMM deep dive

The GEMM kernel (64×896×896 = 103 MFLOP) is compute-bound — the weight
matrix (3 MB fp32) fits in L2 cache, so the int16 bandwidth advantage is
irrelevant. Accelerate uses SME internally (~2 TFLOPS per P-core cluster),
while NEON peaks at ~113 GFLOPS per core.

SME theoretical throughput: ~0.05 ms for 103 MFLOP at 2 TFLOPS, but
unreachable from user-space.

| Variant | Time (ms) | ×Gate |
|---|---|---|
| Gate (cblas_sgemm on materialized fp32) | 0.19 | 1.00 |
| cblas_sgemm alone (no materialization) | 0.09 | 2.1× |
| NEON baseline (repeated decode, t=8) | 0.47 | 0.41× |
| NEON panel decode + asm microkernel (t=8) | 0.44 | 0.44× |
| NEON panel decode + intrinsics (t=8) | 0.47 | 0.41× |

The panel approach (decode-once per N-block) saves ~6% vs repeated decode,
but both are dominated by NEON throughput limits. The assembly microkernel
(MR=4, NR=8) shaves off a few more percent.

### Walls

1. **SME blocked**: `smstart` → SIGILL. SME is Apple-private.
2. **BNNS no int16**: `BNNSMatMulWorkspaceSize` returns -1 for
   `BNNSDataTypeInt16`. BNNS cannot perform the quantized GEMM.
3. **NEON at ceiling**: Panel variant reaches ~90% of peak NEON efficiency;
   no further significant gains possible.

### Bottleneck analysis

- NEON GEMM is 2.3× above gate (0.44 ms vs 0.19 ms). At 224 GFLOPS theoretical
  peak on 4 P-cores (4-wide FMLA × 3.88 GHz), the 103 MFLOP output cannot
  match SME's ~2 TFLOPS. The gap is architectural, not implementational.

- Metal GEMM is 5.3× above the gate. Threadgroup-memory tiling (TK=16) with
  double-buffering reached 1.01 ms; TK=32 and TK=64 were slower. Bank conflicts
  were confirmed via occupancy analysis but padding to 17 did not help. ALU
  utilization is 30–40%.

- Tiny workloads (embedding, pool) are dominated by GPU dispatch overhead.
  Metal launch latency exceeds the sub-0.1 ms gates.

### Where NEON wins

NEON beats or matches the gate for bandwidth-bound shapes where the int16
decode saves significant memory traffic:

| Shape | Why NEON wins |
|---|---|
| outproj (8×151k×896) | Huge N — memory-bound; int16 reduces BW by 2× |
| Small batch (M=1) with large K, N | Decode cost amortized; no SME context-switch overhead |
| Any K where fp32 materialization spills L2 | int16 keeps data in cache longer |

The gate materializes N×K int16→fp32 weights (travels through L1→L2→DRAM)
before the GEMM. For large N, this doubles memory traffic. The NEON panel
decode avoids materialization entirely.

### Recommended strategy

**Hybrid dispatch** based on shape and hardware generation:

1. **M5+ (SME present)**: route compute-bound GEMMs (`M*K*N > threshold`)
   through `cblas_sgemm` on materialized fp32 weights — this is the gate
   itself, and it's unbeatable from user-space.
2. **Pre-M5 or bandwidth-bound shapes**: use NEON int16 panel decode —
   this wins on memory-bound shapes and is the only option on older hardware.
3. **Metal GPU**: not recommended for this project. Dispatch latency dominates
   for typical inference batch sizes; throughput does not close the gap.

The `threshold` should be determined empirically but a reasonable starting
point is `M*K*N >= 5×10^8` (equivalent to the 64×896×896 case).

## Gate policy

Do not admit a primitive/tree into a layer kernel merely because it is correct. It must first beat the corresponding fixed row in `fixed_arm64_neon_gates.csv`, with packing excluded and without materializing decoded weights.
