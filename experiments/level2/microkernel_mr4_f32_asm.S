//
// Hand-scheduled AArch64 NEON GEMM microkernel.
// MR=4, NR=8.  Accumulates C[i][j] += sum_k A[i][k] * W[k][j].
//
// Weights are already FP32 in KNR layout (K×8, row-major per K-step).
// This is the "raw" variant — adds into existing C (no scale/bias).
//
// The inner loop is 4× unrolled in K.  8 weight-vector loads are issued,
// then 4×8 FMLAs for each activation row.
//
// CALLING CONVENTION:
//   x0 = A      (4×K activation, stride = lda floats)
//   x1 = lda    (activation row stride in floats)
//   x2 = W      (K×8 weight panel, KNR layout)
//   x3 = C      (4×8 output, ldc=8, accumulates into existing values)
//   x4 = K      (number of K elements)
//   x5 = vm     (number of valid rows, 1..4)
//
// SCRATCH: x6-x15, v0-v31
// PRESERVED (callee-saves): x19-x28, v8-v15 must be saved if used
//   → we use v8-v15 without saving because they're scratch in AArch64
//     (lower 8 v-regs are callee-save, but we avoid v8-v15)

        .text
        .align 5
        .global _microkernel_mr4_f32_asm
_microkernel_mr4_f32_asm:

        // ── zero accumulators ──
        // v0-v1: row0  v2-v3: row1  v4-v5: row2  v6-v7: row3  (lo/hi each)
        movi    v0.4s, #0
        movi    v1.4s, #0
        movi    v2.4s, #0
        movi    v3.4s, #0
        movi    v4.4s, #0
        movi    v5.4s, #0
        movi    v6.4s, #0
        movi    v7.4s, #0

        // ── compute K rounded down to multiple of 4 ──
        mov     x14, x4                 // K
        and     x15, x14, #~3           // K_4 = K & ~3
        cbz     x15, .Ltail_prep        // if K_4 == 0, skip main loop

        // Set up row pointers
        // x6 = A + 0*lda, x7 = A + 1*lda, x8 = A + 2*lda, x9 = A + 3*lda
        mov     x6, x0
        add     x7, x0, x1, lsl #2      // A + lda*4 bytes
        add     x8, x7, x1, lsl #2      // A + 2*lda*4
        add     x9, x8, x1, lsl #2      // A + 3*lda*4

        // x12 = W + K_4 * 32  (32 bytes per K-step: 8 floats × 4 bytes)
        lsl     x12, x15, #5            // K_4 * 32
        add     x12, x2, x12            // W_end for main loop

        .align 4
.Lloop4:
        // Load 8 weight vectors for k..k+3
        ldp     q16, q17, [x2, #0]      // w[k+0] lo, hi
        ldp     q18, q19, [x2, #32]     // w[k+1] lo, hi
        ldp     q20, q21, [x2, #64]     // w[k+2] lo, hi
        ldp     q22, q23, [x2, #96]     // w[k+3] lo, hi

        // Row 0
        ld1     {v24.4s}, [x6], #16     // a0[k..k+3]
        fmla    v0.4s, v16.4s, v24.s[0]
        fmla    v1.4s, v17.4s, v24.s[0]
        fmla    v0.4s, v18.4s, v24.s[1]
        fmla    v1.4s, v19.4s, v24.s[1]
        fmla    v0.4s, v20.4s, v24.s[2]
        fmla    v1.4s, v21.4s, v24.s[2]
        fmla    v0.4s, v22.4s, v24.s[3]
        fmla    v1.4s, v23.4s, v24.s[3]

        // Row 1
        ld1     {v25.4s}, [x7], #16
        fmla    v2.4s, v16.4s, v25.s[0]
        fmla    v3.4s, v17.4s, v25.s[0]
        fmla    v2.4s, v18.4s, v25.s[1]
        fmla    v3.4s, v19.4s, v25.s[1]
        fmla    v2.4s, v20.4s, v25.s[2]
        fmla    v3.4s, v21.4s, v25.s[2]
        fmla    v2.4s, v22.4s, v25.s[3]
        fmla    v3.4s, v23.4s, v25.s[3]

        // Row 2
        ld1     {v26.4s}, [x8], #16
        fmla    v4.4s, v16.4s, v26.s[0]
        fmla    v5.4s, v17.4s, v26.s[0]
        fmla    v4.4s, v18.4s, v26.s[1]
        fmla    v5.4s, v19.4s, v26.s[1]
        fmla    v4.4s, v20.4s, v26.s[2]
        fmla    v5.4s, v21.4s, v26.s[2]
        fmla    v4.4s, v22.4s, v26.s[3]
        fmla    v5.4s, v23.4s, v26.s[3]

        // Row 3
        ld1     {v27.4s}, [x9], #16
        fmla    v6.4s, v16.4s, v27.s[0]
        fmla    v7.4s, v17.4s, v27.s[0]
        fmla    v6.4s, v18.4s, v27.s[1]
        fmla    v7.4s, v19.4s, v27.s[1]
        fmla    v6.4s, v20.4s, v27.s[2]
        fmla    v7.4s, v21.4s, v27.s[2]
        fmla    v6.4s, v22.4s, v27.s[3]
        fmla    v7.4s, v23.4s, v27.s[3]

        add     x2, x2, #128            // advance W by 4 K-steps × 32 bytes
        cmp     x2, x12
        b.lo    .Lloop4

        // ── tail: remaining 0..3 K values ──
.Ltail_prep:
        and     x10, x14, #3            // K % 4
        cbz     x10, .Lstore

        // For tail, use row0 = A + K_4, row1 = A + K_4 + lda, etc.
        // x6 = A + K_4*4,  x7 = A + lda*4 + K_4*4, etc.
        lsl     x13, x15, #2            // K_4 * 4 (bytes per float)
        add     x6, x0, x13
        add     x7, x0, x1, lsl #2
        add     x7, x7, x13
        add     x8, x7, x1, lsl #2
        add     x9, x8, x1, lsl #2

.Ltail_loop:
        ldp     q16, q17, [x2], #32     // w[k] lo, hi

        ld1     {v24.s}[0], [x6], #4    // a0[k]
        fmla    v0.4s, v16.4s, v24.s[0]
        fmla    v1.4s, v17.4s, v24.s[0]

        ld1     {v25.s}[0], [x7], #4
        fmla    v2.4s, v16.4s, v25.s[0]
        fmla    v3.4s, v17.4s, v25.s[0]

        ld1     {v26.s}[0], [x8], #4
        fmla    v4.4s, v16.4s, v26.s[0]
        fmla    v5.4s, v17.4s, v26.s[0]

        ld1     {v27.s}[0], [x9], #4
        fmla    v6.4s, v16.4s, v27.s[0]
        fmla    v7.4s, v17.4s, v27.s[0]

        subs    x10, x10, #1
        b.ne    .Ltail_loop

        // ── store accumulators ──
        // Add to existing C.  C is [4][8] with ldc=8 (32 bytes/row).
.Lstore:
        // Check vm for which rows to write
        cmp     x5, #4
        b.lo    1f
        // vm >= 4
        ldp     q24, q25, [x3, #0]
        fadd    v24.4s, v24.4s, v0.4s
        fadd    v25.4s, v25.4s, v1.4s
        stp     q24, q25, [x3, #0]
        ldp     q24, q25, [x3, #32]
        fadd    v24.4s, v24.4s, v2.4s
        fadd    v25.4s, v25.4s, v3.4s
        stp     q24, q25, [x3, #32]
        ldp     q24, q25, [x3, #64]
        fadd    v24.4s, v24.4s, v4.4s
        fadd    v25.4s, v25.4s, v5.4s
        stp     q24, q25, [x3, #64]
        ldp     q24, q25, [x3, #96]
        fadd    v24.4s, v24.4s, v6.4s
        fadd    v25.4s, v25.4s, v7.4s
        stp     q24, q25, [x3, #96]
        ret

1:      cmp     x5, #3
        b.lo    2f
        // vm = 3
        ldp     q24, q25, [x3, #0]
        fadd    v24.4s, v24.4s, v0.4s
        fadd    v25.4s, v25.4s, v1.4s
        stp     q24, q25, [x3, #0]
        ldp     q24, q25, [x3, #32]
        fadd    v24.4s, v24.4s, v2.4s
        fadd    v25.4s, v25.4s, v3.4s
        stp     q24, q25, [x3, #32]
        ldp     q24, q25, [x3, #64]
        fadd    v24.4s, v24.4s, v4.4s
        fadd    v25.4s, v25.4s, v5.4s
        stp     q24, q25, [x3, #64]
        ret

2:      cmp     x5, #2
        b.lo    3f
        // vm = 2
        ldp     q24, q25, [x3, #0]
        fadd    v24.4s, v24.4s, v0.4s
        fadd    v25.4s, v25.4s, v1.4s
        stp     q24, q25, [x3, #0]
        ldp     q24, q25, [x3, #32]
        fadd    v24.4s, v24.4s, v2.4s
        fadd    v25.4s, v25.4s, v3.4s
        stp     q24, q25, [x3, #32]
        ret

3:      // vm = 1
        ldp     q24, q25, [x3, #0]
        fadd    v24.4s, v24.4s, v0.4s
        fadd    v25.4s, v25.4s, v1.4s
        stp     q24, q25, [x3, #0]
        ret
