// ARM SME FP32 GEMM microkernel: MR=16, NR=16
// Computes C[16][16] += A[16][16] * W[16][16] (one K-step of 16)
// Actually this processes ONE K-element at a time:
//   ZA[i][j] += A[i][k] * W[k][j]
// So called K times with different k values.
//
// Arguments:
//   x0: activations pointer  — packed [16] per k, pointed at current k
//   x1: weight panel pointer — packed [16] per k, pointed at current k
//   x2: K (number of iterations)
//   x3: C panel pointer (output 16×16, ldc=16)

.globl _microkernel_sme_16x16
.align 4
_microkernel_sme_16x16:
        // Enter streaming mode
        smstart

        // Zero the ZA tile
        zero    {za}

        // ptrue for all elements
        ptrue   p0.s
        ptrue   p1.s

        // x0 = activations[k*16], x1 = weights[k*16], x2 = K
        cbz     x2, .Lsme_store

.Lsme_loop:
        // Load 16 activations
        ld1w    {z0.s}, p0/z, [x0]

        // Load 16 weights
        ld1w    {z1.s}, p1/z, [x1]

        // FMOPA: za0.s[p0, p1] += z0.s * z1.s (outer product)
        fmopa   za0.s, p0/m, p1/m, z0.s, z1.s

        // Advance pointers by 16 floats
        add     x0, x0, #64
        add     x1, x1, #64
        sub     x2, x2, #1
        cbnz    x2, .Lsme_loop

.Lsme_store:
        // Read ZA tile rows 0..15 and store to C panel
        // C is row-major with ldc=16 (64 bytes per row)
        // Each za0.s row has 4 horizontal sub-slices of 4 floats each
        mov     w12, #0
.Lsme_store_loop:
        // Read horizontal sub-slices of za0h.s[row w12]
        mova    z0.s, p0/m, za0h.s[w12, #0]     // elements [0:3]
        mova    z1.s, p0/m, za0h.s[w12, #1]     // elements [4:7]
        mova    z2.s, p0/m, za0h.s[w12, #2]     // elements [8:11]
        mova    z3.s, p0/m, za0h.s[w12, #3]     // elements [12:15]
        // Store all 16 floats of this row
        st1w    {z0.s}, p0, [x3]
        st1w    {z1.s}, p0, [x3, #1, mul vl]
        st1w    {z2.s}, p0, [x3, #2, mul vl]
        st1w    {z3.s}, p0, [x3, #3, mul vl]
        // Next row: C ldc=16 (64 bytes per row)
        add     x3, x3, #64
        add     w12, w12, #1
        cmp     w12, #16
        b.lo    .Lsme_store_loop

        // Exit streaming mode
        smstop

        ret
