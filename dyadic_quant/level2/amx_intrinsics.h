#pragma once
#include <stdint.h>

// AMX instruction macros from corsix/amx.
// These encode Apple's undocumented AMX instructions as .word directives.
// No Apple support guarantee; known to work on M1–M5 macOS.

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)  AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

// FMA32 encoding helpers.
// FMA32 does an outer product: Z[j*4+z_row&3][i] += X[i] * Y[j]
//   bits  0:8  = y_offset (byte offset into Y)
//   bits 10:18 = x_offset (byte offset into X)
//   bits 20:25 = z_row (bottom 2 bits select row group)
//   bits 27:29 = ALU mode (0=FMA, 1=mul, 2=add, 7=zero)
//   bits 32:38 = y_writemask (7-bit, 0=all enabled)
//   bits 41:47 = x_writemask (7-bit, 0=all enabled)
static inline uint64_t amx_enc_fma32(int y_off, int x_off, int z_row) {
    return ((uint64_t)(y_off & 0x1FF)) |
           (((uint64_t)(x_off & 0x1FF)) << 10) |
           (((uint64_t)(z_row & 0x3F)) << 20);
}

// Zero-out FMA32 encoding: writes 0 to all elements of the specified Z rows.
// z_row selects which 16-row group (z_row&3 = 0,1,2,3).
static inline uint64_t amx_enc_zero_z(int z_row) {
    return ((uint64_t)(z_row & 0x3F) << 20) | (7ull << 27);
}

// STZ encoding helper: store Z[z_row_idx] to dst.
// dst must be a user-space pointer (bits 56:63 = 0).
static inline uint64_t amx_enc_stz(const void* dst, int z_row_idx) {
    return (uint64_t)dst | (((uint64_t)(z_row_idx & 0x3F)) << 56);
}

// Barrier helper to order AMX operations with memory.
#define amx_barrier __asm volatile("dsb ish" : : : "memory")
