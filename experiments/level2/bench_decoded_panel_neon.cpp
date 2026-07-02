// Decoded FP32 panel experiment for dyadic quant NEON GEMM.
// Tests whether decoding int16 weights once per K-block into a small FP32 panel,
// then reusing across all M-row tiles, beats the current per-tile repeated decode.
//
// Variants:
//   1. "baseline MR=4"    — current: microkernel decodes int16 every tile
//   2. "predecoded MR=4"  — entire weight matrix decoded to FP32 once up front (ceiling)
//   3. "panel MR=4"       — decode one KC×NR panel per K-block, reuse across M rows
//   4. "panel MR=8"       — same as 3, but MR=8 (lower decode overhead, better reuse)
//
// Compile: clang++ -march=armv8-a+fp+simd -O2 -std=c++17 -Xclang -fopenmp \
//           -I/opt/homebrew/opt/libomp/include microkernel_mr4_f32_asm.S \
//           bench_decoded_panel_neon.cpp -L/opt/homebrew/opt/libomp/lib -lomp \
//           -o bench_decoded_panel_neon \
//           && ./bench_decoded_panel_neon

#include <arm_neon.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dyop {

constexpr int NR = 8;
constexpr int ALIGN = 64;

static volatile float g_sink = 0.0f;

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<class U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, ALIGN, n * sizeof(T)) != 0) throw std::bad_alloc();
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
};

template <typename T>
using avec = std::vector<T, AlignedAllocator<T>>;

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

static inline void load_i16_as_f32_8(const int16_t* p, float32x4_t& lo, float32x4_t& hi) {
    int16x8_t v = vld1q_s16(p);
    lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
    hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v)));
}

static double median_ms(std::function<void()> fn, int warmup, int reps, int batches = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> vals;
    vals.reserve(batches);
    for (int b = 0; b < batches; ++b) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        vals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
    }
    std::sort(vals.begin(), vals.end());
    return vals[vals.size() / 2];
}

static void fill_random_float(avec<float>& x, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& v : x) v = dist(rng);
}

static void fill_random_i16(avec<int16_t>& x, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-31, 31);
    for (auto& v : x) {
        int q = dist(rng);
        if (q == 0) q = 1;
        v = static_cast<int16_t>(q);
    }
}

static bool check_result(const float* got, const float* ref, int n, double tol_rel = 2e-3) {
    for (int i = 0; i < n; ++i) {
        double tol = tol_rel * std::max(1.0, double(std::abs(double(ref[i]))));
        if (std::abs(double(got[i] - ref[i])) > tol) {
            std::cerr << "  MISMATCH i=" << i << " ref=" << ref[i] << " got=" << got[i] << "\n";
            return false;
        }
    }
    return true;
}

// ── microkernels ──────────────────────────────────────────────────────

// int16 decode + FMA: MR=4, NR=8. Applies scale+bias on store.
static void microkernel_mr4_i16(
    const float* A, int lda,
    const int16_t* W_block,
    const float* scale, const float* bias,
    float* C, int ldc,
    int K, int vm
) {
    float32x4_t acc0_lo = vdupq_n_f32(0), acc0_hi = vdupq_n_f32(0);
    float32x4_t acc1_lo = vdupq_n_f32(0), acc1_hi = vdupq_n_f32(0);
    float32x4_t acc2_lo = vdupq_n_f32(0), acc2_hi = vdupq_n_f32(0);
    float32x4_t acc3_lo = vdupq_n_f32(0), acc3_hi = vdupq_n_f32(0);
    const float* a0 = A; const float* a1 = A + lda;
    const float* a2 = A + 2 * lda; const float* a3 = A + 3 * lda;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t w0_lo, w0_hi, w1_lo, w1_hi, w2_lo, w2_hi, w3_lo, w3_hi;
        load_i16_as_f32_8(W_block + (k+0)*NR, w0_lo, w0_hi);
        load_i16_as_f32_8(W_block + (k+1)*NR, w1_lo, w1_hi);
        load_i16_as_f32_8(W_block + (k+2)*NR, w2_lo, w2_hi);
        load_i16_as_f32_8(W_block + (k+3)*NR, w3_lo, w3_hi);
        #define FMA_ROW(acc_lo, acc_hi, ap) do { \
            float32x4_t av = vld1q_f32(ap + k); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w0_lo, av, 0); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w0_hi, av, 0); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w1_lo, av, 1); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w1_hi, av, 1); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w2_lo, av, 2); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w2_hi, av, 2); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w3_lo, av, 3); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w3_hi, av, 3); \
        } while(0)
        if (vm > 0) FMA_ROW(acc0_lo, acc0_hi, a0);
        if (vm > 1) FMA_ROW(acc1_lo, acc1_hi, a1);
        if (vm > 2) FMA_ROW(acc2_lo, acc2_hi, a2);
        if (vm > 3) FMA_ROW(acc3_lo, acc3_hi, a3);
        #undef FMA_ROW
    }
    for (; k < K; ++k) {
        float32x4_t w_lo, w_hi; load_i16_as_f32_8(W_block + k*NR, w_lo, w_hi);
        if (vm > 0) { float32x4_t ak = vdupq_n_f32(a0[k]); acc0_lo = vfmaq_f32(acc0_lo, ak, w_lo); acc0_hi = vfmaq_f32(acc0_hi, ak, w_hi); }
        if (vm > 1) { float32x4_t ak = vdupq_n_f32(a1[k]); acc1_lo = vfmaq_f32(acc1_lo, ak, w_lo); acc1_hi = vfmaq_f32(acc1_hi, ak, w_hi); }
        if (vm > 2) { float32x4_t ak = vdupq_n_f32(a2[k]); acc2_lo = vfmaq_f32(acc2_lo, ak, w_lo); acc2_hi = vfmaq_f32(acc2_hi, ak, w_hi); }
        if (vm > 3) { float32x4_t ak = vdupq_n_f32(a3[k]); acc3_lo = vfmaq_f32(acc3_lo, ak, w_lo); acc3_hi = vfmaq_f32(acc3_hi, ak, w_hi); }
    }
    float32x4_t s0 = vld1q_f32(scale), s1 = vld1q_f32(scale+4);
    float32x4_t b0 = vld1q_f32(bias), b1 = vld1q_f32(bias+4);
    if (vm > 0) { vst1q_f32(C + 0*ldc, vfmaq_f32(b0, acc0_lo, s0)); vst1q_f32(C + 0*ldc+4, vfmaq_f32(b1, acc0_hi, s1)); }
    if (vm > 1) { vst1q_f32(C + 1*ldc, vfmaq_f32(b0, acc1_lo, s0)); vst1q_f32(C + 1*ldc+4, vfmaq_f32(b1, acc1_hi, s1)); }
    if (vm > 2) { vst1q_f32(C + 2*ldc, vfmaq_f32(b0, acc2_lo, s0)); vst1q_f32(C + 2*ldc+4, vfmaq_f32(b1, acc2_hi, s1)); }
    if (vm > 3) { vst1q_f32(C + 3*ldc, vfmaq_f32(b0, acc3_lo, s0)); vst1q_f32(C + 3*ldc+4, vfmaq_f32(b1, acc3_hi, s1)); }
}

// FP32×FP32 microkernel: MR=4, NR=8. Applies scale+bias on store.
static void microkernel_mr4_f32(
    const float* A, int lda,
    const float* W_f32,
    const float* scale, const float* bias,
    float* C, int ldc,
    int K, int vm
) {
    float32x4_t acc0_lo = vdupq_n_f32(0), acc0_hi = vdupq_n_f32(0);
    float32x4_t acc1_lo = vdupq_n_f32(0), acc1_hi = vdupq_n_f32(0);
    float32x4_t acc2_lo = vdupq_n_f32(0), acc2_hi = vdupq_n_f32(0);
    float32x4_t acc3_lo = vdupq_n_f32(0), acc3_hi = vdupq_n_f32(0);
    const float* a0 = A; const float* a1 = A + lda;
    const float* a2 = A + 2 * lda; const float* a3 = A + 3 * lda;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t w0_lo = vld1q_f32(W_f32 + (k+0)*NR);
        float32x4_t w0_hi = vld1q_f32(W_f32 + (k+0)*NR+4);
        float32x4_t w1_lo = vld1q_f32(W_f32 + (k+1)*NR);
        float32x4_t w1_hi = vld1q_f32(W_f32 + (k+1)*NR+4);
        float32x4_t w2_lo = vld1q_f32(W_f32 + (k+2)*NR);
        float32x4_t w2_hi = vld1q_f32(W_f32 + (k+2)*NR+4);
        float32x4_t w3_lo = vld1q_f32(W_f32 + (k+3)*NR);
        float32x4_t w3_hi = vld1q_f32(W_f32 + (k+3)*NR+4);
        #define FMA_ROW(acc_lo, acc_hi, ap) do { \
            float32x4_t av = vld1q_f32(ap + k); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w0_lo, av, 0); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w0_hi, av, 0); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w1_lo, av, 1); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w1_hi, av, 1); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w2_lo, av, 2); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w2_hi, av, 2); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w3_lo, av, 3); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w3_hi, av, 3); \
        } while(0)
        if (vm > 0) FMA_ROW(acc0_lo, acc0_hi, a0);
        if (vm > 1) FMA_ROW(acc1_lo, acc1_hi, a1);
        if (vm > 2) FMA_ROW(acc2_lo, acc2_hi, a2);
        if (vm > 3) FMA_ROW(acc3_lo, acc3_hi, a3);
        #undef FMA_ROW
    }
    for (; k < K; ++k) {
        float32x4_t w_lo = vld1q_f32(W_f32 + k*NR);
        float32x4_t w_hi = vld1q_f32(W_f32 + k*NR+4);
        if (vm > 0) { float32x4_t ak = vdupq_n_f32(a0[k]); acc0_lo = vfmaq_f32(acc0_lo, ak, w_lo); acc0_hi = vfmaq_f32(acc0_hi, ak, w_hi); }
        if (vm > 1) { float32x4_t ak = vdupq_n_f32(a1[k]); acc1_lo = vfmaq_f32(acc1_lo, ak, w_lo); acc1_hi = vfmaq_f32(acc1_hi, ak, w_hi); }
        if (vm > 2) { float32x4_t ak = vdupq_n_f32(a2[k]); acc2_lo = vfmaq_f32(acc2_lo, ak, w_lo); acc2_hi = vfmaq_f32(acc2_hi, ak, w_hi); }
        if (vm > 3) { float32x4_t ak = vdupq_n_f32(a3[k]); acc3_lo = vfmaq_f32(acc3_lo, ak, w_lo); acc3_hi = vfmaq_f32(acc3_hi, ak, w_hi); }
    }
    float32x4_t s0 = vld1q_f32(scale), s1 = vld1q_f32(scale+4);
    float32x4_t b0 = vld1q_f32(bias), b1 = vld1q_f32(bias+4);
    if (vm > 0) { vst1q_f32(C + 0*ldc, vfmaq_f32(b0, acc0_lo, s0)); vst1q_f32(C + 0*ldc+4, vfmaq_f32(b1, acc0_hi, s1)); }
    if (vm > 1) { vst1q_f32(C + 1*ldc, vfmaq_f32(b0, acc1_lo, s0)); vst1q_f32(C + 1*ldc+4, vfmaq_f32(b1, acc1_hi, s1)); }
    if (vm > 2) { vst1q_f32(C + 2*ldc, vfmaq_f32(b0, acc2_lo, s0)); vst1q_f32(C + 2*ldc+4, vfmaq_f32(b1, acc2_hi, s1)); }
    if (vm > 3) { vst1q_f32(C + 3*ldc, vfmaq_f32(b0, acc3_lo, s0)); vst1q_f32(C + 3*ldc+4, vfmaq_f32(b1, acc3_hi, s1)); }
}

// Raw FP32 accumulation microkernel: accumulates into existing C (adds).
static void microkernel_mr4_f32_raw(
    const float* A, int lda,
    const float* W_f32,
    float* C, int ldc,
    int K, int vm
) {
    float32x4_t acc0_lo = vdupq_n_f32(0), acc0_hi = vdupq_n_f32(0);
    float32x4_t acc1_lo = vdupq_n_f32(0), acc1_hi = vdupq_n_f32(0);
    float32x4_t acc2_lo = vdupq_n_f32(0), acc2_hi = vdupq_n_f32(0);
    float32x4_t acc3_lo = vdupq_n_f32(0), acc3_hi = vdupq_n_f32(0);
    const float* a0 = A; const float* a1 = A + lda;
    const float* a2 = A + 2 * lda; const float* a3 = A + 3 * lda;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t w0_lo = vld1q_f32(W_f32 + (k+0)*NR);
        float32x4_t w0_hi = vld1q_f32(W_f32 + (k+0)*NR+4);
        float32x4_t w1_lo = vld1q_f32(W_f32 + (k+1)*NR);
        float32x4_t w1_hi = vld1q_f32(W_f32 + (k+1)*NR+4);
        float32x4_t w2_lo = vld1q_f32(W_f32 + (k+2)*NR);
        float32x4_t w2_hi = vld1q_f32(W_f32 + (k+2)*NR+4);
        float32x4_t w3_lo = vld1q_f32(W_f32 + (k+3)*NR);
        float32x4_t w3_hi = vld1q_f32(W_f32 + (k+3)*NR+4);
        #define FMA_ROW(acc_lo, acc_hi, ap) do { \
            float32x4_t av = vld1q_f32(ap + k); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w0_lo, av, 0); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w0_hi, av, 0); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w1_lo, av, 1); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w1_hi, av, 1); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w2_lo, av, 2); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w2_hi, av, 2); \
            acc_lo = vfmaq_laneq_f32(acc_lo, w3_lo, av, 3); \
            acc_hi = vfmaq_laneq_f32(acc_hi, w3_hi, av, 3); \
        } while(0)
        if (vm > 0) FMA_ROW(acc0_lo, acc0_hi, a0);
        if (vm > 1) FMA_ROW(acc1_lo, acc1_hi, a1);
        if (vm > 2) FMA_ROW(acc2_lo, acc2_hi, a2);
        if (vm > 3) FMA_ROW(acc3_lo, acc3_hi, a3);
        #undef FMA_ROW
    }
    for (; k < K; ++k) {
        float32x4_t w_lo = vld1q_f32(W_f32 + k*NR);
        float32x4_t w_hi = vld1q_f32(W_f32 + k*NR+4);
        if (vm > 0) { float32x4_t ak = vdupq_n_f32(a0[k]); acc0_lo = vfmaq_f32(acc0_lo, ak, w_lo); acc0_hi = vfmaq_f32(acc0_hi, ak, w_hi); }
        if (vm > 1) { float32x4_t ak = vdupq_n_f32(a1[k]); acc1_lo = vfmaq_f32(acc1_lo, ak, w_lo); acc1_hi = vfmaq_f32(acc1_hi, ak, w_hi); }
        if (vm > 2) { float32x4_t ak = vdupq_n_f32(a2[k]); acc2_lo = vfmaq_f32(acc2_lo, ak, w_lo); acc2_hi = vfmaq_f32(acc2_hi, ak, w_hi); }
        if (vm > 3) { float32x4_t ak = vdupq_n_f32(a3[k]); acc3_lo = vfmaq_f32(acc3_lo, ak, w_lo); acc3_hi = vfmaq_f32(acc3_hi, ak, w_hi); }
    }
    if (vm > 0) {
        float32x4_t c0 = vld1q_f32(C + 0*ldc); float32x4_t c1 = vld1q_f32(C + 0*ldc+4);
        vst1q_f32(C + 0*ldc, vaddq_f32(c0, acc0_lo)); vst1q_f32(C + 0*ldc+4, vaddq_f32(c1, acc0_hi));
    }
    if (vm > 1) {
        float32x4_t c0 = vld1q_f32(C + 1*ldc); float32x4_t c1 = vld1q_f32(C + 1*ldc+4);
        vst1q_f32(C + 1*ldc, vaddq_f32(c0, acc1_lo)); vst1q_f32(C + 1*ldc+4, vaddq_f32(c1, acc1_hi));
    }
    if (vm > 2) {
        float32x4_t c0 = vld1q_f32(C + 2*ldc); float32x4_t c1 = vld1q_f32(C + 2*ldc+4);
        vst1q_f32(C + 2*ldc, vaddq_f32(c0, acc2_lo)); vst1q_f32(C + 2*ldc+4, vaddq_f32(c1, acc2_hi));
    }
    if (vm > 3) {
        float32x4_t c0 = vld1q_f32(C + 3*ldc); float32x4_t c1 = vld1q_f32(C + 3*ldc+4);
        vst1q_f32(C + 3*ldc, vaddq_f32(c0, acc3_lo)); vst1q_f32(C + 3*ldc+4, vaddq_f32(c1, acc3_hi));
    }
}

// ── MR=8 FP32 microkernel (for panel approach with 8-row tiles) ──────
// 16 accumulator vectors, 2 weight vectors, fits in 32 NEON registers.

static void microkernel_mr8_f32_raw(
    const float* A, int lda,
    const float* W_f32,
    float* C, int ldc,
    int K, int vm
) {
    float32x4_t acc[8][2];
    for (int r = 0; r < 8; ++r) { acc[r][0] = vdupq_n_f32(0); acc[r][1] = vdupq_n_f32(0); }
    const float* ap[8];
    for (int r = 0; r < 8; ++r) ap[r] = A + r * lda;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t w0_lo = vld1q_f32(W_f32 + (k+0)*NR);
        float32x4_t w0_hi = vld1q_f32(W_f32 + (k+0)*NR+4);
        float32x4_t w1_lo = vld1q_f32(W_f32 + (k+1)*NR);
        float32x4_t w1_hi = vld1q_f32(W_f32 + (k+1)*NR+4);
        float32x4_t w2_lo = vld1q_f32(W_f32 + (k+2)*NR);
        float32x4_t w2_hi = vld1q_f32(W_f32 + (k+2)*NR+4);
        float32x4_t w3_lo = vld1q_f32(W_f32 + (k+3)*NR);
        float32x4_t w3_hi = vld1q_f32(W_f32 + (k+3)*NR+4);
        for (int r = 0; r < vm; ++r) {
            float32x4_t av = vld1q_f32(ap[r] + k);
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], w0_lo, av, 0);
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], w0_hi, av, 0);
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], w1_lo, av, 1);
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], w1_hi, av, 1);
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], w2_lo, av, 2);
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], w2_hi, av, 2);
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], w3_lo, av, 3);
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], w3_hi, av, 3);
        }
    }
    for (; k < K; ++k) {
        float32x4_t w_lo = vld1q_f32(W_f32 + k*NR);
        float32x4_t w_hi = vld1q_f32(W_f32 + k*NR+4);
        for (int r = 0; r < vm; ++r) {
            float32x4_t ak = vdupq_n_f32(ap[r][k]);
            acc[r][0] = vfmaq_f32(acc[r][0], ak, w_lo);
            acc[r][1] = vfmaq_f32(acc[r][1], ak, w_hi);
        }
    }
    for (int r = 0; r < vm; ++r) {
        float32x4_t c0 = vld1q_f32(C + r*ldc);
        float32x4_t c1 = vld1q_f32(C + r*ldc+4);
        vst1q_f32(C + r*ldc, vaddq_f32(c0, acc[r][0]));
        vst1q_f32(C + r*ldc+4, vaddq_f32(c1, acc[r][1]));
    }
}

// ── assembly microkernel (extern) ──────────────────────────────────────
extern "C" void microkernel_mr4_f32_asm(
    const float* A, int64_t lda, const float* W,
    float* C, int64_t K, int64_t vm
);

// Assembly-based panel GEMM (MR=4, uses the assembly microkernel)
static void gemm_panel_asm(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MR = 4;
    constexpr int KC = 128;
    const int nblocks = ceil_div(N, NR);
    const int Np = nblocks * NR;
    const int mblocks = ceil_div(M, MR);
    avec<float> sp(Np, 0.0f), bp(Np, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    const int kblocks = ceil_div(K, KC);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        avec<float> cpanel(static_cast<size_t>(mblocks) * MR * NR, 0.0f);
        for (int kb = 0; kb < kblocks; ++kb) {
            int k0 = kb * KC;
            int ksize = std::min(KC, K - k0);
            alignas(64) float wpanel[KC * NR] = {};
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                if (n >= N) continue;
                const int16_t* row = W_rm + static_cast<size_t>(n)*K + k0;
                float* dst = wpanel + lane;
                for (int k = 0; k < ksize; ++k) dst[k*NR] = static_cast<float>(row[k]);
            }
            for (int mb = 0; mb < mblocks; ++mb) {
                int m0 = mb * MR;
                int vm = std::min(MR, M - m0);
                microkernel_mr4_f32_asm(
                    A + static_cast<size_t>(m0)*K + k0, K,
                    wpanel,
                    cpanel.data() + static_cast<size_t>(mb)*MR*NR,
                    ksize, vm);
            }
        }
        float32x4_t s0 = vld1q_f32(sp.data()+n0), s1 = vld1q_f32(sp.data()+n0+4);
        float32x4_t b0 = vld1q_f32(bp.data()+n0), b1 = vld1q_f32(bp.data()+n0+4);
        for (int mb = 0; mb < mblocks; ++mb) {
            int m0 = mb * MR;
            int vm = std::min(MR, M - m0);
            float* cp = cpanel.data() + static_cast<size_t>(mb)*MR*NR;
            for (int m = 0; m < vm; ++m) {
                float32x4_t al = vld1q_f32(cp + m*NR);
                float32x4_t ah = vld1q_f32(cp + m*NR+4);
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0, vfmaq_f32(b0, al, s0));
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0+4, vfmaq_f32(b1, ah, s1));
            }
        }
    }
}

// ── GEMM wrappers ─────────────────────────────────────────────────────

static void pack_i16_knr(const int16_t* W_rm, int16_t* W_packed, int N, int K, int Np) {
    const int nblocks = Np / NR;
    #pragma omp parallel for
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        int16_t* blk = W_packed + static_cast<size_t>(nb) * K * NR;
        for (int k = 0; k < K; ++k) {
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                blk[static_cast<size_t>(k)*NR + lane] = (n < N) ? W_rm[static_cast<size_t>(n)*K + k] : 0;
            }
        }
    }
}

static void pack_f32_knr(const int16_t* W_rm, float* W_f32, int N, int K, int Np) {
    const int nblocks = Np / NR;
    #pragma omp parallel for
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        float* blk = W_f32 + static_cast<size_t>(nb) * K * NR;
        for (int k = 0; k < K; ++k) {
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                blk[static_cast<size_t>(k)*NR + lane] = (n < N) ? static_cast<float>(W_rm[static_cast<size_t>(n)*K + k]) : 0.0f;
            }
        }
    }
}

// Variant 1: baseline — repeated int16 decode per tile
static void gemm_baseline(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MR = 4;
    const int nblocks = ceil_div(N, NR);
    const int Np = nblocks * NR;
    const int mblocks = ceil_div(M, MR);
    avec<int16_t> W_packed(static_cast<size_t>(Np) * K);
    pack_i16_knr(W_rm, W_packed.data(), N, K, Np);
    avec<float> sp(Np, 0.0f), bp(Np, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < mblocks; ++mb) {
        for (int nb = 0; nb < nblocks; ++nb) {
            int m0 = mb * MR, n0 = nb * NR;
            int vm = std::min(MR, M - m0);
            microkernel_mr4_i16(
                A + static_cast<size_t>(m0)*K, K,
                W_packed.data() + static_cast<size_t>(nb)*K*NR,
                sp.data()+n0, bp.data()+n0,
                C + static_cast<size_t>(m0)*N + n0, N,
                K, vm);
        }
    }
}

// Variant 2: predecoded FP32 ceiling
static void gemm_predecoded(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MR = 4;
    const int nblocks = ceil_div(N, NR);
    const int Np = nblocks * NR;
    const int mblocks = ceil_div(M, MR);
    avec<float> W_f32(static_cast<size_t>(Np) * K);
    pack_f32_knr(W_rm, W_f32.data(), N, K, Np);
    avec<float> sp(Np, 0.0f), bp(Np, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < mblocks; ++mb) {
        for (int nb = 0; nb < nblocks; ++nb) {
            int m0 = mb * MR, n0 = nb * NR;
            int vm = std::min(MR, M - m0);
            microkernel_mr4_f32(
                A + static_cast<size_t>(m0)*K, K,
                W_f32.data() + static_cast<size_t>(nb)*K*NR,
                sp.data()+n0, bp.data()+n0,
                C + static_cast<size_t>(m0)*N + n0, N,
                K, vm);
        }
    }
}

// Variant 3: panel-decoded, MR=4
static void gemm_panel_mr4(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MR = 4;
    constexpr int KC = 128;
    const int nblocks = ceil_div(N, NR);
    const int Np = nblocks * NR;
    const int mblocks = ceil_div(M, MR);
    avec<float> sp(Np, 0.0f), bp(Np, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    const int kblocks = ceil_div(K, KC);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        avec<float> cpanel(static_cast<size_t>(mblocks) * MR * NR, 0.0f);
        for (int kb = 0; kb < kblocks; ++kb) {
            int k0 = kb * KC;
            int ksize = std::min(KC, K - k0);
            alignas(64) float wpanel[KC * NR] = {};
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                if (n >= N) continue;
                const int16_t* row = W_rm + static_cast<size_t>(n)*K + k0;
                float* dst = wpanel + lane;
                for (int k = 0; k < ksize; ++k) dst[k*NR] = static_cast<float>(row[k]);
            }
            for (int mb = 0; mb < mblocks; ++mb) {
                int m0 = mb * MR;
                int vm = std::min(MR, M - m0);
                microkernel_mr4_f32_raw(
                    A + static_cast<size_t>(m0)*K + k0, K,
                    wpanel,
                    cpanel.data() + static_cast<size_t>(mb)*MR*NR, NR,
                    ksize, vm);
            }
        }
        // scale+bias + write
        float32x4_t s0 = vld1q_f32(sp.data()+n0), s1 = vld1q_f32(sp.data()+n0+4);
        float32x4_t b0 = vld1q_f32(bp.data()+n0), b1 = vld1q_f32(bp.data()+n0+4);
        for (int mb = 0; mb < mblocks; ++mb) {
            int m0 = mb * MR;
            int vm = std::min(MR, M - m0);
            float* cp = cpanel.data() + static_cast<size_t>(mb)*MR*NR;
            for (int m = 0; m < vm; ++m) {
                float32x4_t al = vld1q_f32(cp + m*NR);
                float32x4_t ah = vld1q_f32(cp + m*NR+4);
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0, vfmaq_f32(b0, al, s0));
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0+4, vfmaq_f32(b1, ah, s1));
            }
        }
    }
}

// Variant 4: panel-decoded, MR=8
static void gemm_panel_mr8(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MR = 8;
    constexpr int KC = 128;
    const int nblocks = ceil_div(N, NR);
    const int Np = nblocks * NR;
    const int mblocks = ceil_div(M, MR);
    avec<float> sp(Np, 0.0f), bp(Np, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    const int kblocks = ceil_div(K, KC);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        avec<float> cpanel(static_cast<size_t>(mblocks) * MR * NR, 0.0f);
        for (int kb = 0; kb < kblocks; ++kb) {
            int k0 = kb * KC;
            int ksize = std::min(KC, K - k0);
            alignas(64) float wpanel[KC * NR] = {};
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                if (n >= N) continue;
                const int16_t* row = W_rm + static_cast<size_t>(n)*K + k0;
                float* dst = wpanel + lane;
                for (int k = 0; k < ksize; ++k) dst[k*NR] = static_cast<float>(row[k]);
            }
            for (int mb = 0; mb < mblocks; ++mb) {
                int m0 = mb * MR;
                int vm = std::min(MR, M - m0);
                microkernel_mr8_f32_raw(
                    A + static_cast<size_t>(m0)*K + k0, K,
                    wpanel,
                    cpanel.data() + static_cast<size_t>(mb)*MR*NR, NR,
                    ksize, vm);
            }
        }
        float32x4_t s0 = vld1q_f32(sp.data()+n0), s1 = vld1q_f32(sp.data()+n0+4);
        float32x4_t b0 = vld1q_f32(bp.data()+n0), b1 = vld1q_f32(bp.data()+n0+4);
        for (int mb = 0; mb < mblocks; ++mb) {
            int m0 = mb * MR;
            int vm = std::min(MR, M - m0);
            float* cp = cpanel.data() + static_cast<size_t>(mb)*MR*NR;
            for (int m = 0; m < vm; ++m) {
                float32x4_t al = vld1q_f32(cp + m*NR);
                float32x4_t ah = vld1q_f32(cp + m*NR+4);
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0, vfmaq_f32(b0, al, s0));
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0+4, vfmaq_f32(b1, ah, s1));
            }
        }
    }
}

// ── benchmark ─────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Decoded FP32 Panel Experiment (ARM64 NEON) ===\n\n";

    const int M = 64, K = 896, N = 896;
    const double gate_ms = 0.192396;

    avec<float> A(static_cast<size_t>(M)*K), bias(N);
    avec<int16_t> W(static_cast<size_t>(N)*K);
    avec<float> scales(N);
    fill_random_float(A, 42, 0.25f);
    fill_random_float(bias, 43, 0.1f);
    fill_random_i16(W, 44);
    for (int n = 0; n < N; ++n) scales[n] = std::ldexp(1.0f, -5 - (n%3));

    // Reference
    avec<float> C_ref(static_cast<size_t>(M)*N);
    gemm_baseline(A.data(), W.data(), scales.data(), bias.data(), C_ref.data(), M, N, K, 1);

    struct Trial {
        const char* name;
        void (*gemm)(const float*, const int16_t*, const float*, const float*, float*, int, int, int, int);
    };
    Trial trials[] = {
        {"baseline MR=4 (repeated decode)", gemm_baseline},
        {"predecoded FP32 MR=4 (ceiling)",  gemm_predecoded},
        {"decode-once panel MR=4",          gemm_panel_mr4},
        {"decode-once panel MR=4 (asm)",    gemm_panel_asm},
        {"decode-once panel MR=8",          gemm_panel_mr8},
    };

    std::vector<int> thread_choices = {1, 2, 4, 6, 8, 10};

    for (auto& t : trials) {
        double best = 1e99;
        int best_t = 1;
        for (int thr : thread_choices) {
            double ms = median_ms([&] {
                avec<float> Cg(static_cast<size_t>(M)*N, 0.0f);
                t.gemm(A.data(), W.data(), scales.data(), bias.data(), Cg.data(), M, N, K, thr);
                g_sink += Cg[(M/2)*N + (N/2)] * 1e-30f;
            }, 3, 10, 5);
            std::cerr << "  " << t.name << " t=" << thr << "  " << ms << " ms\n";
            if (ms < best) { best = ms; best_t = thr; }
        }
        avec<float> Cg(static_cast<size_t>(M)*N);
        t.gemm(A.data(), W.data(), scales.data(), bias.data(), Cg.data(), M, N, K, best_t);
        bool ok = check_result(Cg.data(), C_ref.data(), M*N);
        double ratio = gate_ms / best;
        std::cout << std::left << std::setw(38) << t.name
                  << "  " << std::setw(8) << std::fixed << std::setprecision(4) << best << " ms"
                  << "  ratio=" << std::setw(6) << ratio << "x"
                  << "  gate=" << (best < gate_ms ? "PASS" : "FAIL")
                  << "  t=" << best_t
                  << "  ok=" << (ok ? "yes" : "NO") << "\n";
    }
    std::cout << "\ngate: " << gate_ms << " ms\nsink=" << g_sink << "\n";
    return 0;
}

} // namespace dyop

int main(int argc, char** argv) { return dyop::main(); }
