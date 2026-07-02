// ARM SME + NEON dyadic GEMM benchmark
// - decode-once panel approach for int16 weights
// - SME FMOPA for fp32 panel matmul (MR=16, NR=16)
// - NEON baseline for comparison
// Compile: clang++ -march=armv9-a+sme -O2 -std=c++17 \
//           -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
//           -L/opt/homebrew/opt/libomp/lib -lomp \
//           -o bench_sme_panel_neon bench_sme_panel_neon.cpp \
//           microkernel_mr16_f32_sme.S microkernel_mr4_f32_asm.S

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
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

extern "C" void microkernel_sme_16x16(
    const float* A, const float* W, int64_t K, float* C
);

extern "C" void microkernel_mr4_f32_asm(
    const float* A, int64_t lda, const float* W,
    float* C, int64_t K, int64_t vm
);

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

constexpr int NR = 16;  // SME native
constexpr int MR = 16;
constexpr int KC = 128;

// ── helpers ──────────────────────────────────────────────────────────
template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<class U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, n * sizeof(T)) != 0) throw std::bad_alloc();
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
};
template<typename T>
using avec = std::vector<T, AlignedAllocator<T>>;

static float g_sink = 0.0f;

// ── reference (NEON baseline) ────────────────────────────────────────
static void gemm_baseline(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int NRn = 8;
    constexpr int MRn = 4;
    const int nblocks = ceil_div(N, NRn);
    const int mblocks = ceil_div(M, MRn);
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < mblocks; ++mb) {
        for (int nb = 0; nb < nblocks; ++nb) {
            int m0 = mb * MRn, n0 = nb * NRn;
            int vm = std::min(MRn, M - m0);
            int vn = std::min(NRn, N - n0);
            float32x4_t acc[4][2];
            for (int i = 0; i < 4; ++i) { acc[i][0] = vdupq_n_f32(0.0f); acc[i][1] = vdupq_n_f32(0.0f); }
            const float* a0 = A + m0*K;
            const float* a1 = a0 + K;
            const float* a2 = a1 + K;
            const float* a3 = a2 + K;
            for (int k = 0; k < K; ++k) {
                const int16_t* wb = W_rm + n0*K + k;
                int16x8_t vw = vld1q_s16(wb);
                int32x4_t vl = vmovl_s16(vget_low_s16(vw));
                int32x4_t vh = vmovl_s16(vget_high_s16(vw));
                float32x4_t w0 = vcvtq_f32_s32(vl);
                float32x4_t w1 = vcvtq_f32_s32(vh);
                if (vm > 0) {
                    float32x4_t av = vdupq_n_f32(a0[k]);
                    acc[0][0] = vfmaq_f32(acc[0][0], av, w0);
                    acc[0][1] = vfmaq_f32(acc[0][1], av, w1);
                }
                if (vm > 1) {
                    float32x4_t av = vdupq_n_f32(a1[k]);
                    acc[1][0] = vfmaq_f32(acc[1][0], av, w0);
                    acc[1][1] = vfmaq_f32(acc[1][1], av, w1);
                }
                if (vm > 2) {
                    float32x4_t av = vdupq_n_f32(a2[k]);
                    acc[2][0] = vfmaq_f32(acc[2][0], av, w0);
                    acc[2][1] = vfmaq_f32(acc[2][1], av, w1);
                }
                if (vm > 3) {
                    float32x4_t av = vdupq_n_f32(a3[k]);
                    acc[3][0] = vfmaq_f32(acc[3][0], av, w0);
                    acc[3][1] = vfmaq_f32(acc[3][1], av, w1);
                }
            }
            float32x4_t s0 = vld1q_f32(scales + n0);
            float32x4_t s1 = vld1q_f32(scales + n0 + 4);
            float32x4_t b0 = vld1q_f32(bias + n0);
            float32x4_t b1 = vld1q_f32(bias + n0 + 4);
            for (int m = 0; m < vm; ++m) {
                float* row = C + (m0+m)*N + n0;
                vst1q_f32(row,     vfmaq_f32(b0, acc[m][0], s0));
                vst1q_f32(row + 4, vfmaq_f32(b1, acc[m][1], s1));
            }
        }
    }
}

// ── NEON assembly panel variant ──────────────────────────────────────
static void gemm_panel_asm(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int MRn = 4;
    constexpr int NRn = 8;
    const int nblocks = ceil_div(N, NRn);
    const int mblocks = ceil_div(M, MRn);
    avec<float> sp(N, 0.0f), bp(N, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    const int kblocks = ceil_div(K, KC);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NRn;
        avec<float> cpanel(static_cast<size_t>(mblocks) * MRn * NRn, 0.0f);
        for (int kb = 0; kb < kblocks; ++kb) {
            int k0 = kb * KC;
            int ksize = std::min(KC, K - k0);
            alignas(64) float wpanel[KC * NRn] = {};
            for (int lane = 0; lane < NRn; ++lane) {
                int n = n0 + lane;
                if (n >= N) continue;
                const int16_t* row = W_rm + static_cast<size_t>(n)*K + k0;
                float* dst = wpanel + lane;
                for (int k = 0; k < ksize; ++k) dst[k*NRn] = static_cast<float>(row[k]);
            }
            for (int mb = 0; mb < mblocks; ++mb) {
                int m0 = mb * MRn;
                int vm = std::min(MRn, M - m0);
                microkernel_mr4_f32_asm(
                    A + static_cast<size_t>(m0)*K + k0, K,
                    wpanel,
                    cpanel.data() + static_cast<size_t>(mb)*MRn*NRn,
                    ksize, vm);
            }
        }
        float32x4_t s0 = vld1q_f32(sp.data()+n0), s1 = vld1q_f32(sp.data()+n0+4);
        float32x4_t b0 = vld1q_f32(bp.data()+n0), b1 = vld1q_f32(bp.data()+n0+4);
        for (int mb = 0; mb < mblocks; ++mb) {
            int m0 = mb * MRn;
            int vm = std::min(MRn, M - m0);
            float* cp = cpanel.data() + static_cast<size_t>(mb)*MRn*NRn;
            for (int m = 0; m < vm; ++m) {
                float32x4_t al = vld1q_f32(cp + m*NRn);
                float32x4_t ah = vld1q_f32(cp + m*NRn+4);
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0,
                          vfmaq_f32(b0, al, s0));
                vst1q_f32(C + static_cast<size_t>(m0+m)*N + n0+4,
                          vfmaq_f32(b1, ah, s1));
            }
        }
    }
}

// ── SME panel variant ────────────────────────────────────────────────
static void gemm_sme_panel(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int /*threads*/
) {
    const int nblocks = ceil_div(N, NR);
    const int mblocks = ceil_div(M, MR);
    avec<float> sp(N, 0.0f), bp(N, 0.0f);
    for (int n = 0; n < N; ++n) { sp[n] = scales[n]; bp[n] = bias[n]; }
    const int kblocks = ceil_div(K, KC);

    // SME is shared across P-core cluster — single-thread the matmul,
    // but we can parallelize the per-N-block loop (sequential SME per block)
    // Actually SME is per-cluster, so just use 1 thread.
    for (int nb = 0; nb < nblocks; ++nb) {
        int n0 = nb * NR;
        int vn = std::min(NR, N - n0);
        avec<float> cpanel(static_cast<size_t>(mblocks) * MR * NR, 0.0f);
        for (int kb = 0; kb < kblocks; ++kb) {
            int k0 = kb * KC;
            int ksize = std::min(KC, K - k0);
            // Decode weight panel: [lane][k] → [k][lane] for ld1w
            alignas(64) float wpanel[KC * NR] = {};
            for (int lane = 0; lane < vn; ++lane) {
                int n = n0 + lane;
                const int16_t* row = W_rm + static_cast<size_t>(n)*K + k0;
                float* dst = wpanel + lane;
                for (int k = 0; k < ksize; ++k) dst[k*NR] = static_cast<float>(row[k]);
            }
            for (int mb = 0; mb < mblocks; ++mb) {
                int m0 = mb * MR;
                int vm = std::min(MR, M - m0);
                // Pack activations: [row][k] → [k][row] for ld1w
                alignas(64) float apacked[KC * MR] = {};
                if (vm == MR) {
                    for (int i = 0; i < MR; ++i) {
                        const float* src = A + static_cast<size_t>(m0+i)*K + k0;
                        float* dst = apacked + i;
                        for (int k = 0; k < ksize; ++k) dst[k*MR] = src[k];
                    }
                } else {
                    for (int i = 0; i < vm; ++i) {
                        const float* src = A + static_cast<size_t>(m0+i)*K + k0;
                        float* dst = apacked + i;
                        for (int k = 0; k < ksize; ++k) dst[k*MR] = src[k];
                    }
                }
                // SME microkernel
                microkernel_sme_16x16(
                    apacked, wpanel, ksize,
                    cpanel.data() + static_cast<size_t>(mb)*MR*NR);
            }
        }
        // Apply scale+bias
        for (int mb = 0; mb < mblocks; ++mb) {
            int m0 = mb * MR;
            int vm = std::min(MR, M - m0);
            float* cp = cpanel.data() + static_cast<size_t>(mb)*MR*NR;
            for (int m = 0; m < vm; ++m) {
                for (int lane = 0; lane < vn; ++lane) {
                    int n = n0 + lane;
                    C[static_cast<size_t>(m0+m)*N + n] =
                        cp[m*NR + lane] * sp[n] + bp[n];
                }
            }
        }
    }
}

// ── Benchmark harness ────────────────────────────────────────────────
struct Trial {
    const char* name;
    void (*fn)(const float*, const int16_t*, const float*, const float*,
               float*, int, int, int, int);
};

static double median_ms(std::function<void()> fn, int warmup, int repeats, int batches = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> vals;
    for (int b = 0; b < batches; ++b) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        vals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats);
    }
    std::sort(vals.begin(), vals.end());
    return vals[vals.size() / 2];
}

static bool check_samples(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    const float* C, int M, int N, int K
) {
    std::mt19937 rng(42);
    for (int t = 0; t < 24; ++t) {
        int m = rng() % M;
        int n = rng() % N;
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<double>(A[static_cast<size_t>(m)*K + k]) * W_rm[static_cast<size_t>(n)*K + k];
        float ref = static_cast<float>(acc * scales[n] + bias[n]);
        float got = C[static_cast<size_t>(m)*N + n];
        float tol = 2e-3f * std::max(1.0f, std::abs(ref));
        if (std::abs(ref - got) > tol) {
            std::cerr << "  MISMATCH i=" << t << " ref=" << ref << " got=" << got << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::cerr << "=== SME+NEON Decoded Panel Experiment (M4) ===\n\n";

    constexpr int M = 64, K = 896, N = 896;
    constexpr double GATE_MS = 0.192396;

    std::mt19937 rng(123);
    avec<float> A(static_cast<size_t>(M)*K);
    avec<int16_t> W_rm(static_cast<size_t>(N)*K);
    avec<float> scales(N), bias(N);
    avec<float> C(static_cast<size_t>(M)*N);
    avec<float> ref_C(static_cast<size_t>(M)*N);

    for (auto& v : A) v = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);
    for (auto& v : W_rm) v = static_cast<int16_t>(std::uniform_int_distribution<int>(-31, 31)(rng));
    for (auto& v : scales) v = std::ldexp(1.0f, -(std::uniform_int_distribution<int>(3, 8)(rng)));
    for (auto& v : bias) v = std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng);

    // Reference: naive double-precision
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(A[m*K + k]) * W_rm[static_cast<size_t>(n)*K + k];
            ref_C[m*N + n] = static_cast<float>(acc * scales[n] + bias[n]);
        }

    std::vector<int> threads = {1, 2, 4, 6, 8, 10};
    omp_set_dynamic(0);

    Trial trials[] = {
        {"NEON baseline MR=4 (repeated decode)", gemm_baseline},
        {"NEON panel MR=4 (asm)",                gemm_panel_asm},
        {"SME panel MR=16",                      gemm_sme_panel},
    };

    for (auto& t : trials) {
        double best_ms = std::numeric_limits<double>::infinity();
        int best_t = 1;
        for (int thr : threads) {
            std::fill(C.begin(), C.end(), 0.0f);
            double ms = median_ms(
                [&] {
                    t.fn(A.data(), W_rm.data(), scales.data(), bias.data(),
                         C.data(), M, N, K, thr);
                    g_sink += C[0] * 1e-30f;
                },
                3, 10, 5);
            std::cerr << "  " << t.name << " t=" << thr << "  " << ms << " ms\n";
            if (ms < best_ms) { best_ms = ms; best_t = thr; }
        }
        // Re-run at best thread for correctness
        std::fill(C.begin(), C.end(), 0.0f);
        t.fn(A.data(), W_rm.data(), scales.data(), bias.data(),
             C.data(), M, N, K, best_t);
        bool ok = true;
        for (size_t i = 0; i < C.size(); ++i) {
            float tol = 2e-3f * std::max(1.0f, std::abs(ref_C[i]));
            if (std::abs(C[i] - ref_C[i]) > tol) { ok = false; break; }
        }
        if (ok) ok = check_samples(A.data(), W_rm.data(), scales.data(), bias.data(),
                                    C.data(), M, N, K);
        std::cout << std::left << std::setw(36) << t.name
                  << std::right << std::fixed << std::setprecision(4)
                  << "  " << std::setw(9) << best_ms
                  << " ms  ratio=" << std::setw(7) << (GATE_MS / best_ms) << "x"
                  << "  gate=" << (best_ms < GATE_MS ? "PASS" : "FAIL")
                  << "  ok=" << (ok ? "yes" : "NO")
                  << "  t=" << best_t << "\n";
    }

    std::cout << "\ngate: " << GATE_MS << " ms\n";
    std::cout << "sink=" << g_sink << "\n";
    return 0;
}
