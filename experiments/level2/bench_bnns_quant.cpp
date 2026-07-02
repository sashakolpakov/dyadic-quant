// BNNS quantized GEMM experiment for dyadic quant linear layer
// Compile: clang++ -O2 -std=c++17 -framework Accelerate \
//           -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
//           -L/opt/homebrew/opt/libomp/lib -lomp \
//           -o bench_bnns_quant bench_bnns_quant.cpp

#include <Accelerate/Accelerate.h>
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
#include <omp.h>

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

constexpr double GATE_MS = 0.192396;

// ── BNNS quantized GEMM ──────────────────────────────────────────────
struct BnnsContext {
    BNNSFilter fc_filter = nullptr;
    int M = 0, N = 0, K = 0;
    avec<float> scale, bias;
};

static BnnsContext* create_bnns_gemm(int M, int N, int K,
                                      const int16_t* W,
                                      const float* scale,
                                      const float* bias) {
    auto* ctx = new BnnsContext;
    ctx->M = M; ctx->N = N; ctx->K = K;
    ctx->scale.assign(scale, scale + N);
    ctx->bias.assign(bias, bias + N);

    BNNSNDArrayDescriptor i_desc, w_desc, o_desc;
    memset(&i_desc, 0, sizeof(i_desc));
    memset(&w_desc, 0, sizeof(w_desc));
    memset(&o_desc, 0, sizeof(o_desc));

    // Input descriptor: M vectors of length K
    i_desc.flags = (BNNSNDArrayFlags)0;
    i_desc.layout = BNNSDataLayoutVector;
    i_desc.size[0] = static_cast<size_t>(K);
    i_desc.stride[0] = 1;
    i_desc.data = nullptr;
    i_desc.data_type = BNNSDataTypeFloat32;
    i_desc.data_scale = 1.0f;
    i_desc.data_bias = 0.0f;

    // Weight descriptor: N×K matrix, int16
    // BNNS weight layout: Weight(o,i) stored at i + o * in_size
    // So W[n][k] is at index k + n * K — row-major, output-col as outer
    w_desc.flags = (BNNSNDArrayFlags)0;
    w_desc.layout = BNNSDataLayoutRowMajorMatrix;
    w_desc.size[0] = static_cast<size_t>(N);
    w_desc.size[1] = static_cast<size_t>(K);
    w_desc.stride[0] = static_cast<size_t>(K);
    w_desc.stride[1] = 1;
    w_desc.data = const_cast<int16_t*>(W);
    w_desc.data_type = BNNSDataTypeInt16;
    w_desc.data_scale = 1.0f;
    w_desc.data_bias = 0.0f;

    // Output descriptor: M vectors of length N
    o_desc.flags = (BNNSNDArrayFlags)0;
    o_desc.layout = BNNSDataLayoutVector;
    o_desc.size[0] = static_cast<size_t>(N);
    o_desc.stride[0] = 1;
    o_desc.data = nullptr;
    o_desc.data_type = BNNSDataTypeFloat32;
    o_desc.data_scale = 1.0f;
    o_desc.data_bias = 0.0f;

    BNNSLayerParametersFullyConnected fc_params;
    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.i_desc = i_desc;
    fc_params.w_desc = w_desc;
    fc_params.o_desc = o_desc;
    // bias left as all-zero (no bias in FC; handled after)
    fc_params.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilterParameters fp = {0};
    fp.flags = BNNSFlagsUseClientPtr;

    ctx->fc_filter = BNNSFilterCreateLayerFullyConnected(&fc_params, &fp);
    if (!ctx->fc_filter) {
        std::cerr << "BNNSFilterCreateLayerFullyConnected failed!\n";
        delete ctx;
        return nullptr;
    }
    return ctx;
}

static void apply_bnns_gemm(BnnsContext* ctx,
                             const float* A,
                             float* C,
                             int batch_size) {
    BNNSFilterApplyBatch(ctx->fc_filter, batch_size, A, 0, C, 0);
    // Apply per-column scale and bias
    const int N = ctx->N;
    const float* s = ctx->scale.data();
    const float* b = ctx->bias.data();
    #pragma omp parallel for
    for (int m = 0; m < batch_size; ++m) {
        float* row = C + m * N;
        for (int n = 0; n < N; ++n)
            row[n] = row[n] * s[n] + b[n];
    }
}

static void destroy_bnns_gemm(BnnsContext* ctx) {
    if (ctx && ctx->fc_filter)
        BNNSFilterDestroy(ctx->fc_filter);
    delete ctx;
}

// ── Reference baseline (NEON repeated decode) ───────────────────────
static void gemm_baseline(
    const float* A, const int16_t* W_rm, const float* scales, const float* bias,
    float* C, int M, int N, int K, int threads
) {
    constexpr int NRn = 8;
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (int nb = 0; nb * NRn < N; ++nb) {
            int n0 = nb * NRn;
            int vn = std::min(NRn, N - n0);
            float32x4_t acc[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (int k = 0; k < K; ++k) {
                int16x8_t vw = vld1q_s16(W_rm + n0 * K + k);
                int32x4_t vl = vmovl_s16(vget_low_s16(vw));
                int32x4_t vh = vmovl_s16(vget_high_s16(vw));
                float32x4_t w0 = vcvtq_f32_s32(vl);
                float32x4_t w1 = vcvtq_f32_s32(vh);
                float32x4_t av = vdupq_n_f32(a_row[k]);
                acc[0] = vfmaq_f32(acc[0], av, w0);
                acc[1] = vfmaq_f32(acc[1], av, w1);
            }
            float32x4_t s0 = vld1q_f32(scales + n0);
            float32x4_t b0 = vld1q_f32(bias + n0);
            vst1q_f32(C + m * N + n0, vfmaq_f32(b0, acc[0], s0));
            if (vn > 4) {
                float32x4_t s1 = vld1q_f32(scales + n0 + 4);
                float32x4_t b1 = vld1q_f32(bias + n0 + 4);
                vst1q_f32(C + m * N + n0 + 4, vfmaq_f32(b1, acc[1], s1));
            }
        }
    }
}

// ── Accelerate gate (cblas_sgemm on materialized fp32) ──────────────
static void gemm_gate(const float* A, const int16_t* W_rm,
                       const float* scales, const float* bias,
                       float* C, int M, int N, int K, int threads) {
    omp_set_num_threads(threads);
    const int LDC = N;
    // Materialize W_f32 then call sgemm
    // (This IS what we want to avoid, but it's the gate baseline)
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        // Materialize W_f32[n][k] = W_int16[n][k] * scales[n]
        // (bias applied after sgemm)
        static thread_local avec<float> w_f32;
        w_f32.resize(K);
        const int16_t* src = W_rm + n * K;
        float s = scales[n];
        for (int k = 0; k < K; ++k) w_f32[k] = src[k] * s;
        // sgemm on A @ w_f32^T (single column of C)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                     M, 1, K, 1.0f, A, K, w_f32.data(), K, 0.0f,
                     C + n, LDC);
    }
    // Add bias
    #pragma omp parallel for
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m * N + n] += bias[n];
}

int main() {
    std::cerr << "=== BNNS Quantized GEMM Experiment ===\n\n";

    constexpr int M = 64, K = 896, N = 896;

    std::mt19937 rng(123);
    avec<float> A(static_cast<size_t>(M) * K);
    avec<int16_t> W_rm(static_cast<size_t>(N) * K);
    avec<float> scales(N), bias(N);

    for (auto& v : A) v = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);
    for (auto& v : W_rm) v = static_cast<int16_t>(std::uniform_int_distribution<int>(-31, 31)(rng));
    for (auto& v : scales) v = std::ldexp(1.0f, -(std::uniform_int_distribution<int>(3, 8)(rng)));
    for (auto& v : bias) v = std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng);

    // Reference (fp64)
    avec<float> ref_C(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(A[m * K + k]) * W_rm[n * K + k];
            ref_C[m * N + n] = static_cast<float>(acc * scales[n] + bias[n]);
        }

    auto check = [&](const float* C, const char* name) {
        bool ok = true;
        for (int t = 0; t < 64 && ok; ++t) {
            int m = rng() % M, n = rng() % N;
            float tol = 2e-3f * std::max(1.0f, std::abs(ref_C[m * N + n]));
            if (std::abs(C[m * N + n] - ref_C[m * N + n]) > tol) {
                std::cerr << "  " << name << " MISMATCH at " << m << "," << n
                          << " ref=" << ref_C[m * N + n] << " got=" << C[m * N + n] << "\n";
                ok = false;
            }
        }
        return ok;
    };

    // Create BNNS context
    auto* bnns_ctx = create_bnns_gemm(M, N, K, W_rm.data(), scales.data(), bias.data());
    if (!bnns_ctx) { std::cerr << "BNNS init failed\n"; return 1; }

    // Warm BNNS
    {
        avec<float> Ctmp(static_cast<size_t>(M) * N);
        apply_bnns_gemm(bnns_ctx, A.data(), Ctmp.data(), M);
        std::cerr << "  BNNS warmup ok=" << check(Ctmp.data(), "warm") << "\n";
    }

    // Benchmark BNNS
    {
        avec<float> Cbnns(static_cast<size_t>(M) * N, 0.0f);
        double ms = median_ms([&] {
            apply_bnns_gemm(bnns_ctx, A.data(), Cbnns.data(), M);
            g_sink += Cbnns[0] * 1e-30f;
        }, 5, 30, 7);
        bool ok = check(Cbnns.data(), "BNNS");
        std::cout << std::left << std::setw(36) << "BNNS quantized GEMM"
                  << std::right << std::fixed << std::setprecision(4)
                  << "  " << std::setw(9) << ms
                  << " ms  ratio=" << std::setw(7) << (GATE_MS / ms) << "x"
                  << "  gate=" << (ms < GATE_MS ? "PASS" : "FAIL")
                  << "  ok=" << (ok ? "yes" : "NO") << "  t=1\n";
    }

    destroy_bnns_gemm(bnns_ctx);

    // Benchmark NEON baseline and gate
    std::vector<int> threads = {1, 2, 4, 6, 8};

    for (int thr : threads) {
        avec<float> Ctmp(static_cast<size_t>(M) * N, 0.0f);
        double ms = median_ms([&] {
            gemm_baseline(A.data(), W_rm.data(), scales.data(), bias.data(),
                          Ctmp.data(), M, N, K, thr);
            g_sink += Ctmp[0] * 1e-30f;
        }, 3, 10, 5);
        bool ok = check(Ctmp.data(), "NEON baseline");
        std::cout << std::left << std::setw(36) << "NEON baseline (repeated decode)"
                  << std::right << std::fixed << std::setprecision(4)
                  << "  " << std::setw(9) << ms
                  << " ms  ratio=" << std::setw(7) << (GATE_MS / ms) << "x"
                  << "  gate=" << (ms < GATE_MS ? "PASS" : "FAIL")
                  << "  ok=" << (ok ? "yes" : "NO") << "  t=" << thr << "\n";
    }

    for (int thr : threads) {
        avec<float> Ctmp(static_cast<size_t>(M) * N, 0.0f);
        double ms = median_ms([&] {
            gemm_gate(A.data(), W_rm.data(), scales.data(), bias.data(),
                      Ctmp.data(), M, N, K, thr);
            g_sink += Ctmp[0] * 1e-30f;
        }, 3, 5, 5);
        bool ok = check(Ctmp.data(), "Gate");
        std::cout << std::left << std::setw(36) << "Gate (materialize+sgemm)"
                  << std::right << std::fixed << std::setprecision(4)
                  << "  " << std::setw(9) << ms
                  << " ms  ratio=" << std::setw(7) << (GATE_MS / ms) << "x"
                  << "  gate=" << (ms < GATE_MS ? "PASS" : "FAIL")
                  << "  ok=" << (ok ? "yes" : "NO") << "  t=" << thr << "\n";
    }

    std::cout << "\ngate: " << GATE_MS << " ms\n";
    std::cout << "sink=" << g_sink << "\n";
    return 0;
}
