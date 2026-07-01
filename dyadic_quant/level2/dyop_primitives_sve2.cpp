// ARM64 SVE2 port of the dyadic primitives.
// Tile geometry: MR=4, NR=vl (SVE vector length, 64 on M5 with VL=2048).
// Weights are pre-converted from int16 to float32 at pack time
// and pre-multiplied by their scale factor, so the hot loop is
// just svmul+svadd (compiler-fused FMA) on 64-element vectors.
// Compile: clang++ -march=armv9-a+sve2 -O2 -std=c++17 -o dyop_sve2 dyop_primitives_sve2.cpp

#include <arm_sve.h>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace dyop {

constexpr int MR = 4;
constexpr int ALIGN = 64;

static volatile float g_sink = 0.0f;

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<class U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, ALIGN, n * sizeof(T)) != 0) throw std::bad_alloc();
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
};

template<class T, class U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }
template<class T, class U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }

template <typename T>
using avec = std::vector<T, AlignedAllocator<T>>;

struct PackedWeightF32 {
    int N = 0;
    int K = 0;
    int Np = 0;
    int vl = 0;                  // SVE vector length in floats
    avec<float> codes;           // [Np/vl][K][vl] float32 (pre-scaled)
    avec<float> bias;            // [Np]
};

struct RowMajorDyadicWeight {
    int N = 0;
    int K = 0;
    avec<int16_t> codes;         // [N][K]
    avec<float> scales;          // [N]
};

struct ConvShape {
    const char* name;
    int B, IC, IH, IW, OC, KH, KW, stride, pad;
};

// -----------------------------------------------------------------------
// Packing: int16 + scales → float32 × (pre-scaled)
// -----------------------------------------------------------------------

static int sve_vl() {
    static int vl = 0;
    if (vl == 0) vl = svcntw();
    return vl;
}

PackedWeightF32 pack_weight_f32(
    const int16_t* rowmajor_codes,
    const float* scales,
    const float* bias,
    int N,
    int K
) {
    int vl = sve_vl();
    PackedWeightF32 p;
    p.N = N; p.K = K; p.vl = vl;
    p.Np = ceil_div(N, vl) * vl;
    p.codes.resize(static_cast<size_t>(p.Np) * K);
    p.bias.assign(p.Np, 0.0f);
    for (int nb = 0; nb < p.Np / vl; ++nb) {
        for (int k = 0; k < K; ++k) {
            float* dst = p.codes.data() + (static_cast<size_t>(nb) * K + k) * vl;
            for (int lane = 0; lane < vl; ++lane) {
                int n = nb * vl + lane;
                dst[lane] = (n < N) ? static_cast<float>(rowmajor_codes[static_cast<size_t>(n) * K + k]) * scales[n] : 0.0f;
            }
        }
        for (int lane = 0; lane < vl; ++lane) {
            int n = nb * vl + lane;
            p.bias[n] = (n < N && bias) ? bias[n] : 0.0f;
        }
    }
    return p;
}

// -----------------------------------------------------------------------
// SVE2 GEMM microkernel: MR × vl (64 at VL=2048)
// -----------------------------------------------------------------------

static inline void microkernel_mr_nr_f32_sve2(
    const float* A,
    int lda,
    const float* W_block,
    const float* bias,
    float* C,
    int ldc,
    int K,
    int valid_m,
    int valid_n
) {
    int vl = sve_vl();
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    svbool_t pg_all = svptrue_b32();

    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;

    for (int k = 0; k < K; ++k) {
        svfloat32_t w = svld1_f32(pg_all, W_block + static_cast<size_t>(k) * vl);
        if (valid_m > 0) {
            svfloat32_t a = svdup_f32(a0[k]);
            acc0 = svadd_f32_x(pg_all, acc0, svmul_f32_x(pg_all, a, w));
        }
        if (valid_m > 1) {
            svfloat32_t a = svdup_f32(a1[k]);
            acc1 = svadd_f32_x(pg_all, acc1, svmul_f32_x(pg_all, a, w));
        }
        if (valid_m > 2) {
            svfloat32_t a = svdup_f32(a2[k]);
            acc2 = svadd_f32_x(pg_all, acc2, svmul_f32_x(pg_all, a, w));
        }
        if (valid_m > 3) {
            svfloat32_t a = svdup_f32(a3[k]);
            acc3 = svadd_f32_x(pg_all, acc3, svmul_f32_x(pg_all, a, w));
        }
    }

    svbool_t pg_store = svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(valid_n));
    svfloat32_t b = svld1_f32(pg_store, bias);
    if (valid_m > 0) svst1_f32(pg_store, C + 0 * ldc, svadd_f32_m(pg_store, acc0, b));
    if (valid_m > 1) svst1_f32(pg_store, C + 1 * ldc, svadd_f32_m(pg_store, acc1, b));
    if (valid_m > 2) svst1_f32(pg_store, C + 2 * ldc, svadd_f32_m(pg_store, acc2, b));
    if (valid_m > 3) svst1_f32(pg_store, C + 3 * ldc, svadd_f32_m(pg_store, acc3, b));
}

// -----------------------------------------------------------------------
// Linear GEMM
// -----------------------------------------------------------------------

void linear_gemm(
    const float* A,
    const PackedWeightF32& W,
    float* C,
    int M
) {
    int vl = W.vl;
    const int mblocks = ceil_div(M, MR);
    const int nblocks = W.Np / vl;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < mblocks; ++mb) {
        for (int nb = 0; nb < nblocks; ++nb) {
            const int m0 = mb * MR;
            const int n0 = nb * vl;
            const int vm = std::min(MR, M - m0);
            const int vn = std::min(vl, W.N - n0);
            const float* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * vl;
            microkernel_mr_nr_f32_sve2(
                A + static_cast<size_t>(m0) * W.K,
                W.K, wb,
                W.bias.data() + n0,
                C + static_cast<size_t>(m0) * W.N + n0,
                W.N, W.K, vm, vn
            );
        }
    }
}

// -----------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------

void embedding_decode(
    const int32_t* indices,
    int count,
    const RowMajorDyadicWeight& Wr,
    float* out
) {
    int vl = sve_vl();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const int row = indices[i];
        const int16_t* src = Wr.codes.data() + static_cast<size_t>(row) * Wr.K;
        float* dst = out + static_cast<size_t>(i) * Wr.K;
        const float scale = Wr.scales[row];
        // Process K in chunks of vl, then remainder
        int k = 0;
        svbool_t pg;
        for (; k + vl <= Wr.K; k += vl) {
            // Load vl int16, sign-extend to int32 via NEON widening, convert
            int16x8_t v[8];  // worst case: vl=64 → 8×8
            for (int j = 0; j < vl / 8; ++j)
                v[j] = vld1q_s16(src + k + j * 8);
            // We need vl float32 outputs. Convert each int16 to float32.
            // Use NEON for conversion, SVE for accumulate/store? Actually store to stack and load.
            float buf[64];
            for (int j = 0; j < vl / 8; ++j) {
                int32x4_t lo = vmovl_s16(vget_low_s16(v[j]));
                int32x4_t hi = vmovl_s16(vget_high_s16(v[j]));
                float32x4_t flo = vcvtq_f32_s32(lo);
                float32x4_t fhi = vcvtq_f32_s32(hi);
                vst1q_f32(buf + j * 8, flo);
                vst1q_f32(buf + j * 8 + 4, fhi);
            }
            pg = svptrue_b32();
            svfloat32_t x = svld1_f32(pg, buf);
            svfloat32_t s = svdup_f32(scale);
            svst1_f32(pg, dst + k, svmul_f32_x(pg, x, s));
        }
        pg = svwhilelt_b32(static_cast<uint32_t>(k), static_cast<uint32_t>(Wr.K));
        if (svptest_any(svptrue_b32(), pg)) {
            float buf[64];
            int r = 0;
            for (; k < Wr.K; ++k, ++r) buf[r] = static_cast<float>(src[k]) * scale;
            svfloat32_t x = svld1_f32(pg, buf);
            svst1_f32(pg, dst + k - r, x);
        }
    }
}

// -----------------------------------------------------------------------
// Conv2d (naive indirect gather into MR×K tile, then call microkernel)
// -----------------------------------------------------------------------

static inline void gather_activation_tile(
    const float* input,
    const ConvShape& s,
    int OH, int OW,
    int b, int p0, int valid_m,
    float* tile
) {
    int K = s.IC * s.KH * s.KW;
    for (int m = 0; m < valid_m; ++m) {
        const int pos = p0 + m;
        const int oh = pos / OW;
        const int ow = pos - oh * OW;
        float* dst = tile + static_cast<size_t>(m) * K;
        int kk = 0;
        for (int ic = 0; ic < s.IC; ++ic) {
            const float* base = input + ((static_cast<size_t>(b) * s.IC + ic) * s.IH) * s.IW;
            for (int kh = 0; kh < s.KH; ++kh) {
                const int ih = oh * s.stride + kh - s.pad;
                for (int kw = 0; kw < s.KW; ++kw, ++kk) {
                    const int iw = ow * s.stride + kw - s.pad;
                    dst[kk] = (static_cast<unsigned>(ih) < static_cast<unsigned>(s.IH) &&
                               static_cast<unsigned>(iw) < static_cast<unsigned>(s.IW))
                                  ? base[static_cast<size_t>(ih) * s.IW + iw]
                                  : 0.0f;
                }
            }
        }
    }
    for (int m = valid_m; m < MR; ++m)
        std::memset(tile + static_cast<size_t>(m) * K, 0, static_cast<size_t>(K) * sizeof(float));
}

void conv2d_indirect_gemm(
    const float* input,
    const PackedWeightF32& W,
    float* output,
    const ConvShape& s
) {
    int vl = W.vl;
    const int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    const int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    const int P = OH * OW;
    const int tiles_per_batch = ceil_div(P, MR);
    const int nblocks = W.Np / vl;
    const int total_tiles = s.B * tiles_per_batch;

    #pragma omp parallel
    {
        avec<float> tile(static_cast<size_t>(MR) * W.K);
        #pragma omp for schedule(static)
        for (int tg = 0; tg < total_tiles; ++tg) {
            const int b = tg / tiles_per_batch;
            const int tile_idx = tg - b * tiles_per_batch;
            const int p0 = tile_idx * MR;
            const int vm = std::min(MR, P - p0);
            gather_activation_tile(input, s, OH, OW, b, p0, vm, tile.data());

            for (int nb = 0; nb < nblocks; ++nb) {
                const int n0 = nb * vl;
                const int vn = std::min(vl, s.OC - n0);
                const float* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * vl;

                alignas(64) float tmp[MR * 64];
                microkernel_mr_nr_f32_sve2(
                    tile.data(), W.K, wb,
                    W.bias.data() + n0,
                    tmp, vl, W.K, vm, vn);

                for (int m = 0; m < vm; ++m) {
                    const int pos = p0 + m;
                    const int oh = pos / OW;
                    const int ow = pos - oh * OW;
                    for (int lane = 0; lane < vn; ++lane) {
                        const int oc = n0 + lane;
                        output[((static_cast<size_t>(b) * s.OC + oc) * OH + oh) * OW + ow] =
                            tmp[m * vl + lane];
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// Adaptive avgpool 1x1 (reduce49 specialization with SVE2)
// -----------------------------------------------------------------------

void adaptive_avgpool_1x1(const float* input, float* output, int B, int C, int H, int W) {
    int vl = sve_vl();
    const int HW = H * W;
    const int rows = B * C;
    const float inv = 1.0f / static_cast<float>(HW);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* x = input + static_cast<size_t>(r) * HW;
        if (HW == 49) {
            // hand-unrolled NEON reduce49 (fits in 128-bit registers)
            float32x4_t a = vld1q_f32(x);
            float32x4_t b = vld1q_f32(x + 4);
            float32x4_t c = vld1q_f32(x + 8);
            float32x4_t d = vld1q_f32(x + 12);
            float32x4_t e = vld1q_f32(x + 16);
            float32x4_t f = vld1q_f32(x + 20);
            float32x4_t g = vld1q_f32(x + 24);
            float32x4_t h = vld1q_f32(x + 28);
            float32x4_t ii = vld1q_f32(x + 32);
            float32x4_t j = vld1q_f32(x + 36);
            float32x4_t k = vld1q_f32(x + 40);
            float32x4_t l = vld1q_f32(x + 44);
            float32x4_t s1 = vaddq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)), vaddq_f32(vaddq_f32(e, f), vaddq_f32(g, h)));
            float32x4_t s2 = vaddq_f32(vaddq_f32(vaddq_f32(ii, j), k), l);
            output[r] = (vaddvq_f32(vaddq_f32(s1, s2)) + x[48]) * inv;
        } else {
            svfloat32_t acc = svdup_f32(0.0f);
            int i = 0;
            svbool_t pg;
            for (; i + vl <= HW; i += vl) {
                pg = svptrue_b32();
                acc = svadd_f32_x(pg, acc, svld1_f32(pg, x + i));
            }
            pg = svwhilelt_b32(static_cast<uint32_t>(i), static_cast<uint32_t>(HW));
            if (svptest_any(svptrue_b32(), pg))
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, x + i));
            float sum = svaddv_f32(svptrue_b32(), acc);
            output[r] = sum * inv;
        }
    }
}

// -----------------------------------------------------------------------
// Dot product for verification
// -----------------------------------------------------------------------

float dot_f32_i16(const float* activation, const int16_t* codes, int K) {
    int vl = sve_vl();
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    int k = 0;
    // Process 2×vl per iteration via NEON i16→f32 + SVE reduce
    for (; k + 2*vl <= K; k += 2*vl) {
        float buf[128];
        for (int j = 0; j < 2*vl / 8; ++j) {
            int16x8_t v = vld1q_s16(codes + k + j * 8);
            int32x4_t lo = vmovl_s16(vget_low_s16(v));
            int32x4_t hi = vmovl_s16(vget_high_s16(v));
            vst1q_f32(buf + j * 8, vcvtq_f32_s32(lo));
            vst1q_f32(buf + j * 8 + 4, vcvtq_f32_s32(hi));
        }
        svfloat32_t a0 = svld1_f32(svptrue_b32(), activation + k);
        svfloat32_t a1 = svld1_f32(svptrue_b32(), activation + k + vl);
        svfloat32_t w0 = svld1_f32(svptrue_b32(), buf);
        svfloat32_t w1 = svld1_f32(svptrue_b32(), buf + vl);
        acc0 = svadd_f32_x(svptrue_b32(), acc0, svmul_f32_x(svptrue_b32(), a0, w0));
        acc1 = svadd_f32_x(svptrue_b32(), acc1, svmul_f32_x(svptrue_b32(), a1, w1));
    }
    float sum = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), acc0, acc1));
    for (; k < K; ++k) sum += activation[k] * static_cast<float>(codes[k]);
    return sum;
}

// -----------------------------------------------------------------------
// Benchmark infrastructure
// -----------------------------------------------------------------------

static double median_ms(std::function<void()> fn, int warmup, int repeats, int batches = 7) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> vals;
    vals.reserve(batches);
    for (int b = 0; b < batches; ++b) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        vals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats);
    }
    std::sort(vals.begin(), vals.end());
    return vals[vals.size()/2];
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

struct Result {
    std::string name;
    double gate_ms;
    double best_ms;
    int threads;
    bool correct;
    std::string tree;
};

static bool check_linear_samples(
    const avec<float>& A,
    const RowMajorDyadicWeight& Wr,
    const avec<float>& bias,
    const avec<float>& C,
    int M, int N, int K
) {
    std::mt19937 rng(123);
    for (int t = 0; t < 24; ++t) {
        int m = rng() % M;
        int n = rng() % N;
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<double>(A[static_cast<size_t>(m)*K+k]) * Wr.codes[static_cast<size_t>(n)*K+k];
        float ref = static_cast<float>(acc * Wr.scales[n] + bias[n]);
        float got = C[static_cast<size_t>(m)*N+n];
        float tol = 2e-3f * std::max(1.0f, std::abs(ref));
        if (std::abs(ref-got) > tol) {
            std::cerr << "linear mismatch m="<<m<<" n="<<n<<" ref="<<ref<<" got="<<got<<"\n";
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Benchmarks
// -----------------------------------------------------------------------

static Result bench_linear_case(
    const std::string& name, int M, int K, int N, double gate_ms
) {
    avec<float> A(static_cast<size_t>(M)*K), bias(N), C(static_cast<size_t>(M)*N);
    RowMajorDyadicWeight Wr;
    Wr.N = N; Wr.K = K;
    Wr.codes.resize(static_cast<size_t>(N)*K);
    Wr.scales.resize(N);
    fill_random_float(A, 10 + M + N, 0.25f);
    fill_random_float(bias, 20 + N, 0.1f);
    fill_random_i16(Wr.codes, 30 + N);
    for (int n = 0; n < N; ++n) Wr.scales[n] = std::ldexp(1.0f, -5 - (n%3));
    PackedWeightF32 W = pack_weight_f32(Wr.codes.data(), Wr.scales.data(), bias.data(), N, K);

    int reps = (N > 100000) ? 1 : 10;
    double ms = median_ms([&]{ linear_gemm(A.data(), W, C.data(), M); g_sink = C[(M/2)*N + (N/2)] * 1e-30f; }, 3, reps, 5);
    linear_gemm(A.data(), W, C.data(), M);
    bool ok = check_linear_samples(A, Wr, bias, C, M, N, K);
    std::cerr << "  " << name << " ms=" << ms << (ok ? "" : " FAIL") << "\n";
    return {name, gate_ms, ms, 1, ok,
            "pack_weight_f32 -> microkernel_mr_nr_sve2"};
}

static Result bench_embedding_case(double gate_ms) {
    const int vocab = 151936, K = 896, count = 256;
    RowMajorDyadicWeight W;
    W.N = vocab; W.K = K;
    W.codes.resize(static_cast<size_t>(vocab)*K);
    W.scales.resize(vocab);
    fill_random_i16(W.codes, 88);
    for (int n = 0; n < vocab; ++n) W.scales[n] = std::ldexp(1.0f, -5 - (n%3));
    std::vector<int32_t> idx(count);
    for (int i = 0; i < count; ++i) idx[i] = (i * 593 + 17) % vocab;
    avec<float> out(static_cast<size_t>(count)*K);

    double ms = median_ms([&]{
        embedding_decode(idx.data(), count, W, out.data());
        g_sink = out[(count/2)*K] * 1e-30f;
    }, 10, 200, 5);

    embedding_decode(idx.data(), count, W, out.data());
    bool ok = true;
    for (int i = 0; i < 16 && ok; ++i) {
        int r = idx[i];
        for (int k = 0; k < K; k += 97) {
            float ref = static_cast<float>(W.codes[static_cast<size_t>(r)*K+k]) * W.scales[r];
            if (out[static_cast<size_t>(i)*K+k] != ref) { ok = false; break; }
        }
    }
    std::cerr << "  embedding ms=" << ms << (ok ? "" : " FAIL") << "\n";
    return {"embedding_qwen_vocab_width", gate_ms, ms, 1, ok,
            "embedding_decode_sve2"};
}

static bool check_conv_samples(
    const avec<float>& input,
    const RowMajorDyadicWeight& Wr,
    const avec<float>& bias,
    const avec<float>& out,
    const ConvShape& s
) {
    int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    std::mt19937 rng(778 + s.OC);
    for (int t = 0; t < 12; ++t) {
        int b = rng() % s.B, oc = rng() % s.OC, oh = rng() % OH, ow = rng() % OW;
        double acc = 0.0; int kk = 0;
        for (int ic = 0; ic < s.IC; ++ic)
            for (int kh = 0; kh < s.KH; ++kh)
                for (int kw = 0; kw < s.KW; ++kw, ++kk) {
                    int ih = oh * s.stride + kh - s.pad, iw = ow * s.stride + kw - s.pad;
                    if ((unsigned)ih < (unsigned)s.IH && (unsigned)iw < (unsigned)s.IW) {
                        float x = input[((static_cast<size_t>(b)*s.IC+ic)*s.IH+ih)*s.IW+iw];
                        acc += static_cast<double>(x) * Wr.codes[static_cast<size_t>(oc)*Wr.K+kk];
                    }
                }
        float ref = static_cast<float>(acc * Wr.scales[oc] + bias[oc]);
        float got = out[((static_cast<size_t>(b)*s.OC+oc)*OH+oh)*OW+ow];
        float tol = 3e-3f * std::max(1.0f, std::abs(ref));
        if (std::abs(ref-got) > tol) {
            std::cerr << "conv mismatch " << s.name << " ref=" << ref << " got=" << got << "\n";
            return false;
        }
    }
    return true;
}

static Result bench_conv_case(const ConvShape& s, double gate_ms) {
    int K = s.IC * s.KH * s.KW;
    int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    avec<float> input(static_cast<size_t>(s.B)*s.IC*s.IH*s.IW);
    avec<float> bias(s.OC);
    avec<float> out(static_cast<size_t>(s.B)*s.OC*OH*OW);
    RowMajorDyadicWeight Wr;
    Wr.N = s.OC; Wr.K = K;
    Wr.codes.resize(static_cast<size_t>(s.OC)*K);
    Wr.scales.resize(s.OC);
    fill_random_float(input, 400 + s.OC, 0.25f);
    fill_random_float(bias, 500 + s.OC, 0.1f);
    fill_random_i16(Wr.codes, 600 + s.OC);
    for (int n = 0; n < s.OC; ++n) Wr.scales[n] = std::ldexp(1.0f, -5 - (n%3));
    PackedWeightF32 W = pack_weight_f32(Wr.codes.data(), Wr.scales.data(), bias.data(), s.OC, K);

    int reps = (s.B == 8 && s.KH == 3) ? 2 : 10;
    auto fn = [&]{
        conv2d_indirect_gemm(input.data(), W, out.data(), s);
        g_sink = out[out.size()/2] * 1e-30f;
    };
    double ms = median_ms(fn, 3, reps, 5);
    conv2d_indirect_gemm(input.data(), W, out.data(), s);
    bool ok = check_conv_samples(input, Wr, bias, out, s);
    std::cerr << "  " << s.name << " ms=" << ms << (ok ? "" : " FAIL") << "\n";
    return {s.name, gate_ms, ms, 1, ok,
            "gather_activation_tile -> microkernel_mr_nr_sve2"};
}

static Result bench_pool_case(double gate_ms) {
    int B = 8, C = 512, H = 7, W = 7;
    avec<float> input(static_cast<size_t>(B)*C*H*W);
    avec<float> out(static_cast<size_t>(B)*C);
    fill_random_float(input, 111, 1.0f);

    double ms = median_ms([&]{
        adaptive_avgpool_1x1(input.data(), out.data(), B, C, H, W);
        g_sink = out[out.size()/2] * 1e-30f;
    }, 10, 300, 5);

    adaptive_avgpool_1x1(input.data(), out.data(), B, C, H, W);
    bool ok = true;
    for (int r = 0; r < B*C; r += 257) {
        double sum = 0;
        for (int i = 0; i < H*W; ++i) sum += input[static_cast<size_t>(r)*H*W + i];
        float ref = static_cast<float>(sum / (H*W));
        if (std::abs(ref - out[r]) > 1e-5f) { ok = false; break; }
    }
    std::cerr << "  pool ms=" << ms << (ok ? "" : " FAIL") << "\n";
    return {"adaptive_avgpool2d_resnet_global", gate_ms, ms, 1, ok,
            "adaptive_avgpool_sve2"};
}

static void write_csv(const std::string& path, const std::vector<Result>& rs) {
    std::ofstream f(path);
    f << "subkernel,materialized_gate_ms,arm64_sve2_ms,speedup_vs_gate,best_threads,passes_fixed_gate,correct,op_tree\n";
    for (auto& r : rs)
        f << r.name << ',' << std::fixed << std::setprecision(6) << r.gate_ms << ','
          << r.best_ms << ',' << (r.gate_ms / r.best_ms) << ',' << r.threads << ','
          << (r.best_ms < r.gate_ms ? "true" : "false") << ','
          << (r.correct ? "true" : "false") << ",\"" << r.tree << "\"\n";
}

int main(int argc, char** argv) {
    std::string out = "arm64_sve2_gate_results.csv";
    if (argc > 1) out = argv[1];

    std::cerr << "SVE2 vector length: " << sve_vl() << " floats\n";

    std::vector<Result> rs;
    const std::string only = std::getenv("DYOP_ONLY") ? std::getenv("DYOP_ONLY") : "all";
    auto want = [&](const std::string& key) { return only == "all" || only == key; };

    if (want("gemm")) {
        std::cerr << "bench linear_gemm_qwen_seq\n";
        rs.push_back(bench_linear_case("linear_gemm_qwen_seq", 64, 896, 896, 0.192396));
    }
    if (want("outproj")) {
        std::cerr << "bench linear_output_projection\n";
        rs.push_back(bench_linear_case("linear_output_projection", 8, 896, 151936, 10.843443));
    }
    if (want("embedding")) {
        std::cerr << "bench embedding\n";
        rs.push_back(bench_embedding_case(0.015501));
    }

    std::vector<std::pair<ConvShape, double>> convs = {
        {{"resnet_conv3x3",8,64,56,56,64,3,3,1,1},3.935540},
        {{"resnet_layer2_stride2_3x3",1,64,56,56,128,3,3,2,1},0.347237},
        {{"resnet_layer3_stride2_3x3",1,128,28,28,256,3,3,2,1},0.262369},
        {{"resnet_layer4_stride2_3x3",1,256,14,14,512,3,3,2,1},0.228865},
        {{"resnet_downsample",8,128,28,28,256,1,1,2,0},0.265602}
    };
    if (only == "all" || only == "conv")
        for (auto& [cs, g] : convs) rs.push_back(bench_conv_case(cs, g));
    if (only == "conv0") rs.push_back(bench_conv_case(convs[0].first, convs[0].second));
    if (only == "conv1") rs.push_back(bench_conv_case(convs[1].first, convs[1].second));
    if (only == "conv2") rs.push_back(bench_conv_case(convs[2].first, convs[2].second));
    if (only == "conv3") rs.push_back(bench_conv_case(convs[3].first, convs[3].second));
    if (only == "conv4") rs.push_back(bench_conv_case(convs[4].first, convs[4].second));

    if (want("pool")) {
        std::cerr << "bench adaptive pool\n";
        rs.push_back(bench_pool_case(0.013332));
    }

    write_csv(out, rs);
    std::cout << "CPU primitive profile: SVE2, VL=" << sve_vl() << " floats, MR=" << MR << "\n";
    for (auto& r : rs)
        std::cout << std::left << std::setw(38) << r.name
                  << " gate=" << std::right << std::setw(9) << std::fixed << std::setprecision(4) << r.gate_ms
                  << " ms  native=" << std::setw(9) << r.best_ms
                  << " ms  ratio=" << std::setw(7) << (r.gate_ms / r.best_ms) << "x"
                  << "  " << (r.best_ms < r.gate_ms ? "PASS" : "FAIL")
                  << "  correct=" << (r.correct ? "yes" : "NO") << "\n";
    std::cout << "Wrote " << out << " sink=" << g_sink << "\n";
    return std::all_of(rs.begin(), rs.end(), [](const Result& r) { return r.correct; }) ? 0 : 2;
}

} // namespace dyop

int main(int argc, char** argv) { return dyop::main(argc, argv); }
