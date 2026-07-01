// ARM64 NEON port of the AVX-512 dyadic primitives.
// Tile geometry: MR=4, NR=8 (NEON 128-bit vs AVX-512 512-bit).
// Each MR activation row uses 2 x float32x4_t accumulators for the 8 output lanes.
// Compile: clang++ -march=armv8-a+fp+simd -O2 -std=c++17 -o dyop_neon dyop_primitives_neon.cpp

#include <arm_neon.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cassert>
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
#include <tuple>
#include <vector>

namespace dyop {

constexpr int NR = 8;
constexpr int MR = 4;
constexpr int ALIGN = 64;

static volatile float g_sink = 0.0f;

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


// Persistent pthread-based worker pool (ports SpinPool from AVX-512).
class SpinPool {
public:
    using Fn = void(*)(void*, int, int);
    explicit SpinPool(int total_threads) : total_(std::max(1, total_threads)) {
        for (int tid = 1; tid < total_; ++tid) {
            workers_.emplace_back([this, tid] {
                uint64_t seen = 0;
                for (;;) {
                    uint64_t e;
                    while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
                        if (stop_.load(std::memory_order_relaxed)) return;
                        __asm__ __volatile__("yield" ::: "memory");
                    }
                    if (stop_.load(std::memory_order_relaxed)) return;
                    seen = e;
                    Fn f = fn_;
                    void* c = ctx_;
                    f(c, tid, total_);
                    done_.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }
    ~SpinPool() {
        stop_.store(true, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        for (auto& w : workers_) w.join();
    }
    int size() const { return total_; }
    SpinPool(const SpinPool&) = delete;
    SpinPool& operator=(const SpinPool&) = delete;
    void run(Fn fn, void* ctx) {
        if (total_ == 1) { fn(ctx, 0, 1); return; }
        done_.store(0, std::memory_order_relaxed);
        fn_ = fn; ctx_ = ctx;
        std::atomic_thread_fence(std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        fn(ctx, 0, total_);
        while (done_.load(std::memory_order_acquire) != total_ - 1)
            __asm__ __volatile__("yield" ::: "memory");
    }
private:
    int total_;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
};


struct PackedWeightKNR {
    int N = 0;
    int K = 0;
    int Np = 0;
    avec<int16_t> codes;  // [Nblock][K][NR]
    avec<float> scales;   // [Np]
    avec<float> bias;     // [Np]
};

struct RowMajorDyadicWeight {
    int N = 0;
    int K = 0;
    avec<int16_t> codes;  // [N][K]
    avec<float> scales;   // [N]
};

struct ConvShape {
    const char* name;
    int B, IC, IH, IW, OC, KH, KW, stride, pad;
};

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

PackedWeightKNR pack_weight_k_nr(
    const int16_t* rowmajor_codes,
    const float* scales,
    const float* bias,
    int N,
    int K
) {
    PackedWeightKNR p;
    p.N = N; p.K = K; p.Np = ceil_div(N, NR) * NR;
    p.codes.resize(static_cast<size_t>(p.Np) * K);
    p.scales.assign(p.Np, 0.0f);
    p.bias.assign(p.Np, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < p.Np / NR; ++nb) {
        int n0 = nb * NR;
        int16_t* dst_block = p.codes.data() + static_cast<size_t>(nb) * K * NR;
        for (int k = 0; k < K; ++k) {
            int16_t* dst = dst_block + static_cast<size_t>(k) * NR;
            for (int lane = 0; lane < NR; ++lane) {
                int n = n0 + lane;
                dst[lane] = (n < N) ? rowmajor_codes[static_cast<size_t>(n) * K + k] : 0;
            }
        }
        for (int lane = 0; lane < NR; ++lane) {
            int n = n0 + lane;
            p.scales[n] = (n < N) ? scales[n] : 0.0f;
            p.bias[n] = (n < N && bias) ? bias[n] : 0.0f;
        }
    }
    return p;
}

static inline void pack_activation_mr_k(
    const float* A,
    int lda,
    int valid_m,
    int K,
    float* tile
) {
    for (int m = 0; m < valid_m; ++m) {
        std::memcpy(tile + static_cast<size_t>(m) * K,
                    A + static_cast<size_t>(m) * lda,
                    static_cast<size_t>(K) * sizeof(float));
    }
    for (int m = valid_m; m < MR; ++m) {
        std::memset(tile + static_cast<size_t>(m) * K, 0,
                    static_cast<size_t>(K) * sizeof(float));
    }
}

// Load 8 int16 values, widen to int32x4_t (low/high), convert to float32x4_t.
static inline void load_i16_as_f32_8(const int16_t* p, float32x4_t& lo, float32x4_t& hi) {
    int16x8_t v = vld1q_s16(p);
    int32x4_t v_lo32 = vmovl_s16(vget_low_s16(v));
    int32x4_t v_hi32 = vmovl_s16(vget_high_s16(v));
    lo = vcvtq_f32_s32(v_lo32);
    hi = vcvtq_f32_s32(v_hi32);
}


float dot_f32_i16(const float* activation, const int16_t* codes, int K) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 7 < K; k += 8) {
        float32x4_t a0 = vld1q_f32(activation + k);
        float32x4_t a1 = vld1q_f32(activation + k + 4);
        float32x4_t w0, w1;
        load_i16_as_f32_8(codes + k, w0, w1);
        acc0 = vfmaq_f32(acc0, a0, w0);
        acc1 = vfmaq_f32(acc1, a1, w1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < K; ++k) sum += activation[k] * static_cast<float>(codes[k]);
    return sum;
}


void apply_scale_bias(float* tile, int rows, int cols, int stride,
                      const float* scale, const float* bias) {
    int c = 0;
    for (; c + 7 < cols; c += 8) {
        float32x4_t s0 = vld1q_f32(scale + c);
        float32x4_t s1 = vld1q_f32(scale + c + 4);
        float32x4_t b0 = vld1q_f32(bias + c);
        float32x4_t b1 = vld1q_f32(bias + c + 4);
        for (int r = 0; r < rows; ++r) {
            float* dst = tile + static_cast<size_t>(r) * stride + c;
            float32x4_t x0 = vld1q_f32(dst);
            float32x4_t x1 = vld1q_f32(dst + 4);
            vst1q_f32(dst, vfmaq_f32(b0, x0, s0));
            vst1q_f32(dst + 4, vfmaq_f32(b1, x1, s1));
        }
    }
    for (; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            tile[static_cast<size_t>(r) * stride + c] =
                tile[static_cast<size_t>(r) * stride + c] * scale[c] + bias[c];
}


void writeback_strided(const float* tile, int rows, int cols, int tile_stride,
                       float* output, int row_stride, int col_stride) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            output[static_cast<size_t>(r) * row_stride + static_cast<size_t>(c) * col_stride] =
                tile[static_cast<size_t>(r) * tile_stride + c];
}


enum class ElementwiseKind { Relu, Add, AddRelu };

void elementwise_fused(const float* a, const float* b, float* out,
                       size_t count, ElementwiseKind kind) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        float32x4_t x0 = vld1q_f32(a + i);
        float32x4_t x1 = vld1q_f32(a + i + 4);
        if (kind != ElementwiseKind::Relu) {
            float32x4_t y0 = vld1q_f32(b + i);
            float32x4_t y1 = vld1q_f32(b + i + 4);
            x0 = vaddq_f32(x0, y0);
            x1 = vaddq_f32(x1, y1);
        }
        if (kind != ElementwiseKind::Add) {
            x0 = vmaxq_f32(x0, zero);
            x1 = vmaxq_f32(x1, zero);
        }
        vst1q_f32(out + i, x0);
        vst1q_f32(out + i + 4, x1);
    }
    for (; i < count; ++i) {
        float x = a[i] + (kind == ElementwiseKind::Relu ? 0.0f : b[i]);
        out[i] = (kind == ElementwiseKind::Add) ? x : std::max(0.0f, x);
    }
}


float reduction_window(const float* x, int count, bool take_max) {
    if (count <= 0) return take_max ? -std::numeric_limits<float>::infinity() : 0.0f;
    float32x4_t acc = take_max ? vdupq_n_f32(-std::numeric_limits<float>::infinity())
                               : vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        if (take_max) {
            acc = vmaxq_f32(vmaxq_f32(acc, v0), v1);
        } else {
            acc = vaddq_f32(vaddq_f32(acc, v0), v1);
        }
    }
    float result = take_max ? vmaxvq_f32(acc) : vaddvq_f32(acc);
    for (; i < count; ++i)
        result = take_max ? std::max(result, x[i]) : result + x[i];
    return result;
}


// Core MRxNR primitive. MR=4, NR=8. Two float32x4_t accumulators per row.
// Uses vfmaq_laneq_f32 to broadcast activation scalars from a loaded vector,
// replacing separate scalar loads + vdupq_n_f32.
static inline void microkernel_mr_nr_f32_i16(
    const float* A,
    int lda,
    const int16_t* W_block,
    const float* scale,
    const float* bias,
    float* C,
    int ldc,
    int K,
    int valid_m,
    int valid_n
) {
    float32x4_t acc0_lo = vdupq_n_f32(0.0f);
    float32x4_t acc0_hi = vdupq_n_f32(0.0f);
    float32x4_t acc1_lo = vdupq_n_f32(0.0f);
    float32x4_t acc1_hi = vdupq_n_f32(0.0f);
    float32x4_t acc2_lo = vdupq_n_f32(0.0f);
    float32x4_t acc2_hi = vdupq_n_f32(0.0f);
    float32x4_t acc3_lo = vdupq_n_f32(0.0f);
    float32x4_t acc3_hi = vdupq_n_f32(0.0f);

    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t w0_lo, w0_hi, w1_lo, w1_hi, w2_lo, w2_hi, w3_lo, w3_hi;
        load_i16_as_f32_8(W_block + static_cast<size_t>(k + 0) * NR, w0_lo, w0_hi);
        load_i16_as_f32_8(W_block + static_cast<size_t>(k + 1) * NR, w1_lo, w1_hi);
        load_i16_as_f32_8(W_block + static_cast<size_t>(k + 2) * NR, w2_lo, w2_hi);
        load_i16_as_f32_8(W_block + static_cast<size_t>(k + 3) * NR, w3_lo, w3_hi);

        if (valid_m > 0) {
            float32x4_t a0v = vld1q_f32(a0 + k);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w0_lo, a0v, 0);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w0_hi, a0v, 0);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w1_lo, a0v, 1);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w1_hi, a0v, 1);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w2_lo, a0v, 2);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w2_hi, a0v, 2);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w3_lo, a0v, 3);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w3_hi, a0v, 3);
        }
        if (valid_m > 1) {
            float32x4_t a1v = vld1q_f32(a1 + k);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w0_lo, a1v, 0);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w0_hi, a1v, 0);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w1_lo, a1v, 1);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w1_hi, a1v, 1);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w2_lo, a1v, 2);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w2_hi, a1v, 2);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w3_lo, a1v, 3);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w3_hi, a1v, 3);
        }
        if (valid_m > 2) {
            float32x4_t a2v = vld1q_f32(a2 + k);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w0_lo, a2v, 0);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w0_hi, a2v, 0);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w1_lo, a2v, 1);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w1_hi, a2v, 1);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w2_lo, a2v, 2);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w2_hi, a2v, 2);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w3_lo, a2v, 3);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w3_hi, a2v, 3);
        }
        if (valid_m > 3) {
            float32x4_t a3v = vld1q_f32(a3 + k);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w0_lo, a3v, 0);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w0_hi, a3v, 0);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w1_lo, a3v, 1);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w1_hi, a3v, 1);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w2_lo, a3v, 2);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w2_hi, a3v, 2);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w3_lo, a3v, 3);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w3_hi, a3v, 3);
        }
    }
    for (; k < K; ++k) {
        float32x4_t w_lo, w_hi;
        load_i16_as_f32_8(W_block + static_cast<size_t>(k) * NR, w_lo, w_hi);
        if (valid_m > 0) {
            float32x4_t a_k = vdupq_n_f32(a0[k]);
            acc0_lo = vfmaq_f32(acc0_lo, a_k, w_lo);
            acc0_hi = vfmaq_f32(acc0_hi, a_k, w_hi);
        }
        if (valid_m > 1) {
            float32x4_t a_k = vdupq_n_f32(a1[k]);
            acc1_lo = vfmaq_f32(acc1_lo, a_k, w_lo);
            acc1_hi = vfmaq_f32(acc1_hi, a_k, w_hi);
        }
        if (valid_m > 2) {
            float32x4_t a_k = vdupq_n_f32(a2[k]);
            acc2_lo = vfmaq_f32(acc2_lo, a_k, w_lo);
            acc2_hi = vfmaq_f32(acc2_hi, a_k, w_hi);
        }
        if (valid_m > 3) {
            float32x4_t a_k = vdupq_n_f32(a3[k]);
            acc3_lo = vfmaq_f32(acc3_lo, a_k, w_lo);
            acc3_hi = vfmaq_f32(acc3_hi, a_k, w_hi);
        }
    }

    const float32x4_t s0 = vld1q_f32(scale);
    const float32x4_t s1 = vld1q_f32(scale + 4);
    const float32x4_t b0 = vld1q_f32(bias);
    const float32x4_t b1 = vld1q_f32(bias + 4);

    acc0_lo = vfmaq_f32(b0, acc0_lo, s0); acc0_hi = vfmaq_f32(b1, acc0_hi, s1);
    acc1_lo = vfmaq_f32(b0, acc1_lo, s0); acc1_hi = vfmaq_f32(b1, acc1_hi, s1);
    acc2_lo = vfmaq_f32(b0, acc2_lo, s0); acc2_hi = vfmaq_f32(b1, acc2_hi, s1);
    acc3_lo = vfmaq_f32(b0, acc3_lo, s0); acc3_hi = vfmaq_f32(b1, acc3_hi, s1);

    if (valid_m > 0) { vst1q_f32(C + 0 * ldc, acc0_lo); vst1q_f32(C + 0 * ldc + 4, acc0_hi); }
    if (valid_m > 1) { vst1q_f32(C + 1 * ldc, acc1_lo); vst1q_f32(C + 1 * ldc + 4, acc1_hi); }
    if (valid_m > 2) { vst1q_f32(C + 2 * ldc, acc2_lo); vst1q_f32(C + 2 * ldc + 4, acc2_hi); }
    if (valid_m > 3) { vst1q_f32(C + 3 * ldc, acc3_lo); vst1q_f32(C + 3 * ldc + 4, acc3_hi); }

    for (int m = 0; m < valid_m && valid_n < NR; ++m) {
        float* row = C + static_cast<size_t>(m) * ldc;
        for (int lane = valid_n; lane < NR; ++lane) row[lane] = 0.0f;
    }
}


void linear_gemm(
    const float* A,
    const PackedWeightKNR& W,
    float* C,
    int M,
    int threads
) {
    const int mblocks = ceil_div(M, MR);
    const int nblocks = W.Np / NR;
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < mblocks; ++mb) {
        for (int nb = 0; nb < nblocks; ++nb) {
            const int m0 = mb * MR;
            const int n0 = nb * NR;
            const int vm = std::min(MR, M - m0);
            const int vn = std::min(NR, W.N - n0);
            const int16_t* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * NR;
            microkernel_mr_nr_f32_i16(
                A + static_cast<size_t>(m0) * W.K,
                W.K,
                wb,
                W.scales.data() + n0,
                W.bias.data() + n0,
                C + static_cast<size_t>(m0) * W.N + n0,
                W.N,
                W.K,
                vm,
                vn
            );
        }
    }
}


void embedding_decode(
    const int32_t* indices,
    int count,
    const RowMajorDyadicWeight& W,
    float* out,
    int threads
) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const int row = indices[i];
        const int16_t* src = W.codes.data() + static_cast<size_t>(row) * W.K;
        float* dst = out + static_cast<size_t>(i) * W.K;
        const float32x4_t scale = vdupq_n_f32(W.scales[row]);
        int k = 0;
        for (; k + 15 < W.K; k += 16) {
            float32x4_t x0_lo, x0_hi, x1_lo, x1_hi;
            load_i16_as_f32_8(src + k, x0_lo, x0_hi);
            load_i16_as_f32_8(src + k + 8, x1_lo, x1_hi);
            vst1q_f32(dst + k, vmulq_f32(x0_lo, scale));
            vst1q_f32(dst + k + 4, vmulq_f32(x0_hi, scale));
            vst1q_f32(dst + k + 8, vmulq_f32(x1_lo, scale));
            vst1q_f32(dst + k + 12, vmulq_f32(x1_hi, scale));
        }
        for (; k + 7 < W.K; k += 8) {
            float32x4_t w_lo, w_hi;
            load_i16_as_f32_8(src + k, w_lo, w_hi);
            vst1q_f32(dst + k, vmulq_f32(w_lo, scale));
            vst1q_f32(dst + k + 4, vmulq_f32(w_hi, scale));
        }
        for (; k < W.K; ++k) dst[k] = static_cast<float>(src[k]) * W.scales[row];
    }
}


struct EmbeddingContext {
    const int32_t* indices;
    int count;
    const RowMajorDyadicWeight* W;
    float* out;
};

static void embedding_worker(void* opaque, int tid, int nt) {
    auto* c = static_cast<EmbeddingContext*>(opaque);
    int begin = (c->count * tid) / nt;
    int end = (c->count * (tid + 1)) / nt;
    const auto& W = *c->W;
    for (int i = begin; i < end; ++i) {
        const int row = c->indices[i];
        const int16_t* src = W.codes.data() + static_cast<size_t>(row) * W.K;
        float* dst = c->out + static_cast<size_t>(i) * W.K;
        const float32x4_t scale = vdupq_n_f32(W.scales[row]);
        int k = 0;
        for (; k + 15 < W.K; k += 16) {
            float32x4_t x0_lo, x0_hi, x1_lo, x1_hi;
            load_i16_as_f32_8(src + k, x0_lo, x0_hi);
            load_i16_as_f32_8(src + k + 8, x1_lo, x1_hi);
            vst1q_f32(dst + k, vmulq_f32(x0_lo, scale));
            vst1q_f32(dst + k + 4, vmulq_f32(x0_hi, scale));
            vst1q_f32(dst + k + 8, vmulq_f32(x1_lo, scale));
            vst1q_f32(dst + k + 12, vmulq_f32(x1_hi, scale));
        }
        for (; k + 7 < W.K; k += 8) {
            float32x4_t w_lo, w_hi;
            load_i16_as_f32_8(src + k, w_lo, w_hi);
            vst1q_f32(dst + k, vmulq_f32(w_lo, scale));
            vst1q_f32(dst + k + 4, vmulq_f32(w_hi, scale));
        }
        for (; k < W.K; ++k) dst[k] = static_cast<float>(src[k]) * W.scales[row];
    }
}

static inline void embedding_decode_persistent(
    const int32_t* indices, int count, const RowMajorDyadicWeight& W, float* out, SpinPool& pool
) {
    EmbeddingContext ctx{indices, count, &W, out};
    pool.run(embedding_worker, &ctx);
}


static inline void make_indirect_window_tile(
    const float* input,
    const ConvShape& s,
    int OH,
    int OW,
    int b,
    int p0,
    int valid_m,
    float* tile
) {
    const int K = s.IC * s.KH * s.KW;
    if (s.stride == 1 && s.pad == 1 && valid_m == MR) {
        int oh0 = p0 / OW;
        int ow0 = p0 - oh0 * OW;
        int oh3 = (p0 + 3) / OW;
        int ow3 = (p0 + 3) - oh3 * OW;
        if (oh0 == oh3 && oh0 >= 1 && oh0 < s.IH - 1 && ow0 >= 1 && ow3 < s.IW - 1) {
            const int ih_base = oh0 - 1;
            const int iw_base = ow0 - 1;
            for (int m = 0; m < MR; ++m) {
                float* dst = tile + static_cast<size_t>(m) * K;
                int kk = 0;
                for (int ic = 0; ic < s.IC; ++ic) {
                    const float* plane = input + ((static_cast<size_t>(b) * s.IC + ic) * s.IH + ih_base) * s.IW + iw_base;
                    float32x4_t r0 = vld1q_f32(plane);
                    float32x4_t r1 = vld1q_f32(plane + s.IW);
                    float32x4_t r2 = vld1q_f32(plane + 2 * s.IW);
                    float32x4_t r0b = vld1q_f32(plane + 4);
                    float32x4_t r1b = vld1q_f32(plane + s.IW + 4);
                    float32x4_t r2b = vld1q_f32(plane + 2 * s.IW + 4);
                    float buf[18];
                    vst1q_f32(buf, r0);
                    vst1q_f32(buf + 4, r0b);
                    vst1q_f32(buf + 8, r1);
                    vst1q_f32(buf + 12, r1b);
                    vst1q_f32(buf + 16, r2);
                    // buf[0..5] = row0 columns iw_base..iw_base+5
                    // buf[6..11] = row1
                    // buf[12..17] = row2
                    // Window m uses columns m, m+1, m+2 from each row
                    int c0 = m;
                    int c1 = m + 1;
                    int c2 = m + 2;
                    dst[kk++] = buf[c0];
                    dst[kk++] = buf[c1];
                    dst[kk++] = buf[c2];
                    dst[kk++] = buf[6 + c0];
                    dst[kk++] = buf[6 + c1];
                    dst[kk++] = buf[6 + c2];
                    dst[kk++] = buf[12 + c0];
                    dst[kk++] = buf[12 + c1];
                    dst[kk++] = buf[12 + c2];
                }
            }
            for (int m = valid_m; m < MR; ++m)
                std::memset(tile + static_cast<size_t>(m) * K, 0, static_cast<size_t>(K) * sizeof(float));
            return;
        }
    }
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
    for (int m = valid_m; m < MR; ++m) {
        std::memset(tile + static_cast<size_t>(m) * K, 0, static_cast<size_t>(K) * sizeof(float));
    }
}


void conv2d_indirect_gemm(
    const float* input,
    const PackedWeightKNR& W,
    float* output,
    const ConvShape& s,
    int threads
) {
    const int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    const int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    const int P = OH * OW;
    const int tiles_per_batch = ceil_div(P, MR);
    const int nblocks = W.Np / NR;
    const int total_tiles = s.B * tiles_per_batch;

    int nb_group = nblocks;
    if (total_tiles < threads) {
        const int groups_needed = ceil_div(threads, std::max(1, total_tiles));
        nb_group = ceil_div(nblocks, groups_needed);
        nb_group = std::max(1, nb_group);
    }
    const int ngroups = ceil_div(nblocks, nb_group);
    const int total_tasks = total_tiles * ngroups;

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        avec<float> tile(static_cast<size_t>(MR) * W.K);
        #pragma omp for schedule(static)
        for (int task = 0; task < total_tasks; ++task) {
            const int tg = task / ngroups;
            const int group = task - tg * ngroups;
            const int b = tg / tiles_per_batch;
            const int tile_idx = tg - b * tiles_per_batch;
            const int p0 = tile_idx * MR;
            const int vm = std::min(MR, P - p0);
            make_indirect_window_tile(input, s, OH, OW, b, p0, vm, tile.data());

            const int nb_begin = group * nb_group;
            const int nb_end = std::min(nblocks, nb_begin + nb_group);
            for (int nb = nb_begin; nb < nb_end; ++nb) {
                const int n0 = nb * NR;
                const int vn = std::min(NR, s.OC - n0);
                const int16_t* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * NR;

                alignas(64) float tmp[MR * NR];
                microkernel_mr_nr_f32_i16(
                    tile.data(), W.K, wb,
                    W.scales.data() + n0,
                    W.bias.data() + n0,
                    tmp, NR, W.K, vm, vn);
                for (int m = 0; m < vm; ++m) {
                    const int pos = p0 + m;
                    const int oh = pos / OW;
                    const int ow = pos - oh * OW;
                    for (int lane = 0; lane < vn; ++lane) {
                        const int oc = n0 + lane;
                        output[((static_cast<size_t>(b) * s.OC + oc) * OH + oh) * OW + ow] =
                            tmp[m * NR + lane];
                    }
                }
            }
        }
    }
}


struct ConvPersistentContext {
    const float* input;
    const PackedWeightKNR* W;
    float* output;
    ConvShape s;
    int OH, OW, P, tiles_per_batch, nblocks, nb_group, ngroups, total_tasks;
};

static void conv_persistent_worker(void* opaque, int tid, int nt) {
    auto* c = static_cast<ConvPersistentContext*>(opaque);
    const auto& s = c->s;
    const auto& W = *c->W;
    thread_local avec<float> tile;
    const size_t need = static_cast<size_t>(MR) * W.K;
    if (tile.size() < need) tile.resize(need);
    int begin = (c->total_tasks * tid) / nt;
    int end = (c->total_tasks * (tid + 1)) / nt;
    for (int task = begin; task < end; ++task) {
        const int tg = task / c->ngroups;
        const int group = task - tg * c->ngroups;
        const int b = tg / c->tiles_per_batch;
        const int tile_idx = tg - b * c->tiles_per_batch;
        const int p0 = tile_idx * MR;
        const int vm = std::min(MR, c->P - p0);
        if (s.KH == 1 && s.KW == 1) {
            for (int m = 0; m < vm; ++m) {
                const int pos = p0 + m;
                const int oh = pos / c->OW;
                const int ow = pos - oh * c->OW;
                const int ih = oh * s.stride;
                const int iw = ow * s.stride;
                for (int ic = 0; ic < s.IC; ++ic)
                    tile[static_cast<size_t>(m) * W.K + ic] = c->input[((static_cast<size_t>(b) * s.IC + ic) * s.IH + ih) * s.IW + iw];
            }
            for (int m = vm; m < MR; ++m)
                std::memset(tile.data() + static_cast<size_t>(m) * W.K, 0, static_cast<size_t>(W.K) * sizeof(float));
        } else {
            make_indirect_window_tile(c->input, s, c->OH, c->OW, b, p0, vm, tile.data());
        }
        const int nb_begin = group * c->nb_group;
        const int nb_end = std::min(c->nblocks, nb_begin + c->nb_group);
        for (int nb = nb_begin; nb < nb_end; ++nb) {
            const int n0 = nb * NR;
            const int vn = std::min(NR, s.OC - n0);
            const int16_t* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * NR;
            alignas(64) float tmp[MR * NR];
            microkernel_mr_nr_f32_i16(tile.data(), W.K, wb,
                W.scales.data() + n0, W.bias.data() + n0,
                tmp, NR, W.K, vm, vn);
            for (int m = 0; m < vm; ++m) {
                const int pos = p0 + m;
                const int oh = pos / c->OW;
                const int ow = pos - oh * c->OW;
                for (int lane = 0; lane < vn; ++lane) {
                    int oc = n0 + lane;
                    c->output[((static_cast<size_t>(b) * s.OC + oc) * c->OH + oh) * c->OW + ow] = tmp[m * NR + lane];
                }
            }
        }
    }
}

static inline void conv2d_persistent(
    const float* input, const PackedWeightKNR& W, float* output, const ConvShape& s, SpinPool& pool
) {
    const int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    const int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    const int P = OH * OW;
    const int tiles_per_batch = ceil_div(P, MR);
    const int nblocks = W.Np / NR;
    const int total_tiles = s.B * tiles_per_batch;
    int nb_group = nblocks;
    if (total_tiles < pool.size()) {
        const int groups_needed = ceil_div(pool.size(), std::max(1, total_tiles));
        nb_group = std::max(1, ceil_div(nblocks, groups_needed));
    }
    const int ngroups = ceil_div(nblocks, nb_group);
    ConvPersistentContext ctx{input, &W, output, s, OH, OW, P, tiles_per_batch, nblocks, nb_group, ngroups, total_tiles * ngroups};
    pool.run(conv_persistent_worker, &ctx);
}


void conv2d_1x1_strided(
    const float* input,
    const PackedWeightKNR& W,
    float* output,
    const ConvShape& s,
    int threads
) {
    const int OH = (s.IH - 1) / s.stride + 1;
    const int OW = (s.IW - 1) / s.stride + 1;
    const int P = OH * OW;
    const int tiles_per_batch = ceil_div(P, MR);
    const int nblocks = W.Np / NR;
    omp_set_num_threads(threads);

    #pragma omp parallel
    {
        alignas(64) float tile[MR * 4096];
        #pragma omp for schedule(static)
        for (int tg = 0; tg < s.B * tiles_per_batch; ++tg) {
            const int b = tg / tiles_per_batch;
            const int tile_idx = tg - b * tiles_per_batch;
            const int p0 = tile_idx * MR;
            const int vm = std::min(MR, P - p0);
            for (int m = 0; m < vm; ++m) {
                const int pos = p0 + m;
                const int oh = pos / OW;
                const int ow = pos - oh * OW;
                const int ih = oh * s.stride;
                const int iw = ow * s.stride;
                for (int ic = 0; ic < s.IC; ++ic) {
                    tile[m * s.IC + ic] = input[((static_cast<size_t>(b) * s.IC + ic) * s.IH + ih) * s.IW + iw];
                }
            }
            for (int m = vm; m < MR; ++m)
                std::memset(tile + m * s.IC, 0, static_cast<size_t>(s.IC) * sizeof(float));

            for (int nb = 0; nb < nblocks; ++nb) {
                const int n0 = nb * NR;
                const int vn = std::min(NR, s.OC - n0);
                const int16_t* wb = W.codes.data() + static_cast<size_t>(nb) * W.K * NR;
                alignas(64) float tmp[MR * NR];
                microkernel_mr_nr_f32_i16(tile, s.IC, wb,
                    W.scales.data() + n0, W.bias.data() + n0,
                    tmp, NR, W.K, vm, vn);
                for (int m = 0; m < vm; ++m) {
                    const int pos = p0 + m;
                    const int oh = pos / OW;
                    const int ow = pos - oh * OW;
                    for (int lane = 0; lane < vn; ++lane) {
                        int oc = n0 + lane;
                        output[((static_cast<size_t>(b) * s.OC + oc) * OH + oh) * OW + ow] = tmp[m * NR + lane];
                    }
                }
            }
        }
    }
}


struct Pool49Context {
    const float* input;
    float* output;
    int rows;
};

static inline float reduce49(const float* x) {
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
    return (vaddvq_f32(vaddq_f32(s1, s2)) + x[48]) * (1.0f / 49.0f);
}

static void pool49_worker(void* opaque, int tid, int nt) {
    auto* c = static_cast<Pool49Context*>(opaque);
    int begin = (c->rows * tid) / nt;
    int end = (c->rows * (tid + 1)) / nt;
    for (int r = begin; r < end; ++r)
        c->output[r] = reduce49(c->input + static_cast<size_t>(r) * 49);
}

void adaptive_avgpool_1x1_persistent(
    const float* input, float* output, int B, int C, SpinPool& pool
) {
    Pool49Context ctx{input, output, B * C};
    pool.run(pool49_worker, &ctx);
}

void adaptive_avgpool_1x1(
    const float* input,
    float* output,
    int B,
    int C,
    int H,
    int W,
    int threads
) {
    const int HW = H * W;
    const int rows = B * C;
    const float inv = 1.0f / static_cast<float>(HW);
    omp_set_num_threads(threads);
    if (threads <= 1) {
        for (int r = 0; r < rows; ++r) {
            const float* x = input + static_cast<size_t>(r) * HW;
            if (HW == 49) output[r] = reduce49(x);
            else {
                float32x4_t acc = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i + 7 < HW; i += 8) {
                    float32x4_t v0 = vld1q_f32(x + i);
                    float32x4_t v1 = vld1q_f32(x + i + 4);
                    acc = vaddq_f32(vaddq_f32(acc, v0), v1);
                }
                float sum = vaddvq_f32(acc);
                for (; i < HW; ++i) sum += x[i];
                output[r] = sum * inv;
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const float* x = input + static_cast<size_t>(r) * HW;
            if (HW == 49) output[r] = reduce49(x);
            else {
                float32x4_t acc = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i + 7 < HW; i += 8) {
                    float32x4_t v0 = vld1q_f32(x + i);
                    float32x4_t v1 = vld1q_f32(x + i + 4);
                    acc = vaddq_f32(vaddq_f32(acc, v0), v1);
                }
                float sum = vaddvq_f32(acc);
                for (; i < HW; ++i) sum += x[i];
                output[r] = sum * inv;
            }
        }
    }
}


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

static bool check_linear_samples(
    const avec<float>& A,
    const RowMajorDyadicWeight& Wr,
    const avec<float>& bias,
    const avec<float>& C,
    int M,
    int N,
    int K
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


struct Result {
    std::string name;
    double gate_ms;
    double best_ms;
    int threads;
    bool correct;
    std::string tree;
};

static Result bench_linear_case(
    const std::string& name,
    int M, int K, int N,
    double gate_ms,
    const std::vector<int>& thread_choices
) {
    avec<float> A(static_cast<size_t>(M)*K), bias(N), C(static_cast<size_t>(M)*N);
    RowMajorDyadicWeight Wr;
    Wr.N=N; Wr.K=K; Wr.codes.resize(static_cast<size_t>(N)*K); Wr.scales.resize(N);
    fill_random_float(A, 10 + M + N, 0.25f);
    fill_random_float(bias, 20 + N, 0.1f);
    fill_random_i16(Wr.codes, 30 + N);
    for (int n=0;n<N;++n) Wr.scales[n]=std::ldexp(1.0f, -5 - (n%3));
    PackedWeightKNR W = pack_weight_k_nr(Wr.codes.data(), Wr.scales.data(), bias.data(), N, K);

    double best = std::numeric_limits<double>::infinity();
    int best_t=1;
    for (int t : thread_choices) {
        int reps = (N > 100000) ? 1 : 10;
        double ms = median_ms([&]{ linear_gemm(A.data(), W, C.data(), M, t); g_sink += C[(M/2)*N + (N/2)]*1e-30f; }, 2, reps, 3);
        std::cerr << "  " << name << " t=" << t << " ms=" << ms << "\n";
        if (ms < best) { best=ms; best_t=t; }
    }
    linear_gemm(A.data(), W, C.data(), M, best_t);
    bool ok = check_linear_samples(A, Wr, bias, C, M, N, K);
    return {name, gate_ms, best, best_t, ok,
            "pack_weight_k_nr -> microkernel_4x8 -> apply_scale_bias -> contiguous_writeback"};
}

static Result bench_embedding_case(double gate_ms, const std::vector<int>& thread_choices) {
    const int vocab=151936, K=896, count=256;
    RowMajorDyadicWeight W; W.N=vocab; W.K=K;
    W.codes.resize(static_cast<size_t>(vocab)*K); W.scales.resize(vocab);
    fill_random_i16(W.codes, 88);
    for(int n=0;n<vocab;++n) W.scales[n]=std::ldexp(1.0f,-5-(n%3));
    std::vector<int32_t> idx(count);
    for(int i=0;i<count;++i) idx[i]=(i*593+17)%vocab;
    avec<float> out(static_cast<size_t>(count)*K);
    double best=1e99; int best_t=1;
    for(int t: thread_choices){
        SpinPool pool(t);
        double ms=median_ms([&]{ embedding_decode_persistent(idx.data(),count,W,out.data(),pool); g_sink += out[(count/2)*K]*1e-30f; },10,200,5);
        std::cerr << "  embedding persistent t=" << t << " ms=" << ms << "\n";
        if(ms<best){best=ms;best_t=t;}
    }
    SpinPool best_pool(best_t);
    embedding_decode_persistent(idx.data(),count,W,out.data(),best_pool);
    bool ok=true;
    for(int i=0;i<16 && ok;++i){
        int r=idx[i];
        for(int k=0;k<K;k+=97){
            float ref=static_cast<float>(W.codes[static_cast<size_t>(r)*K+k])*W.scales[r];
            if(out[static_cast<size_t>(i)*K+k]!=ref){ok=false;break;}
        }
    }
    return {"embedding_qwen_vocab_width",gate_ms,best,best_t,ok,
            "persistent_spin_workers -> gather_row -> neon_i16_to_f32 -> apply_row_scale -> streaming_writeback"};
}

static bool check_conv_samples(
    const avec<float>& input,
    const RowMajorDyadicWeight& Wr,
    const avec<float>& bias,
    const avec<float>& out,
    const ConvShape& s
){
    int OH=(s.IH+2*s.pad-s.KH)/s.stride+1;
    int OW=(s.IW+2*s.pad-s.KW)/s.stride+1;
    std::mt19937 rng(778+s.OC);
    for(int t=0;t<12;++t){
        int b=rng()%s.B, oc=rng()%s.OC, oh=rng()%OH, ow=rng()%OW;
        double acc=0.0; int kk=0;
        for(int ic=0;ic<s.IC;++ic) for(int kh=0;kh<s.KH;++kh) for(int kw=0;kw<s.KW;++kw,++kk){
            int ih=oh*s.stride+kh-s.pad, iw=ow*s.stride+kw-s.pad;
            if((unsigned)ih<(unsigned)s.IH && (unsigned)iw<(unsigned)s.IW){
                float x=input[((static_cast<size_t>(b)*s.IC+ic)*s.IH+ih)*s.IW+iw];
                int16_t q=Wr.codes[static_cast<size_t>(oc)*Wr.K+kk];
                acc += static_cast<double>(x)*q;
            }
        }
        float ref=static_cast<float>(acc*Wr.scales[oc]+bias[oc]);
        float got=out[((static_cast<size_t>(b)*s.OC+oc)*OH+oh)*OW+ow];
        float tol=3e-3f*std::max(1.0f,std::abs(ref));
        if(std::abs(ref-got)>tol){
            std::cerr<<"conv mismatch "<<s.name<<" ref="<<ref<<" got="<<got<<"\n";
            return false;
        }
    }
    return true;
}

static Result bench_conv_case(const ConvShape& s,double gate_ms,const std::vector<int>& thread_choices){
    int K=s.IC*s.KH*s.KW;
    int OH=(s.IH+2*s.pad-s.KH)/s.stride+1;
    int OW=(s.IW+2*s.pad-s.KW)/s.stride+1;
    avec<float> input(static_cast<size_t>(s.B)*s.IC*s.IH*s.IW), bias(s.OC), out(static_cast<size_t>(s.B)*s.OC*OH*OW);
    RowMajorDyadicWeight Wr; Wr.N=s.OC; Wr.K=K; Wr.codes.resize(static_cast<size_t>(s.OC)*K); Wr.scales.resize(s.OC);
    fill_random_float(input,400+s.OC,0.25f); fill_random_float(bias,500+s.OC,0.1f); fill_random_i16(Wr.codes,600+s.OC);
    for(int n=0;n<s.OC;++n) Wr.scales[n]=std::ldexp(1.0f,-5-(n%3));
    PackedWeightKNR W=pack_weight_k_nr(Wr.codes.data(),Wr.scales.data(),bias.data(),s.OC,K);
    double best=1e99; int best_t=1;
    for(int t:thread_choices){
        SpinPool pool(t);
        int reps = (s.B==8 && s.KH==3) ? 2 : 10;
        auto fn=[&]{
            conv2d_persistent(input.data(),W,out.data(),s,pool);
            g_sink += out[out.size()/2]*1e-30f;
        };
        double ms=median_ms(fn,3,reps,5);
        std::cerr << "  " << s.name << " persistent t=" << t << " ms=" << ms << "\n";
        if(ms<best){best=ms;best_t=t;}
    }
    SpinPool best_pool(best_t);
    conv2d_persistent(input.data(),W,out.data(),s,best_pool);
    bool ok=check_conv_samples(input,Wr,bias,out,s);
    std::string tree=(s.KH==1)?
      "gather_stride1x1_activation_tile -> microkernel_4x8 -> scale_bias -> nchw_strided_writeback":
      "make_indirect_window_tile_4xK -> microkernel_4x8 -> scale_bias -> nchw_strided_writeback";
    return {s.name,gate_ms,best,best_t,ok,tree};
}

static Result bench_pool_case(double gate_ms,const std::vector<int>& thread_choices){
    int B=8,C=512,H=7,W=7;
    avec<float> input(static_cast<size_t>(B)*C*H*W), out(static_cast<size_t>(B)*C);
    fill_random_float(input,111,1.0f);
    double best=1e99; int best_t=1;
    for(int t:thread_choices){
        SpinPool pool(t);
        double ms=median_ms([&]{ adaptive_avgpool_1x1_persistent(input.data(),out.data(),B,C,pool); g_sink += out[out.size()/2]*1e-30f; },10,300,5);
        std::cerr << "  pool persistent t=" << t << " ms=" << ms << "\n";
        if(ms<best){best=ms;best_t=t;}
    }
    SpinPool best_pool(best_t);
    adaptive_avgpool_1x1_persistent(input.data(),out.data(),B,C,best_pool);
    bool ok=true;
    for(int r=0;r<B*C;r+=257){
        double sum=0; for(int i=0;i<H*W;++i) sum+=input[static_cast<size_t>(r)*H*W+i];
        float ref=static_cast<float>(sum/(H*W));
        if(std::abs(ref-out[r])>1e-5f){ok=false;break;}
    }
    return {"adaptive_avgpool2d_resnet_global",gate_ms,best,best_t,ok,
            "persistent_spin_workers -> shape_specialized_reduction_49 -> contiguous_writeback"};
}

static void write_csv(const std::string& path,const std::vector<Result>& rs){
    std::ofstream f(path);
    f<<"subkernel,materialized_gate_ms,arm64_neon_ms,speedup_vs_arm64_gate,best_threads,passes_fixed_gate,correct,op_tree\n";
    for(auto& r:rs){
        f << r.name << ',' << std::fixed << std::setprecision(6) << r.gate_ms << ',' << r.best_ms << ',' << (r.gate_ms / r.best_ms) << ',' << r.threads << ',' << (r.best_ms < r.gate_ms ? "true" : "false") << ',' << (r.correct ? "true" : "false") << ",\"" << r.tree << "\"\n";
    }
}

int main(int argc,char** argv){
    std::string out="arm64_neon_gate_results.csv";
    if(argc>1) out=argv[1];
    omp_set_dynamic(0);
    std::vector<int> threads={1,2,4,6,8,10};
    int max_t=omp_get_max_threads();
    threads.erase(std::remove_if(threads.begin(),threads.end(),[&](int t){return t>max_t;}),threads.end());
    if(threads.empty()) threads={1};

    std::vector<Result> rs;
    const std::string only = std::getenv("DYOP_ONLY") ? std::getenv("DYOP_ONLY") : "all";
    auto want = [&](const std::string& key){ return only == "all" || only == key; };
    if (want("gemm")) { std::cerr << "bench linear_gemm_qwen_seq\n"; rs.push_back(bench_linear_case("linear_gemm_qwen_seq",64,896,896,0.192396,threads)); }
    if (want("outproj")) { std::cerr << "bench linear_output_projection\n"; rs.push_back(bench_linear_case("linear_output_projection",8,896,151936,10.843443,threads)); }
    if (want("embedding")) { std::cerr << "bench embedding\n"; rs.push_back(bench_embedding_case(0.015501,threads)); }

    std::vector<std::pair<ConvShape,double>> convs={
      {{"resnet_conv3x3",8,64,56,56,64,3,3,1,1},3.935540},
      {{"resnet_layer2_stride2_3x3",1,64,56,56,128,3,3,2,1},0.347237},
      {{"resnet_layer3_stride2_3x3",1,128,28,28,256,3,3,2,1},0.262369},
      {{"resnet_layer4_stride2_3x3",1,256,14,14,512,3,3,2,1},0.228865},
      {{"resnet_downsample",8,128,28,28,256,1,1,2,0},0.265602}
    };
    if (only == "all" || only == "conv") for(auto& [cs,g]:convs) { std::cerr << "bench " << cs.name << "\n"; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (only == "conv0") { auto& [cs,g]=convs[0]; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (only == "conv1") { auto& [cs,g]=convs[1]; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (only == "conv2") { auto& [cs,g]=convs[2]; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (only == "conv3") { auto& [cs,g]=convs[3]; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (only == "conv4") { auto& [cs,g]=convs[4]; rs.push_back(bench_conv_case(cs,g,threads)); }
    if (want("pool")) { std::cerr << "bench adaptive pool\n"; rs.push_back(bench_pool_case(0.013332,threads)); }

    write_csv(out,rs);
    std::cout<<"CPU primitive profile: arm64_neon, MR="<<MR<<" NR="<<NR<<"\n";
    for(auto& r:rs){
      std::cout<<std::left<<std::setw(38)<<r.name
               <<" gate="<<std::right<<std::setw(9)<<std::fixed<<std::setprecision(4)<<r.gate_ms
               <<" ms  native="<<std::setw(9)<<r.best_ms
               <<" ms  ratio="<<std::setw(7)<<(r.gate_ms/r.best_ms)<<"x"
               <<"  t="<<std::setw(2)<<r.threads
               <<"  "<<(r.best_ms<r.gate_ms?"PASS":"FAIL")
               <<"  correct="<<(r.correct?"yes":"NO")<<"\n";
    }
    std::cout<<"Wrote "<<out<<" sink="<<g_sink<<"\n";
    return std::all_of(rs.begin(),rs.end(),[](const Result&r){return r.correct;})?0:2;
}

} // namespace dyop

int main(int argc,char**argv){return dyop::main(argc,argv);}
