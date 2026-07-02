// AMX port of the dyadic quant GEMM primitives.
// Tile geometry: MR=16, NR=16 (native AMX outer product).
// Activations are pre-decoded fp32; weights are int16 packed K×NR.
// Compile: clang++ -march=armv8-a+fp+simd -O2 -std=c++17 -o dyop_amx dyop_primitives_amx.cpp

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

#include "amx_intrinsics.h"

namespace dyop {

constexpr int NR = 16;
constexpr int MR = 16;
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
    avec<int16_t> codes;
    avec<float> scales;
    avec<float> bias;
};

struct RowMajorDyadicWeight {
    int N = 0;
    int K = 0;
    avec<int16_t> codes;
    avec<float> scales;
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
                const int16_t code = (n < N) ? rowmajor_codes[static_cast<size_t>(n) * K + k] : 0;
                dst[lane] = code;
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


static inline void load_i16_as_f32_16(const int16_t* p, float* dst) {
    int16x8_t v0 = vld1q_s16(p);
    int16x8_t v1 = vld1q_s16(p + 8);
    int32x4_t w00 = vmovl_s16(vget_low_s16(v0));
    int32x4_t w01 = vmovl_s16(vget_high_s16(v0));
    int32x4_t w10 = vmovl_s16(vget_low_s16(v1));
    int32x4_t w11 = vmovl_s16(vget_high_s16(v1));
    vst1q_f32(dst + 0, vcvtq_f32_s32(w00));
    vst1q_f32(dst + 4, vcvtq_f32_s32(w01));
    vst1q_f32(dst + 8, vcvtq_f32_s32(w10));
    vst1q_f32(dst + 12, vcvtq_f32_s32(w11));
}


void apply_scale_bias(float* tile, int rows, int cols, int stride,
                      const float* scale, const float* bias) {
    int c = 0;
    for (; c + 15 < cols; c += 16) {
        float32x4_t s0 = vld1q_f32(scale + c);
        float32x4_t s1 = vld1q_f32(scale + c + 4);
        float32x4_t s2 = vld1q_f32(scale + c + 8);
        float32x4_t s3 = vld1q_f32(scale + c + 12);
        float32x4_t b0 = vld1q_f32(bias + c);
        float32x4_t b1 = vld1q_f32(bias + c + 4);
        float32x4_t b2 = vld1q_f32(bias + c + 8);
        float32x4_t b3 = vld1q_f32(bias + c + 12);
        for (int r = 0; r < rows; ++r) {
            float* dst = tile + static_cast<size_t>(r) * stride + c;
            float32x4_t x0 = vld1q_f32(dst);
            float32x4_t x1 = vld1q_f32(dst + 4);
            float32x4_t x2 = vld1q_f32(dst + 8);
            float32x4_t x3 = vld1q_f32(dst + 12);
            vst1q_f32(dst,     vfmaq_f32(b0, x0, s0));
            vst1q_f32(dst + 4, vfmaq_f32(b1, x1, s1));
            vst1q_f32(dst + 8, vfmaq_f32(b2, x2, s2));
            vst1q_f32(dst + 12,vfmaq_f32(b3, x3, s3));
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


// AMX MR×NR microkernel with fp32 weights (pre-decoded).
// A_tile_buf is [K][16] pre-transposed from [16][K] row-major.
// W_float is [K][16] fp32.
// Saves the int16→fp32 decode inside the K loop.
static inline void microkernel_mr16_nr16_f32_f32(
    const float* A_tile_buf,
    const float* W_float,
    const float* scale,
    const float* bias,
    float* C,
    int row_stride,
    int col_stride,
    int K,
    int z_row,
    int valid_m,
    int valid_n
) {
    AMX_FMA32(amx_enc_zero_z(z_row));

    for (int k = 0; k < K; k++) {
        AMX_LDX((uint64_t)(A_tile_buf + k * 16));
        AMX_LDY((uint64_t)(W_float + k * 16));
        AMX_FMA32(amx_enc_fma32(0, 0, z_row));
    }

    alignas(64) float z_buf[16][16];
    for (int j = 0; j < 16; j++) {
        int z_idx = j * 4 + z_row;
        AMX_STZ(amx_enc_stz(&z_buf[j][0], z_idx));
    }
    for (int i = 0; i < valid_m; i++) {
        float* dst = C + static_cast<size_t>(i) * row_stride;
        for (int j = 0; j < valid_n; j++)
            dst[static_cast<size_t>(j) * col_stride] = z_buf[j][i] * scale[j] + bias[j];
    }
}


// Converts an int16 packed panel and calls the fp32 AMX microkernel.
static inline void microkernel_mr16_nr16_f32_i16(
    const float* A,
    int lda,
    const int16_t* W_block,
    const float* scale,
    const float* bias,
    float* C,
    int row_stride,
    int col_stride,
    int K,
    int z_row,
    int valid_m,
    int valid_n
) {
    thread_local avec<float> A_tile_storage;
    thread_local avec<float> W_f32_storage;
    A_tile_storage.resize(static_cast<size_t>(K) * 16);
    W_f32_storage.resize(static_cast<size_t>(K) * 16);
    float* A_tile = A_tile_storage.data();
    float* W_f32 = W_f32_storage.data();

    for (int k = 0; k < K; k++) {
        for (int i = 0; i < valid_m; i++)
            A_tile[k * 16 + i] = A[static_cast<size_t>(i) * lda + k];
        for (int i = valid_m; i < 16; i++)
            A_tile[k * 16 + i] = 0.0f;
    }

    for (int k = 0; k < K; k++)
        load_i16_as_f32_16(W_block + static_cast<size_t>(k) * NR, W_f32 + k * NR);

    microkernel_mr16_nr16_f32_f32(A_tile, W_f32, scale, bias, C,
                                  row_stride, col_stride, K, z_row,
                                  valid_m, valid_n);
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
    const int K = W.K;
    const int Np = W.Np;
    const int Mp = mblocks * MR;

    // Pre-transpose A: [M][K] -> [K][Mp], padding partial AMX tiles.
    const size_t A_T_sz = static_cast<size_t>(K) * Mp;
    std::vector<float, AlignedAllocator<float>> A_T(A_T_sz, 0.0f);

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int k = 0; k < K; k++)
            for (int i = 0; i < M; i++)
                A_T[static_cast<size_t>(k) * Mp + i] = A[static_cast<size_t>(i) * K + k];

        AMX_SET();
        thread_local avec<float> W_f32_storage;
        W_f32_storage.resize(static_cast<size_t>(K) * NR);
        float* W_f32 = W_f32_storage.data();
        alignas(64) float z_buf[16][16];
        #pragma omp for schedule(static)
        for (int nb = 0; nb < nblocks; ++nb) {
            const int n0 = nb * NR;
            const int vn = std::min(NR, W.N - n0);
            const int16_t* const W_i16 = W.codes.data() + static_cast<size_t>(nb) * K * NR;
            const float* const scale = W.scales.data() + n0;
            const float* const bias = W.bias.data() + n0;

            for (int k = 0; k < K; k++)
                load_i16_as_f32_16(W_i16 + static_cast<size_t>(k) * NR, W_f32 + static_cast<size_t>(k) * NR);

            for (int mb = 0; mb < mblocks; ++mb) {
                const int m0 = mb * MR;
                const int vm = std::min(MR, M - m0);
                const int z_row = mb % 4;
                const float* const A_ptr = A_T.data() + m0;
                float* const C_tile = C + static_cast<size_t>(m0) * W.N + n0;

                AMX_FMA32(amx_enc_zero_z(z_row));

                for (int k = 0; k < K; k++) {
                    AMX_LDX((uint64_t)(A_ptr + static_cast<size_t>(k) * Mp));
                    AMX_LDY((uint64_t)(W_f32 + static_cast<size_t>(k) * NR));
                    AMX_FMA32(amx_enc_fma32(0, 0, z_row));
                }

                for (int j = 0; j < 16; j++)
                    AMX_STZ(amx_enc_stz(&z_buf[j][0], j * 4 + z_row));

                for (int i = 0; i < vm; i++) {
                    for (int j = 0; j < vn; j++)
                        C_tile[static_cast<size_t>(i) * W.N + j] =
                            z_buf[j][i] * scale[j] + bias[j];
                }
            }
        }
        AMX_CLR();
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
            float buf[16];
            load_i16_as_f32_16(src + k, buf);
            float32x4_t x0 = vld1q_f32(buf + 0);
            float32x4_t x1 = vld1q_f32(buf + 4);
            float32x4_t x2 = vld1q_f32(buf + 8);
            float32x4_t x3 = vld1q_f32(buf + 12);
            vst1q_f32(dst + k,      vmulq_f32(x0, scale));
            vst1q_f32(dst + k + 4,  vmulq_f32(x1, scale));
            vst1q_f32(dst + k + 8,  vmulq_f32(x2, scale));
            vst1q_f32(dst + k + 12, vmulq_f32(x3, scale));
        }
        for (; k + 7 < W.K; k += 8) {
            float32x4_t w_lo, w_hi;
            int16x8_t v = vld1q_s16(src + k);
            int32x4_t lo = vmovl_s16(vget_low_s16(v));
            int32x4_t hi = vmovl_s16(vget_high_s16(v));
            w_lo = vcvtq_f32_s32(lo);
            w_hi = vcvtq_f32_s32(hi);
            vst1q_f32(dst + k,     vmulq_f32(w_lo, scale));
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
            float buf[16];
            load_i16_as_f32_16(src + k, buf);
            float32x4_t x0 = vld1q_f32(buf + 0);
            float32x4_t x1 = vld1q_f32(buf + 4);
            float32x4_t x2 = vld1q_f32(buf + 8);
            float32x4_t x3 = vld1q_f32(buf + 12);
            vst1q_f32(dst + k,      vmulq_f32(x0, scale));
            vst1q_f32(dst + k + 4,  vmulq_f32(x1, scale));
            vst1q_f32(dst + k + 8,  vmulq_f32(x2, scale));
            vst1q_f32(dst + k + 12, vmulq_f32(x3, scale));
        }
        for (; k + 7 < W.K; k += 8) {
            float32x4_t w_lo, w_hi;
            int16x8_t v = vld1q_s16(src + k);
            int32x4_t lo = vmovl_s16(vget_low_s16(v));
            int32x4_t hi = vmovl_s16(vget_high_s16(v));
            w_lo = vcvtq_f32_s32(lo);
            w_hi = vcvtq_f32_s32(hi);
            vst1q_f32(dst + k,     vmulq_f32(w_lo, scale));
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

    for (int m = 0; m < valid_m; ++m) {
        int p = p0 + m;
        int oh = p / OW;
        int ow = p - oh * OW;
        int ih = oh * s.stride - s.pad;
        int iw = ow * s.stride - s.pad;
        float* dst = tile + static_cast<size_t>(m) * K;
        int kk = 0;
        for (int ic = 0; ic < s.IC; ++ic)
            for (int kh = 0; kh < s.KH; ++kh)
                for (int kw = 0; kw < s.KW; ++kw) {
                    int h = ih + kh, w = iw + kw;
                    dst[kk++] = (h >= 0 && h < s.IH && w >= 0 && w < s.IW)
                        ? input[((static_cast<size_t>(b) * s.IC + ic) * s.IH + h) * s.IW + w] : 0.0f;
                }
    }
    for (int m = valid_m; m < MR; ++m)
        std::memset(tile + static_cast<size_t>(m) * K, 0, static_cast<size_t>(K) * sizeof(float));
}


void conv2d_persistent(
    const float* input,
    const PackedWeightKNR& W,
    float* output,
    const ConvShape& s,
    SpinPool& pool
) {
    const int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    const int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    const int K = s.IC * s.KH * s.KW;
    const int nblocks = W.Np / NR;
    const int oplane = OH * OW;

    struct Ctx {
        const float* input;
        const PackedWeightKNR* W;
        float* output;
        const ConvShape* s;
        int OH, OW, K, nblocks, oplane;
    };
    Ctx ctx{input, &W, output, &s, OH, OW, K, nblocks, oplane};
    pool.run([](void* opaque, int tid, int nt) {
        auto& c = *static_cast<Ctx*>(opaque);
        int total_batches = c.s->B * c.oplane;
        int seg = (total_batches + nt - 1) / nt;
        int p_begin = tid * seg;
        int p_end = std::min(total_batches, (tid + 1) * seg);

        const int Kmax = c.K;
        thread_local avec<float> tile_storage;
        tile_storage.resize(static_cast<size_t>(MR) * Kmax);
        float* tile = tile_storage.data();

        AMX_SET();

        for (int p = p_begin; p < p_end; ) {
            int b = p / c.oplane;
            int p0 = p - b * c.oplane;
            int vm = std::min({MR, p_end - p, c.oplane - p0});

            make_indirect_window_tile(c.input, *c.s, c.OH, c.OW, b, p0, vm, tile);

            for (int nb = 0; nb < c.nblocks; ++nb) {
                int n0 = nb * NR;
                int vn = std::min(NR, c.W->N - n0);
                const int16_t* wb = c.W->codes.data() + static_cast<size_t>(nb) * Kmax * NR;

                microkernel_mr16_nr16_f32_i16(
                    tile, Kmax, wb,
                    c.W->scales.data() + n0, c.W->bias.data() + n0,
                    c.output + ((static_cast<size_t>(b) * c.W->N + n0) * c.oplane) + p0,
                    1,
                    c.oplane,
                    Kmax, 0, vm, vn
                );
            }
            p += vm;
        }

        AMX_CLR();
    }, &ctx);
}


static inline void conv2d_1x1_strided_persistent(
    const float* input,
    const PackedWeightKNR& W,
    float* output,
    const ConvShape& s,
    SpinPool& pool
) {
    const int OH = (s.IH + 2 * s.pad - s.KH) / s.stride + 1;
    const int OW = (s.IW + 2 * s.pad - s.KW) / s.stride + 1;
    const int K = s.IC;
    const int nblocks = W.Np / NR;
    const int oplane = OH * OW;

    struct Ctx {
        const float* input;
        const PackedWeightKNR* W;
        float* output;
        const ConvShape* s;
        int OH, OW, K, nblocks, oplane;
    };
    Ctx ctx{input, &W, output, &s, OH, OW, K, nblocks, oplane};
    pool.run([](void* opaque, int tid, int nt) {
        auto& c = *static_cast<Ctx*>(opaque);
        int total_tiles = c.s->B * c.oplane;
        int seg = (total_tiles + nt - 1) / nt;
        int p_begin = tid * seg;
        int p_end = std::min(total_tiles, (tid + 1) * seg);

        const int Kmax = c.K;
        thread_local avec<float> tile_storage;
        tile_storage.resize(static_cast<size_t>(MR) * Kmax);
        float* tile = tile_storage.data();

        AMX_SET();

        for (int p = p_begin; p < p_end; ) {
            int b = p / c.oplane;
            int p0 = p - b * c.oplane;
            int vm = std::min({MR, p_end - p, c.oplane - p0});
            int stride = c.s->stride;

            for (int m = 0; m < vm; ++m) {
                float* dst = tile + static_cast<size_t>(m) * Kmax;
                int p_out = p0 + m;
                int oh_m = p_out / c.OW;
                int ow_m = p_out - oh_m * c.OW;
                int ih = oh_m * stride - c.s->pad;
                int iw = ow_m * stride - c.s->pad;
                const float* src = c.input + ((static_cast<size_t>(b) * c.s->IC + 0) * c.s->IH + ih) * c.s->IW + iw;
                for (int ic = 0; ic < c.s->IC; ++ic)
                    dst[ic] = (ih >= 0 && ih < c.s->IH && iw >= 0 && iw < c.s->IW)
                        ? src[static_cast<size_t>(ic) * c.s->IH * c.s->IW] : 0.0f;
            }
            for (int m = vm; m < MR; ++m)
                std::memset(tile + static_cast<size_t>(m) * Kmax, 0, static_cast<size_t>(Kmax) * sizeof(float));

            for (int nb = 0; nb < c.nblocks; ++nb) {
                int n0 = nb * NR;
                int vn = std::min(NR, c.W->N - n0);
                const int16_t* wb = c.W->codes.data() + static_cast<size_t>(nb) * Kmax * NR;

                microkernel_mr16_nr16_f32_i16(
                    tile, Kmax, wb,
                    c.W->scales.data() + n0, c.W->bias.data() + n0,
                    c.output + ((static_cast<size_t>(b) * c.W->N + n0) * c.oplane) + p0,
                    1,
                    c.oplane,
                    Kmax, 0, vm, vn
                );
            }
            p += vm;
        }

        AMX_CLR();
    }, &ctx);
}


static inline float reduce49(const float* x) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < 49; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(x + i));
    }
    float sum = vaddvq_f32(acc);
    for (; i < 49; ++i) sum += x[i];
    return sum;
}

void adaptive_avgpool_1x1_persistent(
    const float* input,
    float* output,
    int B,
    int C,
    SpinPool& pool
) {
    int HW = 49;
    int rows = B * C;
    float inv = 1.0f / HW;
    struct Ctx { const float* input; float* output; int rows; int HW; float inv; };
    Ctx ctx{input, output, rows, HW, inv};
    pool.run([](void* opaque, int tid, int nt) {
        auto& c = *static_cast<Ctx*>(opaque);
        int seg = (c.rows + nt - 1) / nt;
        int begin = tid * seg;
        int end = std::min(c.rows, (tid + 1) * seg);
        if (c.HW == 49) {
            for (int r = begin; r < end; ++r)
                c.output[r] = reduce49(c.input + static_cast<size_t>(r) * 49) * c.inv;
        } else {
            for (int r = begin; r < end; ++r) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                const float* x = c.input + static_cast<size_t>(r) * c.HW;
                int i = 0;
                for (; i + 7 < c.HW; i += 8) {
                    float32x4_t v0 = vld1q_f32(x + i);
                    float32x4_t v1 = vld1q_f32(x + i + 4);
                    acc = vaddq_f32(vaddq_f32(acc, v0), v1);
                }
                float sum = vaddvq_f32(acc);
                for (; i < c.HW; ++i) sum += x[i];
                c.output[r] = sum * c.inv;
            }
        }
    }, &ctx);
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
            "pack_weight_k_nr -> microkernel_16x16_amx -> apply_scale_bias -> contiguous_writeback"};
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
      "gather_stride1x1_activation_tile -> microkernel_16x16_amx -> scale_bias -> nchw_strided_writeback":
      "make_indirect_window_tile_16xK -> microkernel_16x16_amx -> scale_bias -> nchw_strided_writeback";
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
    f<<"subkernel,materialized_gate_ms,arm64_amx_ms,speedup_vs_arm64_gate,best_threads,passes_fixed_gate,correct,op_tree\n";
    for(auto& r:rs){
        f << r.name << ',' << std::fixed << std::setprecision(6) << r.gate_ms << ',' << r.best_ms << ',' << (r.gate_ms / r.best_ms) << ',' << r.threads << ',' << (r.best_ms < r.gate_ms ? "true" : "false") << ',' << (r.correct ? "true" : "false") << ",\"" << r.tree << "\"\n";
    }
}

int main(int argc,char** argv){
    std::string out="arm64_amx_gate_results.csv";
    if(argc>1) out=argv[1];
    omp_set_dynamic(0);
    std::vector<int> threads={1,2,3,4,5,6,8,10};
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
    std::cout<<"CPU primitive profile: arm64_amx, MR="<<MR<<" NR="<<NR<<"\n";
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
