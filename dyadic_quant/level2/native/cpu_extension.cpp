#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include "../amx_intrinsics.h"
#endif

namespace py = pybind11;

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <class U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, n * sizeof(T)) != 0) throw std::bad_alloc();
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { free(p); }
};

template <class T, class U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }

class NativeWorkerPool {
public:
    void warm(int threads) {
        ensure_workers(std::max(1, threads));
    }

    template <typename Fn>
    void parallel(int64_t begin, int64_t end, int threads, Fn fn) {
        const int64_t total = end - begin;
        if (threads <= 1 || total <= 1) {
            fn(begin, end);
            return;
        }
        threads = std::min<int>(threads, int(total));
        ensure_workers(threads);

        {
            std::unique_lock<std::mutex> lock(mutex_);
            begin_ = begin;
            end_ = end;
            active_threads_ = threads;
            remaining_workers_ = threads - 1;
            task_ = [&](int64_t task_begin, int64_t task_end) {
                fn(task_begin, task_end);
            };
            ++generation_;
        }
        start_cv_.notify_all();

        run_chunk(0, threads, begin, end, fn);

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return remaining_workers_ == 0; });
        task_ = nullptr;
    }

    ~NativeWorkerPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

private:
    template <typename Fn>
    static void run_chunk(int thread_index, int threads, int64_t begin, int64_t end, Fn& fn) {
        const int64_t total = end - begin;
        const int64_t chunk_begin = begin + (total * thread_index) / threads;
        const int64_t chunk_end = begin + (total * (thread_index + 1)) / threads;
        fn(chunk_begin, chunk_end);
    }

    void ensure_workers(int threads) {
        const int needed_workers = std::max(0, threads - 1);
        while (int(workers_.size()) < needed_workers) {
            const int worker_index = int(workers_.size()) + 1;
            workers_.emplace_back([this, worker_index] { worker_loop(worker_index); });
        }
    }

    void worker_loop(int worker_index) {
        size_t seen_generation = 0;
        while (true) {
            std::function<void(int64_t, int64_t)> task;
            int64_t begin = 0;
            int64_t end = 0;
            int active_threads = 1;
            bool participates = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) return;
                seen_generation = generation_;
                task = task_;
                begin = begin_;
                end = end_;
                active_threads = active_threads_;
                participates = worker_index < active_threads;
            }

            if (participates && task) {
                const int64_t total = end - begin;
                const int64_t chunk_begin = begin + (total * worker_index) / active_threads;
                const int64_t chunk_end = begin + (total * (worker_index + 1)) / active_threads;
                task(chunk_begin, chunk_end);
            }

            if (participates) {
                std::unique_lock<std::mutex> lock(mutex_);
                --remaining_workers_;
                if (remaining_workers_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::function<void(int64_t, int64_t)> task_;
    int64_t begin_ = 0;
    int64_t end_ = 0;
    int active_threads_ = 1;
    int remaining_workers_ = 0;
    size_t generation_ = 0;
    bool stopping_ = false;
};

static NativeWorkerPool& native_worker_pool() {
    static NativeWorkerPool pool;
    return pool;
}

static int native_thread_count() {
    if (const char* value = std::getenv("DYOP_CPU_THREADS")) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && parsed > 0) {
            return int(std::min<long>(parsed, 256));
        }
    }
    return std::max(1, at::get_num_threads());
}

static int native_amx_thread_count() {
    if (const char* value = std::getenv("DYOP_AMX_THREADS")) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && parsed > 0) {
            return int(std::min<long>(parsed, 256));
        }
    }
    return std::min(native_thread_count(), 7);
}

template <typename Fn>
static void parallel_for_threads(int64_t begin, int64_t end, int threads, Fn fn) {
    native_worker_pool().parallel(begin, end, threads, fn);
}

struct PackedDyopWeight {
    at::Tensor codes;   // int16 [out, K], signed odd prefix code
    at::Tensor codes_knr;  // int16 [ceil(out/8), K, 8]
    at::Tensor codes_knr16;  // int16 [ceil(out/16), K, 16]
    at::Tensor codes_i8_knr;  // int8 [ceil(out/8), K, 8] when the prefix fits
    at::Tensor codes_f32_knr;  // float32 [ceil(out/8), K, 8]
    at::Tensor codes_f32_knr16;  // float32 [ceil(out/16), K, 16]
    at::Tensor scales;  // float32 [out, blocks], dyadic step / 2 per group
    at::Tensor scales_padded;  // float32 [ceil(out/8)*8, blocks]
    at::Tensor scales_padded16;  // float32 [ceil(out/16)*16, blocks]
    std::vector<int64_t> shape;
    int64_t out = 0;
    int64_t out_padded = 0;
    int64_t k = 0;
    int64_t group_size = 0;
    int64_t bits = 0;
};

static PackedDyopWeight unpack_weight(const py::dict& packed) {
    PackedDyopWeight w;
    w.codes = packed["codes"].cast<at::Tensor>().contiguous();
    if (packed.contains("codes_knr")) {
        w.codes_knr = packed["codes_knr"].cast<at::Tensor>().contiguous();
    }
    if (packed.contains("codes_knr16")) {
        w.codes_knr16 = packed["codes_knr16"].cast<at::Tensor>().contiguous();
    }
    if (packed.contains("codes_i8_knr")) {
        w.codes_i8_knr = packed["codes_i8_knr"].cast<at::Tensor>().contiguous();
    }
    if (packed.contains("codes_f32_knr")) {
        w.codes_f32_knr = packed["codes_f32_knr"].cast<at::Tensor>().contiguous();
    }
    if (packed.contains("codes_f32_knr16")) {
        w.codes_f32_knr16 = packed["codes_f32_knr16"].cast<at::Tensor>().contiguous();
    }
    w.scales = packed["scales"].cast<at::Tensor>().contiguous();
    if (packed.contains("scales_padded")) {
        w.scales_padded = packed["scales_padded"].cast<at::Tensor>().contiguous();
    }
    if (packed.contains("scales_padded16")) {
        w.scales_padded16 = packed["scales_padded16"].cast<at::Tensor>().contiguous();
    }
    w.shape = packed["shape"].cast<std::vector<int64_t>>();
    w.group_size = packed["group_size"].cast<int64_t>();
    w.bits = packed["bits"].cast<int64_t>();
    TORCH_CHECK(w.codes.is_cpu() && w.codes.scalar_type() == at::kShort,
                "packed dyop codes must be a CPU int16 tensor");
    TORCH_CHECK(w.scales.is_cpu() && w.scales.scalar_type() == at::kFloat,
                "packed dyop scales must be a CPU float32 tensor");
    TORCH_CHECK(w.codes.dim() == 2, "packed dyop codes must be [out, K]");
    TORCH_CHECK(w.scales.dim() == 2, "packed dyop scales must be [out, blocks]");
    w.out = w.codes.size(0);
    w.k = w.codes.size(1);
    w.out_padded = w.codes_knr.defined() ? w.codes_knr.size(0) * 8 : w.out;
    TORCH_CHECK(w.scales.size(0) == w.out, "scale/code output dimension mismatch");
    TORCH_CHECK(w.group_size > 0, "group_size must be positive");
    return w;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline void load_i16_as_f32_8(const int16_t* p, float32x4_t& lo, float32x4_t& hi) {
    int16x8_t v = vld1q_s16(p);
    lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
    hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v)));
}

static inline void load_i8_as_f32_8(const int8_t* p, float32x4_t& lo, float32x4_t& hi) {
    int16x8_t v = vmovl_s8(vld1_s8(p));
    lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
    hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v)));
}

template <typename CodeT>
static inline void load_code_as_f32_8(const CodeT* p, float32x4_t& lo, float32x4_t& hi);

template <>
inline void load_code_as_f32_8<int16_t>(const int16_t* p, float32x4_t& lo, float32x4_t& hi) {
    load_i16_as_f32_8(p, lo, hi);
}

template <>
inline void load_code_as_f32_8<int8_t>(const int8_t* p, float32x4_t& lo, float32x4_t& hi) {
    load_i8_as_f32_8(p, lo, hi);
}

template <>
inline void load_code_as_f32_8<float>(const float* p, float32x4_t& lo, float32x4_t& hi) {
    lo = vld1q_f32(p);
    hi = vld1q_f32(p + 4);
}

static inline void accumulate_4k_8cols(
    float32x4_t& acc_lo,
    float32x4_t& acc_hi,
    const float* activations,
    int64_t k,
    float32x4_t w0_lo,
    float32x4_t w0_hi,
    float32x4_t w1_lo,
    float32x4_t w1_hi,
    float32x4_t w2_lo,
    float32x4_t w2_hi,
    float32x4_t w3_lo,
    float32x4_t w3_hi
) {
    float32x4_t a = vld1q_f32(activations + k);
    acc_lo = vfmaq_laneq_f32(acc_lo, w0_lo, a, 0);
    acc_hi = vfmaq_laneq_f32(acc_hi, w0_hi, a, 0);
    acc_lo = vfmaq_laneq_f32(acc_lo, w1_lo, a, 1);
    acc_hi = vfmaq_laneq_f32(acc_hi, w1_hi, a, 1);
    acc_lo = vfmaq_laneq_f32(acc_lo, w2_lo, a, 2);
    acc_hi = vfmaq_laneq_f32(acc_hi, w2_hi, a, 2);
    acc_lo = vfmaq_laneq_f32(acc_lo, w3_lo, a, 3);
    acc_hi = vfmaq_laneq_f32(acc_hi, w3_hi, a, 3);
}

template <typename CodeT>
static inline void linear_microkernel_4x8(
    const float* input,
    int64_t input_stride,
    const CodeT* weight_block,
    const float* scales,
    const float* bias,
    float* output,
    int64_t output_stride,
    int64_t k_size,
    int valid_m,
    int valid_n
) {
    float32x4_t acc0_lo = vdupq_n_f32(0.0f), acc0_hi = vdupq_n_f32(0.0f);
    float32x4_t acc1_lo = vdupq_n_f32(0.0f), acc1_hi = vdupq_n_f32(0.0f);
    float32x4_t acc2_lo = vdupq_n_f32(0.0f), acc2_hi = vdupq_n_f32(0.0f);
    float32x4_t acc3_lo = vdupq_n_f32(0.0f), acc3_hi = vdupq_n_f32(0.0f);

    const float* a0 = input;
    const float* a1 = input + input_stride;
    const float* a2 = input + 2 * input_stride;
    const float* a3 = input + 3 * input_stride;

    int64_t k = 0;
    for (; k + 3 < k_size; k += 4) {
        float32x4_t w0_lo, w0_hi, w1_lo, w1_hi, w2_lo, w2_hi, w3_lo, w3_hi;
        load_code_as_f32_8(weight_block + (k + 0) * 8, w0_lo, w0_hi);
        load_code_as_f32_8(weight_block + (k + 1) * 8, w1_lo, w1_hi);
        load_code_as_f32_8(weight_block + (k + 2) * 8, w2_lo, w2_hi);
        load_code_as_f32_8(weight_block + (k + 3) * 8, w3_lo, w3_hi);
        if (valid_m > 0) {
            float32x4_t a = vld1q_f32(a0 + k);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w0_lo, a, 0);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w0_hi, a, 0);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w1_lo, a, 1);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w1_hi, a, 1);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w2_lo, a, 2);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w2_hi, a, 2);
            acc0_lo = vfmaq_laneq_f32(acc0_lo, w3_lo, a, 3);
            acc0_hi = vfmaq_laneq_f32(acc0_hi, w3_hi, a, 3);
        }
        if (valid_m > 1) {
            float32x4_t a = vld1q_f32(a1 + k);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w0_lo, a, 0);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w0_hi, a, 0);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w1_lo, a, 1);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w1_hi, a, 1);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w2_lo, a, 2);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w2_hi, a, 2);
            acc1_lo = vfmaq_laneq_f32(acc1_lo, w3_lo, a, 3);
            acc1_hi = vfmaq_laneq_f32(acc1_hi, w3_hi, a, 3);
        }
        if (valid_m > 2) {
            float32x4_t a = vld1q_f32(a2 + k);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w0_lo, a, 0);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w0_hi, a, 0);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w1_lo, a, 1);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w1_hi, a, 1);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w2_lo, a, 2);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w2_hi, a, 2);
            acc2_lo = vfmaq_laneq_f32(acc2_lo, w3_lo, a, 3);
            acc2_hi = vfmaq_laneq_f32(acc2_hi, w3_hi, a, 3);
        }
        if (valid_m > 3) {
            float32x4_t a = vld1q_f32(a3 + k);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w0_lo, a, 0);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w0_hi, a, 0);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w1_lo, a, 1);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w1_hi, a, 1);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w2_lo, a, 2);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w2_hi, a, 2);
            acc3_lo = vfmaq_laneq_f32(acc3_lo, w3_lo, a, 3);
            acc3_hi = vfmaq_laneq_f32(acc3_hi, w3_hi, a, 3);
        }
    }
    for (; k < k_size; ++k) {
        float32x4_t w_lo, w_hi;
        load_code_as_f32_8(weight_block + k * 8, w_lo, w_hi);
        if (valid_m > 0) {
            float32x4_t a = vdupq_n_f32(a0[k]);
            acc0_lo = vfmaq_f32(acc0_lo, a, w_lo);
            acc0_hi = vfmaq_f32(acc0_hi, a, w_hi);
        }
        if (valid_m > 1) {
            float32x4_t a = vdupq_n_f32(a1[k]);
            acc1_lo = vfmaq_f32(acc1_lo, a, w_lo);
            acc1_hi = vfmaq_f32(acc1_hi, a, w_hi);
        }
        if (valid_m > 2) {
            float32x4_t a = vdupq_n_f32(a2[k]);
            acc2_lo = vfmaq_f32(acc2_lo, a, w_lo);
            acc2_hi = vfmaq_f32(acc2_hi, a, w_hi);
        }
        if (valid_m > 3) {
            float32x4_t a = vdupq_n_f32(a3[k]);
            acc3_lo = vfmaq_f32(acc3_lo, a, w_lo);
            acc3_hi = vfmaq_f32(acc3_hi, a, w_hi);
        }
    }

    float32x4_t s0 = vld1q_f32(scales);
    float32x4_t s1 = vld1q_f32(scales + 4);
    float32x4_t b0 = bias ? vld1q_f32(bias) : vdupq_n_f32(0.0f);
    float32x4_t b1 = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.0f);
    float32x4_t out0_lo = vfmaq_f32(b0, acc0_lo, s0);
    float32x4_t out0_hi = vfmaq_f32(b1, acc0_hi, s1);
    float32x4_t out1_lo = vfmaq_f32(b0, acc1_lo, s0);
    float32x4_t out1_hi = vfmaq_f32(b1, acc1_hi, s1);
    float32x4_t out2_lo = vfmaq_f32(b0, acc2_lo, s0);
    float32x4_t out2_hi = vfmaq_f32(b1, acc2_hi, s1);
    float32x4_t out3_lo = vfmaq_f32(b0, acc3_lo, s0);
    float32x4_t out3_hi = vfmaq_f32(b1, acc3_hi, s1);
    if (valid_n == 8) {
        if (valid_m > 0) {
            vst1q_f32(output, out0_lo);
            vst1q_f32(output + 4, out0_hi);
        }
        if (valid_m > 1) {
            vst1q_f32(output + output_stride, out1_lo);
            vst1q_f32(output + output_stride + 4, out1_hi);
        }
        if (valid_m > 2) {
            vst1q_f32(output + 2 * output_stride, out2_lo);
            vst1q_f32(output + 2 * output_stride + 4, out2_hi);
        }
        if (valid_m > 3) {
            vst1q_f32(output + 3 * output_stride, out3_lo);
            vst1q_f32(output + 3 * output_stride + 4, out3_hi);
        }
    } else {
        float tmp[4][8];
        if (valid_m > 0) {
            vst1q_f32(tmp[0], out0_lo);
            vst1q_f32(tmp[0] + 4, out0_hi);
        }
        if (valid_m > 1) {
            vst1q_f32(tmp[1], out1_lo);
            vst1q_f32(tmp[1] + 4, out1_hi);
        }
        if (valid_m > 2) {
            vst1q_f32(tmp[2], out2_lo);
            vst1q_f32(tmp[2] + 4, out2_hi);
        }
        if (valid_m > 3) {
            vst1q_f32(tmp[3], out3_lo);
            vst1q_f32(tmp[3] + 4, out3_hi);
        }
        for (int m = 0; m < valid_m; ++m) {
            for (int n = 0; n < valid_n; ++n) {
                output[m * output_stride + n] = tmp[m][n];
            }
        }
    }
}

template <typename CodeT>
static inline void linear_microkernel_8x8(
    const float* input,
    int64_t input_stride,
    const CodeT* weight_block,
    const float* scales,
    const float* bias,
    float* output,
    int64_t output_stride,
    int64_t k_size,
    int valid_m,
    int valid_n
) {
    float32x4_t acc0_lo = vdupq_n_f32(0.0f), acc0_hi = vdupq_n_f32(0.0f);
    float32x4_t acc1_lo = vdupq_n_f32(0.0f), acc1_hi = vdupq_n_f32(0.0f);
    float32x4_t acc2_lo = vdupq_n_f32(0.0f), acc2_hi = vdupq_n_f32(0.0f);
    float32x4_t acc3_lo = vdupq_n_f32(0.0f), acc3_hi = vdupq_n_f32(0.0f);
    float32x4_t acc4_lo = vdupq_n_f32(0.0f), acc4_hi = vdupq_n_f32(0.0f);
    float32x4_t acc5_lo = vdupq_n_f32(0.0f), acc5_hi = vdupq_n_f32(0.0f);
    float32x4_t acc6_lo = vdupq_n_f32(0.0f), acc6_hi = vdupq_n_f32(0.0f);
    float32x4_t acc7_lo = vdupq_n_f32(0.0f), acc7_hi = vdupq_n_f32(0.0f);

    const float* a0 = input;
    const float* a1 = input + input_stride;
    const float* a2 = input + 2 * input_stride;
    const float* a3 = input + 3 * input_stride;
    const float* a4 = input + 4 * input_stride;
    const float* a5 = input + 5 * input_stride;
    const float* a6 = input + 6 * input_stride;
    const float* a7 = input + 7 * input_stride;

    int64_t k = 0;
    for (; k + 3 < k_size; k += 4) {
        float32x4_t w0_lo, w0_hi, w1_lo, w1_hi, w2_lo, w2_hi, w3_lo, w3_hi;
        load_code_as_f32_8(weight_block + (k + 0) * 8, w0_lo, w0_hi);
        load_code_as_f32_8(weight_block + (k + 1) * 8, w1_lo, w1_hi);
        load_code_as_f32_8(weight_block + (k + 2) * 8, w2_lo, w2_hi);
        load_code_as_f32_8(weight_block + (k + 3) * 8, w3_lo, w3_hi);
        if (valid_m > 0) {
            accumulate_4k_8cols(
                acc0_lo, acc0_hi, a0, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 1) {
            accumulate_4k_8cols(
                acc1_lo, acc1_hi, a1, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 2) {
            accumulate_4k_8cols(
                acc2_lo, acc2_hi, a2, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 3) {
            accumulate_4k_8cols(
                acc3_lo, acc3_hi, a3, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 4) {
            accumulate_4k_8cols(
                acc4_lo, acc4_hi, a4, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 5) {
            accumulate_4k_8cols(
                acc5_lo, acc5_hi, a5, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 6) {
            accumulate_4k_8cols(
                acc6_lo, acc6_hi, a6, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
        if (valid_m > 7) {
            accumulate_4k_8cols(
                acc7_lo, acc7_hi, a7, k, w0_lo, w0_hi, w1_lo, w1_hi,
                w2_lo, w2_hi, w3_lo, w3_hi
            );
        }
    }
    for (; k < k_size; ++k) {
        float32x4_t w_lo, w_hi;
        load_code_as_f32_8(weight_block + k * 8, w_lo, w_hi);
        if (valid_m > 0) {
            float32x4_t a = vdupq_n_f32(a0[k]);
            acc0_lo = vfmaq_f32(acc0_lo, a, w_lo);
            acc0_hi = vfmaq_f32(acc0_hi, a, w_hi);
        }
        if (valid_m > 1) {
            float32x4_t a = vdupq_n_f32(a1[k]);
            acc1_lo = vfmaq_f32(acc1_lo, a, w_lo);
            acc1_hi = vfmaq_f32(acc1_hi, a, w_hi);
        }
        if (valid_m > 2) {
            float32x4_t a = vdupq_n_f32(a2[k]);
            acc2_lo = vfmaq_f32(acc2_lo, a, w_lo);
            acc2_hi = vfmaq_f32(acc2_hi, a, w_hi);
        }
        if (valid_m > 3) {
            float32x4_t a = vdupq_n_f32(a3[k]);
            acc3_lo = vfmaq_f32(acc3_lo, a, w_lo);
            acc3_hi = vfmaq_f32(acc3_hi, a, w_hi);
        }
        if (valid_m > 4) {
            float32x4_t a = vdupq_n_f32(a4[k]);
            acc4_lo = vfmaq_f32(acc4_lo, a, w_lo);
            acc4_hi = vfmaq_f32(acc4_hi, a, w_hi);
        }
        if (valid_m > 5) {
            float32x4_t a = vdupq_n_f32(a5[k]);
            acc5_lo = vfmaq_f32(acc5_lo, a, w_lo);
            acc5_hi = vfmaq_f32(acc5_hi, a, w_hi);
        }
        if (valid_m > 6) {
            float32x4_t a = vdupq_n_f32(a6[k]);
            acc6_lo = vfmaq_f32(acc6_lo, a, w_lo);
            acc6_hi = vfmaq_f32(acc6_hi, a, w_hi);
        }
        if (valid_m > 7) {
            float32x4_t a = vdupq_n_f32(a7[k]);
            acc7_lo = vfmaq_f32(acc7_lo, a, w_lo);
            acc7_hi = vfmaq_f32(acc7_hi, a, w_hi);
        }
    }

    float32x4_t s0 = vld1q_f32(scales);
    float32x4_t s1 = vld1q_f32(scales + 4);
    float32x4_t b0 = bias ? vld1q_f32(bias) : vdupq_n_f32(0.0f);
    float32x4_t b1 = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.0f);
    float32x4_t out0_lo = vfmaq_f32(b0, acc0_lo, s0);
    float32x4_t out0_hi = vfmaq_f32(b1, acc0_hi, s1);
    float32x4_t out1_lo = vfmaq_f32(b0, acc1_lo, s0);
    float32x4_t out1_hi = vfmaq_f32(b1, acc1_hi, s1);
    float32x4_t out2_lo = vfmaq_f32(b0, acc2_lo, s0);
    float32x4_t out2_hi = vfmaq_f32(b1, acc2_hi, s1);
    float32x4_t out3_lo = vfmaq_f32(b0, acc3_lo, s0);
    float32x4_t out3_hi = vfmaq_f32(b1, acc3_hi, s1);
    float32x4_t out4_lo = vfmaq_f32(b0, acc4_lo, s0);
    float32x4_t out4_hi = vfmaq_f32(b1, acc4_hi, s1);
    float32x4_t out5_lo = vfmaq_f32(b0, acc5_lo, s0);
    float32x4_t out5_hi = vfmaq_f32(b1, acc5_hi, s1);
    float32x4_t out6_lo = vfmaq_f32(b0, acc6_lo, s0);
    float32x4_t out6_hi = vfmaq_f32(b1, acc6_hi, s1);
    float32x4_t out7_lo = vfmaq_f32(b0, acc7_lo, s0);
    float32x4_t out7_hi = vfmaq_f32(b1, acc7_hi, s1);

    if (valid_n == 8) {
        if (valid_m > 0) {
            vst1q_f32(output, out0_lo);
            vst1q_f32(output + 4, out0_hi);
        }
        if (valid_m > 1) {
            vst1q_f32(output + output_stride, out1_lo);
            vst1q_f32(output + output_stride + 4, out1_hi);
        }
        if (valid_m > 2) {
            vst1q_f32(output + 2 * output_stride, out2_lo);
            vst1q_f32(output + 2 * output_stride + 4, out2_hi);
        }
        if (valid_m > 3) {
            vst1q_f32(output + 3 * output_stride, out3_lo);
            vst1q_f32(output + 3 * output_stride + 4, out3_hi);
        }
        if (valid_m > 4) {
            vst1q_f32(output + 4 * output_stride, out4_lo);
            vst1q_f32(output + 4 * output_stride + 4, out4_hi);
        }
        if (valid_m > 5) {
            vst1q_f32(output + 5 * output_stride, out5_lo);
            vst1q_f32(output + 5 * output_stride + 4, out5_hi);
        }
        if (valid_m > 6) {
            vst1q_f32(output + 6 * output_stride, out6_lo);
            vst1q_f32(output + 6 * output_stride + 4, out6_hi);
        }
        if (valid_m > 7) {
            vst1q_f32(output + 7 * output_stride, out7_lo);
            vst1q_f32(output + 7 * output_stride + 4, out7_hi);
        }
    } else {
        float tmp[8][8];
        if (valid_m > 0) {
            vst1q_f32(tmp[0], out0_lo);
            vst1q_f32(tmp[0] + 4, out0_hi);
        }
        if (valid_m > 1) {
            vst1q_f32(tmp[1], out1_lo);
            vst1q_f32(tmp[1] + 4, out1_hi);
        }
        if (valid_m > 2) {
            vst1q_f32(tmp[2], out2_lo);
            vst1q_f32(tmp[2] + 4, out2_hi);
        }
        if (valid_m > 3) {
            vst1q_f32(tmp[3], out3_lo);
            vst1q_f32(tmp[3] + 4, out3_hi);
        }
        if (valid_m > 4) {
            vst1q_f32(tmp[4], out4_lo);
            vst1q_f32(tmp[4] + 4, out4_hi);
        }
        if (valid_m > 5) {
            vst1q_f32(tmp[5], out5_lo);
            vst1q_f32(tmp[5] + 4, out5_hi);
        }
        if (valid_m > 6) {
            vst1q_f32(tmp[6], out6_lo);
            vst1q_f32(tmp[6] + 4, out6_hi);
        }
        if (valid_m > 7) {
            vst1q_f32(tmp[7], out7_lo);
            vst1q_f32(tmp[7] + 4, out7_hi);
        }
        for (int m = 0; m < valid_m; ++m) {
            for (int n = 0; n < valid_n; ++n) {
                output[m * output_stride + n] = tmp[m][n];
            }
        }
    }
}

static inline void make_conv_tile_4xk(
    const float* input,
    int64_t batch_index,
    int64_t p0,
    int valid_m,
    int64_t in_channels,
    int64_t ih,
    int64_t iw,
    int64_t oh_width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    float* tile
) {
    const int64_t k_size = in_channels * kernel_h * kernel_w;
    for (int m = 0; m < valid_m; ++m) {
        const int64_t pos = p0 + m;
        const int64_t oh_i = pos / oh_width;
        const int64_t ow_i = pos - oh_i * oh_width;
        float* dst = tile + int64_t(m) * k_size;
        int64_t kk = 0;
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            const float* base = input + ((batch_index * in_channels + ic) * ih) * iw;
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                const int64_t ih_i = oh_i * stride + kh - padding;
                for (int64_t kw = 0; kw < kernel_w; ++kw, ++kk) {
                    const int64_t iw_i = ow_i * stride + kw - padding;
                    dst[kk] = (ih_i >= 0 && ih_i < ih && iw_i >= 0 && iw_i < iw)
                        ? base[ih_i * iw + iw_i]
                        : 0.0f;
                }
            }
        }
    }
    for (int m = valid_m; m < 4; ++m) {
        std::fill(tile + int64_t(m) * k_size, tile + int64_t(m + 1) * k_size, 0.0f);
    }
}

template <typename CodeT>
static void conv2d_knr_neon(
    const float* input,
    const CodeT* codes_knr,
    const float* scales_padded,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t ih,
    int64_t iw,
    int64_t out_channels,
    int64_t blocks,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t oh,
    int64_t ow,
    int threads
) {
    const int64_t k_size = in_channels * kernel_h * kernel_w;
    const int64_t p_count = oh * ow;
    const int64_t tiles_per_batch = (p_count + 3) / 4;
    const int64_t total_tiles = batch * tiles_per_batch;
    const int64_t nblocks = (out_channels + 7) / 8;
    int64_t nb_group = nblocks;
    if (total_tiles < threads) {
        const int64_t groups_needed = (threads + std::max<int64_t>(1, total_tiles) - 1) /
                                      std::max<int64_t>(1, total_tiles);
        nb_group = std::max<int64_t>(1, (nblocks + groups_needed - 1) / groups_needed);
    }
    const int64_t ngroups = (nblocks + nb_group - 1) / nb_group;
    const int64_t total_tasks = total_tiles * ngroups;

    parallel_for_threads(0, total_tasks, threads, [&](int64_t begin, int64_t end) {
        std::vector<float> tile(static_cast<size_t>(4 * k_size));
        alignas(16) float tmp[4 * 8];
        for (int64_t task = begin; task < end; ++task) {
            const int64_t tile_group = task / ngroups;
            const int64_t group = task - tile_group * ngroups;
            const int64_t b = tile_group / tiles_per_batch;
            const int64_t tile_index = tile_group - b * tiles_per_batch;
            const int64_t p0 = tile_index * 4;
            const int valid_m = int(std::min<int64_t>(4, p_count - p0));
            make_conv_tile_4xk(
                input, b, p0, valid_m, in_channels, ih, iw, ow,
                kernel_h, kernel_w, stride, padding, tile.data()
            );
            const int64_t nb_begin = group * nb_group;
            const int64_t nb_end = std::min<int64_t>(nblocks, nb_begin + nb_group);
            for (int64_t nb = nb_begin; nb < nb_end; ++nb) {
                const int64_t n0 = nb * 8;
                const int valid_n = int(std::min<int64_t>(8, out_channels - n0));
                linear_microkernel_4x8(
                    tile.data(),
                    k_size,
                    codes_knr + nb * k_size * 8,
                    scales_padded + n0 * blocks,
                    bias ? bias + n0 : nullptr,
                    tmp,
                    8,
                    k_size,
                    valid_m,
                    valid_n
                );
                for (int m = 0; m < valid_m; ++m) {
                    const int64_t pos = p0 + m;
                    const int64_t oh_i = pos / ow;
                    const int64_t ow_i = pos - oh_i * ow;
                    for (int lane = 0; lane < valid_n; ++lane) {
                        const int64_t oc = n0 + lane;
                        output[((b * out_channels + oc) * oh + oh_i) * ow + ow_i] =
                            tmp[m * 8 + lane];
                    }
                }
            }
        }
    });
}

#if defined(__APPLE__) && defined(__aarch64__)
static bool linear_gemm_amx(
    const float* input,
    const PackedDyopWeight& packed,
    const float* bias,
    float* output,
    int64_t m,
    int64_t blocks
) {
    constexpr int64_t mr = 16;
    constexpr int64_t nr = 16;
    if (!packed.codes_f32_knr16.defined() || !packed.scales_padded.defined()) return false;
    if (m < 8) return false;
    if (packed.group_size < packed.k || blocks != 1) return false;

    const int threads = native_amx_thread_count();
    const int64_t k_size = packed.k;
    const int64_t mblocks = (m + mr - 1) / mr;
    const int64_t nblocks = (packed.out + nr - 1) / nr;
    if (nblocks > packed.codes_f32_knr16.size(0)) return false;

    const int64_t mp = mblocks * mr;
    std::vector<float, AlignedAllocator<float>> a_t(static_cast<size_t>(k_size * mp), 0.0f);
    parallel_for_threads(0, k_size, threads, [&](int64_t begin, int64_t end) {
        for (int64_t k = begin; k < end; ++k) {
            float* dst = a_t.data() + k * mp;
            const float* src = input + k;
            for (int64_t row = 0; row < m; ++row) {
                dst[row] = src[row * k_size];
            }
        }
    });

    const float* codes_f32 = packed.codes_f32_knr16.data_ptr<float>();
    const float* scales_padded = packed.scales_padded.data_ptr<float>();

    parallel_for_threads(0, nblocks, threads, [&](int64_t begin, int64_t end) {
        alignas(64) float z_buf[16][16];

        AMX_SET();
        for (int64_t nb = begin; nb < end; ++nb) {
            const int64_t n0 = nb * nr;
            const int valid_n = int(std::min<int64_t>(nr, packed.out - n0));
            const float* w_panel = codes_f32 + nb * k_size * nr;
            const float* scale = scales_padded + n0 * blocks;
            const float* bias_ptr = bias ? bias + n0 : nullptr;
            for (int64_t mb = 0; mb < mblocks; ++mb) {
                const int64_t m0 = mb * mr;
                const int valid_m = int(std::min<int64_t>(mr, m - m0));
                const int z_row = int(mb & 3);
                const float* a_panel = a_t.data() + m0;
                float* out_tile = output + m0 * packed.out + n0;

                AMX_FMA32(amx_enc_zero_z(z_row));
                for (int64_t k = 0; k < k_size; ++k) {
                    AMX_LDX((uint64_t)(a_panel + k * mp));
                    AMX_LDY((uint64_t)(w_panel + k * nr));
                    AMX_FMA32(amx_enc_fma32(0, 0, z_row));
                }

                for (int j = 0; j < 16; ++j) {
                    AMX_STZ(amx_enc_stz(&z_buf[j][0], j * 4 + z_row));
                }

                for (int i = 0; i < valid_m; ++i) {
                    float* dst = out_tile + int64_t(i) * packed.out;
                    for (int j = 0; j < valid_n; ++j) {
                        const float b = bias_ptr ? bias_ptr[j] : 0.0f;
                        dst[j] = z_buf[j][i] * scale[j] + b;
                    }
                }
            }
        }
        AMX_CLR();
    });
    return true;
}
#endif

#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m512 load_i16_as_f32_16_x86(const int16_t* p) {
    __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i v32 = _mm512_cvtepi16_epi32(v16);
    return _mm512_cvtepi32_ps(v32);
}

static inline void linear_microkernel_4x16_x86(
    const float* input,
    int64_t input_stride,
    const int16_t* weight_block,
    const float* scales,
    const float* bias,
    float* output,
    int64_t output_stride,
    int64_t k_size,
    int valid_m,
    int valid_n
) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    const float* a0 = input;
    const float* a1 = input + input_stride;
    const float* a2 = input + 2 * input_stride;
    const float* a3 = input + 3 * input_stride;

    int64_t k = 0;
    for (; k + 3 < k_size; k += 4) {
        __m512 w0 = load_i16_as_f32_16_x86(weight_block + (k + 0) * 16);
        __m512 w1 = load_i16_as_f32_16_x86(weight_block + (k + 1) * 16);
        __m512 w2 = load_i16_as_f32_16_x86(weight_block + (k + 2) * 16);
        __m512 w3 = load_i16_as_f32_16_x86(weight_block + (k + 3) * 16);

        if (valid_m > 0) {
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k + 0]), w0, acc0);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k + 1]), w1, acc0);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k + 2]), w2, acc0);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k + 3]), w3, acc0);
        }
        if (valid_m > 1) {
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k + 0]), w0, acc1);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k + 1]), w1, acc1);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k + 2]), w2, acc1);
            acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k + 3]), w3, acc1);
        }
        if (valid_m > 2) {
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k + 0]), w0, acc2);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k + 1]), w1, acc2);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k + 2]), w2, acc2);
            acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k + 3]), w3, acc2);
        }
        if (valid_m > 3) {
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k + 0]), w0, acc3);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k + 1]), w1, acc3);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k + 2]), w2, acc3);
            acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k + 3]), w3, acc3);
        }
    }
    for (; k < k_size; ++k) {
        __m512 w = load_i16_as_f32_16_x86(weight_block + k * 16);
        if (valid_m > 0) acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k]), w, acc0);
        if (valid_m > 1) acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k]), w, acc1);
        if (valid_m > 2) acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k]), w, acc2);
        if (valid_m > 3) acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k]), w, acc3);
    }

    const __m512 s = _mm512_loadu_ps(scales);
    const __m512 b = bias ? _mm512_loadu_ps(bias) : _mm512_setzero_ps();
    acc0 = _mm512_fmadd_ps(acc0, s, b);
    acc1 = _mm512_fmadd_ps(acc1, s, b);
    acc2 = _mm512_fmadd_ps(acc2, s, b);
    acc3 = _mm512_fmadd_ps(acc3, s, b);

    const __mmask16 mask = valid_n == 16 ? 0xFFFFu : static_cast<__mmask16>((1u << valid_n) - 1u);
    if (valid_m > 0) _mm512_mask_storeu_ps(output, mask, acc0);
    if (valid_m > 1) _mm512_mask_storeu_ps(output + output_stride, mask, acc1);
    if (valid_m > 2) _mm512_mask_storeu_ps(output + 2 * output_stride, mask, acc2);
    if (valid_m > 3) _mm512_mask_storeu_ps(output + 3 * output_stride, mask, acc3);
}

static void linear_gemm_x86(
    const float* input,
    const int16_t* codes_knr16,
    const float* scales_padded16,
    const float* bias,
    float* output,
    int64_t m,
    int64_t out,
    int64_t k_size,
    int64_t blocks,
    int threads
) {
    const int64_t mblocks = (m + 3) / 4;
    const int64_t nblocks = (out + 15) / 16;
    parallel_for_threads(0, mblocks * nblocks, threads, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t mb = task / nblocks;
            const int64_t nb = task - mb * nblocks;
            const int64_t m0 = mb * 4;
            const int64_t n0 = nb * 16;
            const int valid_m = int(std::min<int64_t>(4, m - m0));
            const int valid_n = int(std::min<int64_t>(16, out - n0));
            linear_microkernel_4x16_x86(
                input + m0 * k_size,
                k_size,
                codes_knr16 + nb * k_size * 16,
                scales_padded16 + n0 * blocks,
                bias ? bias + n0 : nullptr,
                output + m0 * out + n0,
                out,
                k_size,
                valid_m,
                valid_n
            );
        }
    });
}

static inline void embedding_decode_x86_i32(
    const int32_t* indices,
    int64_t count,
    const int16_t* codes,
    const float* scales,
    float* output,
    int64_t out,
    int64_t k_size,
    int threads
) {
    parallel_for_threads(0, count, threads, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t row = indices[i];
            TORCH_CHECK(row >= 0 && row < out, "embedding index out of range");
            const int16_t* src = codes + row * k_size;
            float* dst = output + i * k_size;
            const __m512 scale = _mm512_set1_ps(scales[row]);
            int64_t k = 0;
            for (; k + 31 < k_size; k += 32) {
                __m512 x0 = load_i16_as_f32_16_x86(src + k);
                __m512 x1 = load_i16_as_f32_16_x86(src + k + 16);
                _mm512_storeu_ps(dst + k, _mm512_mul_ps(x0, scale));
                _mm512_storeu_ps(dst + k + 16, _mm512_mul_ps(x1, scale));
            }
            for (; k + 15 < k_size; k += 16) {
                __m512 x = load_i16_as_f32_16_x86(src + k);
                _mm512_storeu_ps(dst + k, _mm512_mul_ps(x, scale));
            }
            for (; k < k_size; ++k) dst[k] = static_cast<float>(src[k]) * scales[row];
        }
    });
}

static inline void embedding_decode_x86_i64(
    const int64_t* indices,
    int64_t count,
    const int16_t* codes,
    const float* scales,
    float* output,
    int64_t out,
    int64_t k_size,
    int threads
) {
    parallel_for_threads(0, count, threads, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t row = indices[i];
            TORCH_CHECK(row >= 0 && row < out, "embedding index out of range");
            const int16_t* src = codes + row * k_size;
            float* dst = output + i * k_size;
            const __m512 scale = _mm512_set1_ps(scales[row]);
            int64_t k = 0;
            for (; k + 31 < k_size; k += 32) {
                __m512 x0 = load_i16_as_f32_16_x86(src + k);
                __m512 x1 = load_i16_as_f32_16_x86(src + k + 16);
                _mm512_storeu_ps(dst + k, _mm512_mul_ps(x0, scale));
                _mm512_storeu_ps(dst + k + 16, _mm512_mul_ps(x1, scale));
            }
            for (; k + 15 < k_size; k += 16) {
                __m512 x = load_i16_as_f32_16_x86(src + k);
                _mm512_storeu_ps(dst + k, _mm512_mul_ps(x, scale));
            }
            for (; k < k_size; ++k) dst[k] = static_cast<float>(src[k]) * scales[row];
        }
    });
}

static inline void make_conv_tile_4xk_x86(
    const float* input,
    int64_t batch_index,
    int64_t p0,
    int valid_m,
    int64_t in_channels,
    int64_t ih,
    int64_t iw,
    int64_t oh_width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    float* tile
) {
    const int64_t k_size = in_channels * kernel_h * kernel_w;
    for (int m = 0; m < valid_m; ++m) {
        const int64_t pos = p0 + m;
        const int64_t oh_i = pos / oh_width;
        const int64_t ow_i = pos - oh_i * oh_width;
        float* dst = tile + int64_t(m) * k_size;
        int64_t kk = 0;
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            const float* base = input + ((batch_index * in_channels + ic) * ih) * iw;
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                const int64_t ih_i = oh_i * stride + kh - padding;
                for (int64_t kw = 0; kw < kernel_w; ++kw, ++kk) {
                    const int64_t iw_i = ow_i * stride + kw - padding;
                    dst[kk] = (ih_i >= 0 && ih_i < ih && iw_i >= 0 && iw_i < iw)
                        ? base[ih_i * iw + iw_i]
                        : 0.0f;
                }
            }
        }
    }
    for (int m = valid_m; m < 4; ++m) {
        std::memset(tile + int64_t(m) * k_size, 0, static_cast<size_t>(k_size) * sizeof(float));
    }
}

static void conv2d_knr16_x86(
    const float* input,
    const int16_t* codes_knr16,
    const float* scales_padded16,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t ih,
    int64_t iw,
    int64_t out_channels,
    int64_t blocks,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t oh,
    int64_t ow,
    int threads
) {
    const int64_t k_size = in_channels * kernel_h * kernel_w;
    const int64_t p_count = oh * ow;
    const int64_t tiles_per_batch = (p_count + 3) / 4;
    const int64_t total_tiles = batch * tiles_per_batch;
    const int64_t nblocks = (out_channels + 15) / 16;
    int64_t nb_group = nblocks;
    if (total_tiles < threads) {
        const int64_t groups_needed = (threads + std::max<int64_t>(1, total_tiles) - 1) /
                                      std::max<int64_t>(1, total_tiles);
        nb_group = std::max<int64_t>(1, (nblocks + groups_needed - 1) / groups_needed);
    }
    const int64_t ngroups = (nblocks + nb_group - 1) / nb_group;
    const int64_t total_tasks = total_tiles * ngroups;

    parallel_for_threads(0, total_tasks, threads, [&](int64_t begin, int64_t end) {
        std::vector<float, AlignedAllocator<float>> tile(static_cast<size_t>(4 * k_size));
        alignas(64) float tmp[4 * 16];
        for (int64_t task = begin; task < end; ++task) {
            const int64_t tile_group = task / ngroups;
            const int64_t group = task - tile_group * ngroups;
            const int64_t b = tile_group / tiles_per_batch;
            const int64_t tile_index = tile_group - b * tiles_per_batch;
            const int64_t p0 = tile_index * 4;
            const int valid_m = int(std::min<int64_t>(4, p_count - p0));
            make_conv_tile_4xk_x86(
                input, b, p0, valid_m, in_channels, ih, iw, ow,
                kernel_h, kernel_w, stride, padding, tile.data()
            );
            const int64_t nb_begin = group * nb_group;
            const int64_t nb_end = std::min<int64_t>(nblocks, nb_begin + nb_group);
            for (int64_t nb = nb_begin; nb < nb_end; ++nb) {
                const int64_t n0 = nb * 16;
                const int valid_n = int(std::min<int64_t>(16, out_channels - n0));
                linear_microkernel_4x16_x86(
                    tile.data(),
                    k_size,
                    codes_knr16 + nb * k_size * 16,
                    scales_padded16 + n0 * blocks,
                    bias ? bias + n0 : nullptr,
                    tmp,
                    16,
                    k_size,
                    valid_m,
                    valid_n
                );
                for (int m = 0; m < valid_m; ++m) {
                    const int64_t pos = p0 + m;
                    const int64_t oh_i = pos / ow;
                    const int64_t ow_i = pos - oh_i * ow;
                    for (int lane = 0; lane < valid_n; ++lane) {
                        const int64_t oc = n0 + lane;
                        output[((b * out_channels + oc) * oh + oh_i) * ow + ow_i] =
                            tmp[m * 16 + lane];
                    }
                }
            }
        }
    });
}
#endif

static inline float dyop_dot(
    const float* input,
    const int16_t* codes,
    const float* scales,
    int64_t k,
    int64_t group_size
) {
    float acc = 0.0f;
    int64_t block = 0;
    for (int64_t start = 0; start < k; start += group_size, ++block) {
        const int64_t end = std::min<int64_t>(start + group_size, k);
        float partial = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int64_t i = start;
        for (; i + 7 < end; i += 8) {
            int16x8_t w16 = vld1q_s16(codes + i);
            int32x4_t w0_32 = vmovl_s16(vget_low_s16(w16));
            int32x4_t w1_32 = vmovl_s16(vget_high_s16(w16));
            float32x4_t w0 = vcvtq_f32_s32(w0_32);
            float32x4_t w1 = vcvtq_f32_s32(w1_32);
            float32x4_t x0 = vld1q_f32(input + i);
            float32x4_t x1 = vld1q_f32(input + i + 4);
            acc0 = vfmaq_f32(acc0, x0, w0);
            acc1 = vfmaq_f32(acc1, x1, w1);
        }
        partial = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < end; ++i) {
            partial += input[i] * static_cast<float>(codes[i]);
        }
#else
        for (int64_t i = start; i < end; ++i) {
            partial += input[i] * static_cast<float>(codes[i]);
        }
#endif
        acc += partial * scales[block];
    }
    return acc;
}

py::dict pack_native_cpu_weight(
    const at::Tensor& signs_in,
    const at::Tensor& magnitude_in,
    const at::Tensor& exponents_in,
    int64_t max_bits,
    int64_t group_size,
    int64_t bits
) {
    TORCH_CHECK(bits >= 2 && bits <= max_bits, "bits must be in [2, max_bits]");
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    auto signs = signs_in.to(at::kCPU).contiguous();
    auto magnitude = magnitude_in.to(at::kCPU).contiguous();
    auto exponents = exponents_in.to(at::kCPU).contiguous();
    TORCH_CHECK(signs.scalar_type() == at::kChar, "signs must be int8");
    TORCH_CHECK(magnitude.scalar_type() == at::kInt, "magnitude_code must be int32");
    TORCH_CHECK(exponents.scalar_type() == at::kShort, "exponents must be int16");
    TORCH_CHECK(signs.sizes() == magnitude.sizes(), "sign/code shape mismatch");
    TORCH_CHECK(signs.dim() >= 2, "dyop weight must have an output dimension");

    const int64_t out = signs.size(0);
    const int64_t k = signs.numel() / out;
    const int64_t blocks = exponents.dim() == 1 ? 1 : exponents.size(1);
    TORCH_CHECK(exponents.size(0) == out, "exponent output dimension mismatch");
    TORCH_CHECK((k + group_size - 1) / group_size == blocks,
                "group_size does not match exponent block count");

    const int64_t prefix_magnitude_bits = bits - 1;
    const int64_t shift = (max_bits - 1) - prefix_magnitude_bits;
    std::vector<int64_t> shape(signs.sizes().begin(), signs.sizes().end());

    auto codes = at::empty({out, k}, signs.options().dtype(at::kShort));
    const int64_t out_padded = ((out + 7) / 8) * 8;
    const int64_t out_padded16 = ((out + 15) / 16) * 16;
    auto codes_knr = at::zeros({out_padded / 8, k, 8}, signs.options().dtype(at::kShort));
    auto codes_knr16 = at::zeros({out_padded16 / 16, k, 16}, signs.options().dtype(at::kShort));
    auto codes_i8_knr = at::empty({0}, signs.options().dtype(at::kChar));
    const bool use_i8_knr = bits <= 7;
    if (use_i8_knr) {
        codes_i8_knr = at::zeros({out_padded / 8, k, 8}, signs.options().dtype(at::kChar));
    }
    auto codes_f32_knr = at::zeros({out_padded / 8, k, 8}, signs.options().dtype(at::kFloat));
    auto codes_f32_knr16 = at::zeros({out_padded16 / 16, k, 16}, signs.options().dtype(at::kFloat));
    auto scales = at::empty({out, blocks}, signs.options().dtype(at::kFloat));
    auto scales_padded = at::zeros({out_padded, blocks}, signs.options().dtype(at::kFloat));
    auto scales_padded16 = at::zeros({out_padded16, blocks}, signs.options().dtype(at::kFloat));

    const int8_t* sign_ptr = signs.data_ptr<int8_t>();
    const int32_t* mag_ptr = magnitude.data_ptr<int32_t>();
    const int16_t* exp_ptr = exponents.data_ptr<int16_t>();
    int16_t* code_ptr = codes.data_ptr<int16_t>();
    int16_t* knr_ptr = codes_knr.data_ptr<int16_t>();
    int16_t* knr16_ptr = codes_knr16.data_ptr<int16_t>();
    int8_t* knr_i8_ptr = use_i8_knr ? codes_i8_knr.data_ptr<int8_t>() : nullptr;
    float* knr_f32_ptr = codes_f32_knr.data_ptr<float>();
    float* knr16_f32_ptr = codes_f32_knr16.data_ptr<float>();
    float* scale_ptr = scales.data_ptr<float>();
    float* scale_pad_ptr = scales_padded.data_ptr<float>();
    float* scale_pad16_ptr = scales_padded16.data_ptr<float>();

    at::parallel_for(0, out, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            for (int64_t b = 0; b < blocks; ++b) {
                const int16_t exponent = exp_ptr[n * blocks + b];
                const float scale = std::ldexp(1.0f, int(exponent + shift - 1));
                scale_ptr[n * blocks + b] = scale;
                scale_pad_ptr[n * blocks + b] = scale;
                scale_pad16_ptr[n * blocks + b] = scale;
            }
            const int64_t row_offset = n * k;
            for (int64_t i = 0; i < k; ++i) {
                const int32_t prefix = mag_ptr[row_offset + i] >> shift;
                const int32_t odd = prefix * 2 + 1;
                const int16_t code =
                    sign_ptr[row_offset + i] < 0 ? int16_t(-odd) : int16_t(odd);
                code_ptr[row_offset + i] = code;
                const int64_t nb = n / 8;
                const int64_t lane = n - nb * 8;
                const int64_t knr_offset = (nb * k + i) * 8 + lane;
                knr_ptr[knr_offset] = code;
                knr_f32_ptr[knr_offset] = static_cast<float>(code);
                const int64_t nb16 = n / 16;
                const int64_t lane16 = n - nb16 * 16;
                const int64_t knr16_offset = (nb16 * k + i) * 16 + lane16;
                knr16_ptr[knr16_offset] = code;
                knr16_f32_ptr[knr16_offset] = static_cast<float>(code);
                if (knr_i8_ptr) {
                    knr_i8_ptr[knr_offset] = static_cast<int8_t>(code);
                }
            }
        }
    });

    py::dict packed;
    packed["codes"] = codes;
    packed["codes_knr"] = codes_knr;
    packed["codes_knr16"] = codes_knr16;
    if (use_i8_knr) packed["codes_i8_knr"] = codes_i8_knr;
    packed["codes_f32_knr"] = codes_f32_knr;
    packed["codes_f32_knr16"] = codes_f32_knr16;
    packed["scales"] = scales;
    packed["scales_padded"] = scales_padded;
    packed["scales_padded16"] = scales_padded16;
    packed["shape"] = shape;
    packed["group_size"] = group_size;
    packed["bits"] = bits;
    return packed;
}

static at::Tensor dyadic_linear_packed_weight_native_cpu(
    const at::Tensor& input_in,
    const PackedDyopWeight& packed,
    const at::Tensor& bias_tensor_in
) {
    auto input = input_in.to(at::kCPU).contiguous();
    TORCH_CHECK(input.scalar_type() == at::kFloat, "native linear input must be CPU float32");
    TORCH_CHECK(input.dim() == 2, "native linear input must be [M, K]");
    TORCH_CHECK(input.size(1) == packed.k, "native linear input width mismatch");

    auto output = at::empty({input.size(0), packed.out}, input.options());
    const float* x = input.data_ptr<float>();
    const int16_t* codes = packed.codes.data_ptr<int16_t>();
    const float* scales = packed.scales.data_ptr<float>();
    const float* bias = nullptr;
    at::Tensor bias_tensor;
    if (bias_tensor_in.defined()) {
        bias_tensor = bias_tensor_in.to(at::kCPU).contiguous();
        TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias_tensor.numel() == packed.out, "bias width mismatch");
        bias = bias_tensor.data_ptr<float>();
    }
    float* y = output.data_ptr<float>();
    const int64_t m = input.size(0);
    const int64_t blocks = packed.scales.size(1);

#if defined(__APPLE__) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if (linear_gemm_amx(x, packed, bias, y, m, blocks)) {
        return output;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (
        packed.group_size >= packed.k &&
        blocks == 1 &&
        packed.codes_knr16.defined() &&
        packed.scales_padded16.defined()
    ) {
        linear_gemm_x86(
            x,
            packed.codes_knr16.data_ptr<int16_t>(),
            packed.scales_padded16.data_ptr<float>(),
            bias,
            y,
            m,
            packed.out,
            packed.k,
            blocks,
            native_thread_count()
        );
        return output;
    }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (packed.group_size >= packed.k && packed.codes_knr.defined() && packed.scales_padded.defined()) {
        const float* scales_padded = packed.scales_padded.data_ptr<float>();
        auto run_knr8 = [&](auto codes_knr) {
            const int64_t mblocks = (m + 7) / 8;
            const int64_t nblocks = (packed.out + 7) / 8;
            parallel_for_threads(0, mblocks * nblocks, native_thread_count(), [&](int64_t begin, int64_t end) {
                for (int64_t task = begin; task < end; ++task) {
                    const int64_t mb = task / nblocks;
                    const int64_t nb = task - mb * nblocks;
                    const int64_t m0 = mb * 8;
                    const int64_t n0 = nb * 8;
                    const int valid_m = int(std::min<int64_t>(8, m - m0));
                    const int valid_n = int(std::min<int64_t>(8, packed.out - n0));
                    linear_microkernel_8x8(
                        x + m0 * packed.k,
                        packed.k,
                        codes_knr + nb * packed.k * 8,
                        scales_padded + n0 * blocks,
                        bias ? bias + n0 : nullptr,
                        y + m0 * packed.out + n0,
                        packed.out,
                        packed.k,
                        valid_m,
                        valid_n
                    );
                }
            });
        };
        auto run_knr4 = [&](auto codes_knr) {
            const int64_t mblocks = (m + 3) / 4;
            const int64_t nblocks = (packed.out + 7) / 8;
            parallel_for_threads(0, mblocks * nblocks, native_thread_count(), [&](int64_t begin, int64_t end) {
                for (int64_t task = begin; task < end; ++task) {
                    const int64_t mb = task / nblocks;
                    const int64_t nb = task - mb * nblocks;
                    const int64_t m0 = mb * 4;
                    const int64_t n0 = nb * 8;
                    const int valid_m = int(std::min<int64_t>(4, m - m0));
                    const int valid_n = int(std::min<int64_t>(8, packed.out - n0));
                    linear_microkernel_4x8(
                        x + m0 * packed.k,
                        packed.k,
                        codes_knr + nb * packed.k * 8,
                        scales_padded + n0 * blocks,
                        bias ? bias + n0 : nullptr,
                        y + m0 * packed.out + n0,
                        packed.out,
                        packed.k,
                        valid_m,
                        valid_n
                    );
                }
            });
        };
        if (packed.codes_f32_knr.defined()) {
            const float* codes_f32_knr = packed.codes_f32_knr.data_ptr<float>();
            if (m >= 8) run_knr8(codes_f32_knr);
            else run_knr4(codes_f32_knr);
        } else if (packed.codes_i8_knr.defined()) {
            const int8_t* codes_i8_knr = packed.codes_i8_knr.data_ptr<int8_t>();
            if (m >= 8) run_knr8(codes_i8_knr);
            else run_knr4(codes_i8_knr);
        } else {
            const int16_t* codes_knr = packed.codes_knr.data_ptr<int16_t>();
            if (m >= 8) run_knr8(codes_knr);
            else run_knr4(codes_knr);
        }
        return output;
    }
#endif

    at::parallel_for(0, m * packed.out, 128, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t row = index / packed.out;
            const int64_t col = index - row * packed.out;
            float value = dyop_dot(
                x + row * packed.k,
                codes + col * packed.k,
                scales + col * blocks,
                packed.k,
                packed.group_size
            );
            if (bias) value += bias[col];
            y[index] = value;
        }
    });
    return output;
}

at::Tensor dyadic_linear_packed_native_cpu(
    const at::Tensor& input_in,
    const py::dict& packed_dict,
    py::object bias_obj
) {
    auto packed = unpack_weight(packed_dict);
    at::Tensor bias_tensor;
    if (!bias_obj.is_none()) {
        bias_tensor = bias_obj.cast<at::Tensor>();
    }
    return dyadic_linear_packed_weight_native_cpu(input_in, packed, bias_tensor);
}

at::Tensor dyadic_embedding_packed_native_cpu(
    const at::Tensor& indices_in,
    const py::dict& packed_dict
) {
    auto indices = indices_in.to(at::kCPU).contiguous();
    TORCH_CHECK(indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
                "indices must be int64 or int32");
    auto packed = unpack_weight(packed_dict);
    std::vector<int64_t> output_shape(indices.sizes().begin(), indices.sizes().end());
    output_shape.push_back(packed.k);
    auto output = at::empty(output_shape, packed.scales.options());

    const int16_t* codes = packed.codes.data_ptr<int16_t>();
    const float* scales = packed.scales.data_ptr<float>();
    float* y = output.data_ptr<float>();
    const int64_t count = indices.numel();
    const int64_t blocks = packed.scales.size(1);

#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (blocks == 1) {
        if (indices.scalar_type() == at::kLong) {
            embedding_decode_x86_i64(
                indices.data_ptr<int64_t>(),
                count,
                codes,
                scales,
                y,
                packed.out,
                packed.k,
                native_thread_count()
            );
        } else {
            embedding_decode_x86_i32(
                indices.data_ptr<int32_t>(),
                count,
                codes,
                scales,
                y,
                packed.out,
                packed.k,
                native_thread_count()
            );
        }
        return output;
    }
#endif

    auto decode_row = [&](int64_t i, int64_t row) {
        TORCH_CHECK(row >= 0 && row < packed.out, "embedding index out of range");
        const int16_t* src = codes + row * packed.k;
        const float* row_scales = scales + row * blocks;
        float* dst = y + i * packed.k;
        for (int64_t start = 0, block = 0; start < packed.k; start += packed.group_size, ++block) {
            const int64_t end = std::min<int64_t>(start + packed.group_size, packed.k);
            const float scale = row_scales[block];
            for (int64_t k = start; k < end; ++k) dst[k] = static_cast<float>(src[k]) * scale;
        }
    };

    if (indices.scalar_type() == at::kLong) {
        const int64_t* idx = indices.data_ptr<int64_t>();
        at::parallel_for(0, count, 16, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) decode_row(i, idx[i]);
        });
    } else {
        const int32_t* idx = indices.data_ptr<int32_t>();
        at::parallel_for(0, count, 16, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) decode_row(i, idx[i]);
        });
    }
    return output;
}

at::Tensor dyadic_qwen_mlp_packed_native_cpu(
    const at::Tensor& input,
    const py::dict& gate_packed,
    const py::dict& up_packed,
    const py::dict& down_packed,
    py::object gate_bias,
    py::object up_bias,
    py::object down_bias
) {
    auto gate = dyadic_linear_packed_native_cpu(input, gate_packed, gate_bias);
    auto up = dyadic_linear_packed_native_cpu(input, up_packed, up_bias);
    auto hidden = at::silu(gate).mul_(up);
    return dyadic_linear_packed_native_cpu(hidden, down_packed, down_bias);
}

static at::Tensor dyadic_qwen_mlp_packed_weight_native_cpu(
    const at::Tensor& input,
    const PackedDyopWeight& gate_packed,
    const PackedDyopWeight& up_packed,
    const PackedDyopWeight& down_packed,
    const at::Tensor& gate_bias,
    const at::Tensor& up_bias,
    const at::Tensor& down_bias,
    at::Tensor* hidden_workspace = nullptr
) {
    auto gate = dyadic_linear_packed_weight_native_cpu(input, gate_packed, gate_bias);
    auto up = dyadic_linear_packed_weight_native_cpu(input, up_packed, up_bias);
    at::Tensor hidden;
    if (hidden_workspace != nullptr) {
        if (
            !hidden_workspace->defined() ||
            hidden_workspace->sizes() != gate.sizes() ||
            hidden_workspace->scalar_type() != gate.scalar_type()
        ) {
            *hidden_workspace = at::empty_like(gate);
        }
        at::silu_out(*hidden_workspace, gate);
        hidden_workspace->mul_(up);
        hidden = *hidden_workspace;
    } else {
        hidden = at::silu(gate).mul_(up);
    }
    return dyadic_linear_packed_weight_native_cpu(hidden, down_packed, down_bias);
}

at::Tensor dyadic_qwen_mlp_stack_packed_native_cpu(
    const at::Tensor& input,
    const py::list& blocks
) {
    at::Tensor current = input;
    for (py::handle block_handle : blocks) {
        py::tuple block = py::cast<py::tuple>(block_handle);
        TORCH_CHECK(block.size() == 6, "each MLP stack block must contain six items");
        current = dyadic_qwen_mlp_packed_native_cpu(
            current,
            block[0].cast<py::dict>(),
            block[1].cast<py::dict>(),
            block[2].cast<py::dict>(),
            py::reinterpret_borrow<py::object>(block[3]),
            py::reinterpret_borrow<py::object>(block[4]),
            py::reinterpret_borrow<py::object>(block[5])
        );
    }
    return current;
}

struct PlannedMlpBlock {
    PackedDyopWeight gate_packed;
    PackedDyopWeight up_packed;
    PackedDyopWeight down_packed;
    at::Tensor gate_bias;
    at::Tensor up_bias;
    at::Tensor down_bias;
    at::Tensor hidden_workspace;
};

struct PlannedMlpStack {
    std::vector<PlannedMlpBlock> blocks;
};

py::capsule pack_qwen_mlp_stack_native_cpu(const py::list& blocks) {
    auto* plan = new PlannedMlpStack();
    plan->blocks.reserve(blocks.size());
    for (py::handle block_handle : blocks) {
        py::tuple block = py::cast<py::tuple>(block_handle);
        TORCH_CHECK(block.size() == 6, "each MLP stack block must contain six items");
        auto maybe_bias = [](py::handle value) -> at::Tensor {
            if (value.is_none()) return at::Tensor();
            return py::reinterpret_borrow<py::object>(value).cast<at::Tensor>();
        };
        plan->blocks.push_back(PlannedMlpBlock{
            unpack_weight(block[0].cast<py::dict>()),
            unpack_weight(block[1].cast<py::dict>()),
            unpack_weight(block[2].cast<py::dict>()),
            maybe_bias(block[3]),
            maybe_bias(block[4]),
            maybe_bias(block[5]),
        });
    }
    return py::capsule(plan, "dyadic_qwen_mlp_stack_plan", [](PyObject* capsule) {
        py::gil_scoped_acquire gil;
        auto* ptr = static_cast<PlannedMlpStack*>(
            PyCapsule_GetPointer(capsule, "dyadic_qwen_mlp_stack_plan")
        );
        delete ptr;
    });
}

at::Tensor dyadic_qwen_mlp_stack_plan_native_cpu(
    const at::Tensor& input,
    py::capsule plan_capsule
) {
    auto* plan = static_cast<PlannedMlpStack*>(
        PyCapsule_GetPointer(plan_capsule.ptr(), "dyadic_qwen_mlp_stack_plan")
    );
    TORCH_CHECK(plan != nullptr, "invalid Qwen MLP stack plan");
    at::Tensor current = input;
    for (PlannedMlpBlock& block : plan->blocks) {
        current = dyadic_qwen_mlp_packed_weight_native_cpu(
            current,
            block.gate_packed,
            block.up_packed,
            block.down_packed,
            block.gate_bias,
            block.up_bias,
            block.down_bias,
            &block.hidden_workspace
        );
    }
    return current;
}

at::Tensor dyadic_conv2d_packed_native_cpu(
    const at::Tensor& input_in,
    const py::dict& packed_dict,
    py::object bias_obj,
    int64_t stride,
    int64_t padding,
    int64_t kernel_h,
    int64_t kernel_w
) {
    auto input = input_in.to(at::kCPU).contiguous();
    TORCH_CHECK(input.scalar_type() == at::kFloat, "native Conv2d input must be CPU float32");
    TORCH_CHECK(input.dim() == 4, "native Conv2d input must be NCHW");
    auto packed = unpack_weight(packed_dict);
    TORCH_CHECK(kernel_h == 1 || kernel_h == 3, "native Conv2d supports 1x1 and 3x3 kernels");
    TORCH_CHECK(kernel_w == kernel_h, "native Conv2d supports square kernels");
    TORCH_CHECK(packed.k % (kernel_h * kernel_w) == 0, "packed Conv2d K/kernel mismatch");
    const int64_t in_channels = packed.k / (kernel_h * kernel_w);
    TORCH_CHECK(input.size(1) == in_channels, "native Conv2d input channel mismatch");
    const int64_t batch = input.size(0);
    const int64_t ih = input.size(2);
    const int64_t iw = input.size(3);
    const int64_t oh = (ih + 2 * padding - kernel_h) / stride + 1;
    const int64_t ow = (iw + 2 * padding - kernel_w) / stride + 1;
    auto output = at::empty({batch, packed.out, oh, ow}, input.options());

    at::Tensor bias_tensor;
    const float* bias = nullptr;
    if (!bias_obj.is_none()) {
        bias_tensor = bias_obj.cast<at::Tensor>().to(at::kCPU).contiguous();
        TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias_tensor.numel() == packed.out, "bias width mismatch");
        bias = bias_tensor.data_ptr<float>();
    }

    const float* x = input.data_ptr<float>();
    const int16_t* codes = packed.codes.data_ptr<int16_t>();
    const float* scales = packed.scales.data_ptr<float>();
    float* y = output.data_ptr<float>();
    const int64_t blocks = packed.scales.size(1);
    const int64_t total = batch * packed.out * oh * ow;

#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (
        packed.group_size >= packed.k &&
        blocks == 1 &&
        packed.codes_knr16.defined() &&
        packed.scales_padded16.defined()
    ) {
        conv2d_knr16_x86(
            x,
            packed.codes_knr16.data_ptr<int16_t>(),
            packed.scales_padded16.data_ptr<float>(),
            bias,
            y,
            batch,
            in_channels,
            ih,
            iw,
            packed.out,
            blocks,
            kernel_h,
            kernel_w,
            stride,
            padding,
            oh,
            ow,
            native_thread_count()
        );
        return output;
    }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (packed.group_size >= packed.k && packed.codes_knr.defined() && packed.scales_padded.defined()) {
        const float* scales_padded = packed.scales_padded.data_ptr<float>();
        const int threads = native_thread_count();
        if (packed.codes_f32_knr.defined()) {
            conv2d_knr_neon(
                x,
                packed.codes_f32_knr.data_ptr<float>(),
                scales_padded,
                bias,
                y,
                batch,
                in_channels,
                ih,
                iw,
                packed.out,
                blocks,
                kernel_h,
                kernel_w,
                stride,
                padding,
                oh,
                ow,
                threads
            );
        } else if (packed.codes_i8_knr.defined()) {
            conv2d_knr_neon(
                x,
                packed.codes_i8_knr.data_ptr<int8_t>(),
                scales_padded,
                bias,
                y,
                batch,
                in_channels,
                ih,
                iw,
                packed.out,
                blocks,
                kernel_h,
                kernel_w,
                stride,
                padding,
                oh,
                ow,
                threads
            );
        } else {
            conv2d_knr_neon(
                x,
                packed.codes_knr.data_ptr<int16_t>(),
                scales_padded,
                bias,
                y,
                batch,
                in_channels,
                ih,
                iw,
                packed.out,
                blocks,
                kernel_h,
                kernel_w,
                stride,
                padding,
                oh,
                ow,
                threads
            );
        }
        return output;
    }
#endif

    at::parallel_for(0, total, 64, [&](int64_t begin, int64_t end) {
        std::vector<float> window(static_cast<size_t>(packed.k));
        for (int64_t index = begin; index < end; ++index) {
            int64_t t = index;
            const int64_t ow_i = t % ow; t /= ow;
            const int64_t oh_i = t % oh; t /= oh;
            const int64_t oc = t % packed.out; t /= packed.out;
            const int64_t b = t;
            int64_t kk = 0;
            for (int64_t ic = 0; ic < in_channels; ++ic) {
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih_i = oh_i * stride + kh - padding;
                    for (int64_t kw = 0; kw < kernel_w; ++kw, ++kk) {
                        const int64_t iw_i = ow_i * stride + kw - padding;
                        if (ih_i >= 0 && ih_i < ih && iw_i >= 0 && iw_i < iw) {
                            window[kk] = x[((b * in_channels + ic) * ih + ih_i) * iw + iw_i];
                        } else {
                            window[kk] = 0.0f;
                        }
                    }
                }
            }
            float value = dyop_dot(
                window.data(),
                codes + oc * packed.k,
                scales + oc * blocks,
                packed.k,
                packed.group_size
            );
            if (bias) value += bias[oc];
            y[index] = value;
        }
    });
    return output;
}

at::Tensor native_add_relu_cpu(const at::Tensor& a, const at::Tensor& b) {
    return at::relu(a + b);
}

at::Tensor native_add_cpu(const at::Tensor& a, const at::Tensor& b) {
    return a + b;
}

at::Tensor native_relu_cpu(const at::Tensor& a) {
    return at::relu(a);
}

at::Tensor native_max_pool2d_cpu(
    const at::Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    return at::max_pool2d(input, {kernel_size, kernel_size}, {stride, stride}, {padding, padding});
}

at::Tensor native_adaptive_avg_pool2d_cpu(
    const at::Tensor& input,
    int64_t output_size
) {
    return at::adaptive_avg_pool2d(input, {output_size, output_size});
}

void warm_native_cpu_workers() {
    native_worker_pool().warm(native_thread_count());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_native_cpu_weight", &pack_native_cpu_weight,
          "Pack signs/magnitude/exponents into native dyop signed-code format");
    m.def("dyadic_linear_packed_native_cpu", &dyadic_linear_packed_native_cpu,
          "Level 2 linear forward on packed dyop weight (CPU)");
    m.def("dyadic_embedding_packed_native_cpu", &dyadic_embedding_packed_native_cpu,
          "Level 2 embedding forward on packed dyop weight (CPU)");
    m.def("dyadic_qwen_mlp_packed_native_cpu", &dyadic_qwen_mlp_packed_native_cpu,
          "Bundled Qwen MLP forward on packed dyop weights (CPU)");
    m.def("dyadic_qwen_mlp_stack_packed_native_cpu", &dyadic_qwen_mlp_stack_packed_native_cpu,
          "Bundled stack of Qwen MLP forwards on packed dyop weights (CPU)");
    m.def("pack_qwen_mlp_stack_native_cpu", &pack_qwen_mlp_stack_native_cpu,
          "Create a reusable native Qwen MLP stack plan");
    m.def("dyadic_qwen_mlp_stack_plan_native_cpu", &dyadic_qwen_mlp_stack_plan_native_cpu,
          "Run a reusable native Qwen MLP stack plan");
    m.def("dyadic_conv2d_packed_native_cpu", &dyadic_conv2d_packed_native_cpu,
          "Level 2 conv2d forward on packed dyop weight (CPU)");
    m.def("native_add_relu_cpu", &native_add_relu_cpu,
          "Fused add + ReLU (CPU)");
    m.def("native_add_cpu", &native_add_cpu,
          "Elementwise add (CPU)");
    m.def("native_relu_cpu", &native_relu_cpu,
          "Elementwise ReLU (CPU)");
    m.def("native_max_pool2d_cpu", &native_max_pool2d_cpu,
          "MaxPool2d (CPU)");
    m.def("native_adaptive_avg_pool2d_cpu", &native_adaptive_avg_pool2d_cpu,
          "AdaptiveAvgPool2d (CPU)");
    m.def("warm_native_cpu_workers", &warm_native_cpu_workers,
          "Warm up native CPU worker threads");
}
