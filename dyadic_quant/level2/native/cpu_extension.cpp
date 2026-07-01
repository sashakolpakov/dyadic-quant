#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace py = pybind11;

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

template <typename Fn>
static void parallel_for_threads(int64_t begin, int64_t end, int threads, Fn fn) {
    native_worker_pool().parallel(begin, end, threads, fn);
}

struct PackedDyopWeight {
    at::Tensor codes;   // int16 [out, K], signed odd prefix code
    at::Tensor codes_knr;  // int16 [ceil(out/8), K, 8]
    at::Tensor codes_i8_knr;  // int8 [ceil(out/8), K, 8] when the prefix fits
    at::Tensor scales;  // float32 [out, blocks], dyadic step / 2 per group
    at::Tensor scales_padded;  // float32 [ceil(out/8)*8, blocks]
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
    if (packed.contains("codes_i8_knr")) {
        w.codes_i8_knr = packed["codes_i8_knr"].cast<at::Tensor>().contiguous();
    }
    w.scales = packed["scales"].cast<at::Tensor>().contiguous();
    if (packed.contains("scales_padded")) {
        w.scales_padded = packed["scales_padded"].cast<at::Tensor>().contiguous();
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
    auto codes_knr = at::zeros({out_padded / 8, k, 8}, signs.options().dtype(at::kShort));
    auto codes_i8_knr = at::empty({0}, signs.options().dtype(at::kChar));
    const bool use_i8_knr = bits <= 7;
    if (use_i8_knr) {
        codes_i8_knr = at::zeros({out_padded / 8, k, 8}, signs.options().dtype(at::kChar));
    }
    auto scales = at::empty({out, blocks}, signs.options().dtype(at::kFloat));
    auto scales_padded = at::zeros({out_padded, blocks}, signs.options().dtype(at::kFloat));

    const int8_t* sign_ptr = signs.data_ptr<int8_t>();
    const int32_t* mag_ptr = magnitude.data_ptr<int32_t>();
    const int16_t* exp_ptr = exponents.data_ptr<int16_t>();
    int16_t* code_ptr = codes.data_ptr<int16_t>();
    int16_t* knr_ptr = codes_knr.data_ptr<int16_t>();
    int8_t* knr_i8_ptr = use_i8_knr ? codes_i8_knr.data_ptr<int8_t>() : nullptr;
    float* scale_ptr = scales.data_ptr<float>();
    float* scale_pad_ptr = scales_padded.data_ptr<float>();

    at::parallel_for(0, out, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            for (int64_t b = 0; b < blocks; ++b) {
                const int16_t exponent = exp_ptr[n * blocks + b];
                const float scale = std::ldexp(1.0f, int(exponent + shift - 1));
                scale_ptr[n * blocks + b] = scale;
                scale_pad_ptr[n * blocks + b] = scale;
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
                if (knr_i8_ptr) {
                    knr_i8_ptr[knr_offset] = static_cast<int8_t>(code);
                }
            }
        }
    });

    py::dict packed;
    packed["codes"] = codes;
    packed["codes_knr"] = codes_knr;
    if (use_i8_knr) packed["codes_i8_knr"] = codes_i8_knr;
    packed["scales"] = scales;
    packed["scales_padded"] = scales_padded;
    packed["shape"] = shape;
    packed["group_size"] = group_size;
    packed["bits"] = bits;
    return packed;
}

at::Tensor dyadic_linear_packed_native_cpu(
    const at::Tensor& input_in,
    const py::dict& packed_dict,
    py::object bias_obj
) {
    auto input = input_in.to(at::kCPU).contiguous();
    TORCH_CHECK(input.scalar_type() == at::kFloat, "native linear input must be CPU float32");
    auto packed = unpack_weight(packed_dict);
    TORCH_CHECK(input.dim() == 2, "native linear input must be [M, K]");
    TORCH_CHECK(input.size(1) == packed.k, "native linear input width mismatch");

    auto output = at::empty({input.size(0), packed.out}, input.options());
    const float* x = input.data_ptr<float>();
    const int16_t* codes = packed.codes.data_ptr<int16_t>();
    const float* scales = packed.scales.data_ptr<float>();
    const float* bias = nullptr;
    at::Tensor bias_tensor;
    if (!bias_obj.is_none()) {
        bias_tensor = bias_obj.cast<at::Tensor>().to(at::kCPU).contiguous();
        TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias_tensor.numel() == packed.out, "bias width mismatch");
        bias = bias_tensor.data_ptr<float>();
    }
    float* y = output.data_ptr<float>();
    const int64_t m = input.size(0);
    const int64_t blocks = packed.scales.size(1);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (packed.group_size >= packed.k && packed.codes_knr.defined() && packed.scales_padded.defined()) {
        const float* scales_padded = packed.scales_padded.data_ptr<float>();
        auto run_knr8 = [&](auto codes_knr) {
            const int64_t mblocks = (m + 7) / 8;
            const int64_t nblocks = (packed.out + 7) / 8;
            parallel_for_threads(0, mblocks * nblocks, at::get_num_threads(), [&](int64_t begin, int64_t end) {
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
            parallel_for_threads(0, mblocks * nblocks, at::get_num_threads(), [&](int64_t begin, int64_t end) {
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
        if (packed.codes_i8_knr.defined()) {
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
    native_worker_pool().warm(at::get_num_threads());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_native_cpu_weight", &pack_native_cpu_weight,
          "Pack signs/magnitude/exponents into native dyop signed-code format");
    m.def("dyadic_linear_packed_native_cpu", &dyadic_linear_packed_native_cpu,
          "Level 2 linear forward on packed dyop weight (CPU)");
    m.def("dyadic_embedding_packed_native_cpu", &dyadic_embedding_packed_native_cpu,
          "Level 2 embedding forward on packed dyop weight (CPU)");
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
