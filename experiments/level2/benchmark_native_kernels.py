from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level1 import encode_tensor_per_output_channel
from dyadic_quant.level2.native import (
    build_native_cpu,
    dyadic_conv2d_packed_native_cpu,
    dyadic_embedding_packed_native_cpu,
    dyadic_linear_packed_native_cpu,
    pack_native_cpu_weight,
    warm_native_cpu_workers,
)


FIELDS = [
    "op",
    "shape",
    "bits",
    "torch_materialized_ms",
    "dyop_native_ms",
    "speedup_vs_torch",
    "max_abs_error",
]


def set_native_threads(threads: int) -> None:
    os.environ["DYOP_CPU_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def time_call(fn, *, warmup: int, repeats: int) -> tuple[torch.Tensor, float]:
    output = fn()
    for _ in range(warmup):
        output = fn()
    start = perf_counter()
    for _ in range(repeats):
        output = fn()
    elapsed_ms = (perf_counter() - start) * 1000 / repeats
    return output, elapsed_ms


def benchmark_linear(
    bits: int,
    repeats: int,
    *,
    torch_threads: int,
    dyop_threads: int,
    shapes_filter: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shapes = [
        ("gemv_qwen_mlp", 1, 896, 4864),
        ("gemm_qwen_seq", 64, 896, 896),
        ("output_projection_gemv", 1, 896, 151936),
        ("output_projection", 8, 896, 151936),
    ]
    for name, rows_in, input_width, output_width in shapes:
        if shapes_filter is not None and name not in shapes_filter:
            continue
        generator = torch.Generator().manual_seed(100 + rows_in + output_width)
        inputs = torch.randn(rows_in, input_width, generator=generator)
        weight = torch.randn(output_width, input_width, generator=generator)
        bias = torch.randn(output_width, generator=generator)
        encoded = encode_tensor_per_output_channel(weight, max_bits=8)
        decoded = encoded.decode(bits)
        packed = pack_native_cpu_weight(
            encoded.signs,
            encoded.magnitude_code,
            encoded.exponents,
            encoded.max_bits,
            encoded.group_size,
            bits,
        )
        torch.set_num_threads(torch_threads)
        expected, torch_ms = time_call(
            lambda: F.linear(inputs, decoded, bias),
            warmup=2,
            repeats=repeats,
        )
        set_native_threads(dyop_threads)
        warm_native_cpu_workers()
        actual, dyop_ms = time_call(
            lambda: dyadic_linear_packed_native_cpu(inputs, packed, bias),
            warmup=2,
            repeats=repeats,
        )
        rows.append(
            {
                "op": "linear",
                "shape": name,
                "bits": bits,
                "torch_materialized_ms": torch_ms,
                "dyop_native_ms": dyop_ms,
                "speedup_vs_torch": torch_ms / dyop_ms,
                "max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
            }
        )
    return rows


def benchmark_embedding(bits: int, repeats: int, *, torch_threads: int, dyop_threads: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(200)
    vocab, width = 151936, 896
    indices = torch.randint(0, vocab, (4, 64), generator=generator)
    weight = torch.randn(vocab, width, generator=generator)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8)
    decoded = encoded.decode(bits)
    packed = pack_native_cpu_weight(
        encoded.signs,
        encoded.magnitude_code,
        encoded.exponents,
        encoded.max_bits,
        encoded.group_size,
        bits,
    )
    torch.set_num_threads(torch_threads)
    expected, torch_ms = time_call(
        lambda: F.embedding(indices, decoded),
        warmup=2,
        repeats=repeats,
    )
    set_native_threads(dyop_threads)
    actual, dyop_ms = time_call(
        lambda: dyadic_embedding_packed_native_cpu(indices, packed),
        warmup=2,
        repeats=repeats,
    )
    return {
        "op": "embedding",
        "shape": "qwen_vocab_width",
        "bits": bits,
        "torch_materialized_ms": torch_ms,
        "dyop_native_ms": dyop_ms,
        "speedup_vs_torch": torch_ms / dyop_ms,
        "max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
    }


def benchmark_conv2d(bits: int, repeats: int, *, torch_threads: int, dyop_threads: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shapes = [
        ("resnet_conv3x3", (8, 64, 56, 56), (64, 64, 3, 3), 1, 1),
        ("resnet_downsample", (8, 128, 28, 28), (256, 128, 1, 1), 2, 0),
    ]
    for name, input_shape, weight_shape, stride, padding in shapes:
        generator = torch.Generator().manual_seed(300 + weight_shape[0])
        inputs = torch.randn(*input_shape, generator=generator)
        weight = torch.randn(*weight_shape, generator=generator)
        bias = torch.randn(weight_shape[0], generator=generator)
        encoded = encode_tensor_per_output_channel(weight, max_bits=8)
        decoded = encoded.decode(bits)
        packed = pack_native_cpu_weight(
            encoded.signs,
            encoded.magnitude_code,
            encoded.exponents,
            encoded.max_bits,
            encoded.group_size,
            bits,
        )
        torch.set_num_threads(torch_threads)
        expected, torch_ms = time_call(
            lambda: F.conv2d(inputs, decoded, bias=bias, stride=stride, padding=padding),
            warmup=1,
            repeats=repeats,
        )
        set_native_threads(dyop_threads)
        actual, dyop_ms = time_call(
            lambda: dyadic_conv2d_packed_native_cpu(
                inputs,
                packed,
                bias,
                stride,
                padding,
                weight_shape[2],
                weight_shape[3],
            ),
            warmup=1,
            repeats=repeats,
        )
        rows.append(
            {
                "op": "conv2d",
                "shape": name,
                "bits": bits,
                "torch_materialized_ms": torch_ms,
                "dyop_native_ms": dyop_ms,
                "speedup_vs_torch": torch_ms / dyop_ms,
                "max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dyop-threads", type=int, default=None)
    parser.add_argument(
        "--embedding-dyop-threads",
        type=int,
        default=None,
        help="Native worker threads for embedding; defaults to --dyop-threads.",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=("linear", "embedding", "conv2d"),
        default=["linear", "embedding", "conv2d"],
    )
    parser.add_argument(
        "--linear-shapes",
        nargs="+",
        choices=(
            "gemv_qwen_mlp",
            "gemm_qwen_seq",
            "output_projection_gemv",
            "output_projection",
        ),
        help="Restrict --ops linear to selected benchmark shapes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/native_kernel_benchmark.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_native_cpu()
    dyop_threads = args.dyop_threads or args.torch_threads
    embedding_dyop_threads = args.embedding_dyop_threads or dyop_threads
    rows: list[dict[str, object]] = []
    if "linear" in args.ops:
        rows.extend(
            benchmark_linear(
                args.bits,
                args.repeats,
                torch_threads=args.torch_threads,
                dyop_threads=dyop_threads,
                shapes_filter=set(args.linear_shapes) if args.linear_shapes else None,
            )
        )
    if "embedding" in args.ops:
        rows.append(
            benchmark_embedding(
                args.bits,
                args.repeats,
                torch_threads=args.torch_threads,
                dyop_threads=embedding_dyop_threads,
            )
        )
    if "conv2d" in args.ops:
        rows.extend(
            benchmark_conv2d(
                args.bits,
                args.repeats,
                torch_threads=args.torch_threads,
                dyop_threads=dyop_threads,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['op']} {row['shape']}: "
            f"torch={row['torch_materialized_ms']:.3f} ms "
            f"dyop={row['dyop_native_ms']:.3f} ms "
            f"speedup={row['speedup_vs_torch']:.3f}x"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
