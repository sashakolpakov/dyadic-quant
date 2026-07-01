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
    pack_native_cpu_weight,
    warm_native_cpu_workers,
)


FIELDS = [
    "shape",
    "bits",
    "dyop_threads",
    "torch_threads",
    "torch_materialized_ms",
    "dyop_native_ms",
    "speedup_vs_torch",
    "max_abs_error",
]


SHAPES = [
    ("resnet_conv3x3", (8, 64, 56, 56), (64, 64, 3, 3), 1, 1),
    ("resnet_layer2_stride2_3x3", (1, 64, 56, 56), (128, 64, 3, 3), 2, 1),
    ("resnet_layer3_stride2_3x3", (1, 128, 28, 28), (256, 128, 3, 3), 2, 1),
    ("resnet_layer4_stride2_3x3", (1, 256, 14, 14), (512, 256, 3, 3), 2, 1),
    ("resnet_downsample", (8, 128, 28, 28), (256, 128, 1, 1), 2, 0),
]


def time_call(fn, *, warmup: int, repeats: int) -> tuple[torch.Tensor, float]:
    output = fn()
    for _ in range(warmup):
        output = fn()
    start = perf_counter()
    for _ in range(repeats):
        output = fn()
    return output, (perf_counter() - start) * 1000 / repeats


def benchmark_shape(
    name: str,
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    stride: int,
    padding: int,
    *,
    bits: int,
    repeats: int,
    warmup: int,
    dyop_threads: int,
    torch_threads: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(900 + weight_shape[0])
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

    expected, torch_ms = time_call(
        lambda: F.conv2d(inputs, decoded, bias=bias, stride=stride, padding=padding),
        warmup=warmup,
        repeats=repeats,
    )
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
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "shape": name,
        "bits": bits,
        "dyop_threads": dyop_threads,
        "torch_threads": torch_threads,
        "torch_materialized_ms": torch_ms,
        "dyop_native_ms": dyop_ms,
        "speedup_vs_torch": torch_ms / dyop_ms,
        "max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dyop-threads", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/native_conv2d_hotworkers.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dyop_threads is not None:
        os.environ["DYOP_CPU_THREADS"] = str(args.dyop_threads)
    dyop_threads = int(os.environ.get("DYOP_CPU_THREADS", "0") or "0")
    build_native_cpu()
    warm_native_cpu_workers()
    torch.set_num_threads(args.torch_threads)

    rows = [
        benchmark_shape(
            name,
            input_shape,
            weight_shape,
            stride,
            padding,
            bits=args.bits,
            repeats=args.repeats,
            warmup=args.warmup,
            dyop_threads=dyop_threads,
            torch_threads=args.torch_threads,
        )
        for name, input_shape, weight_shape, stride, padding in SHAPES
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['shape']}: torch={row['torch_materialized_ms']:.3f} ms "
            f"dyop={row['dyop_native_ms']:.3f} ms "
            f"speedup={row['speedup_vs_torch']:.3f}x "
            f"error={row['max_abs_error']:.6f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
