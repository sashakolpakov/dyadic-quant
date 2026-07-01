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

from dyadic_quant.level2.native import (
    build_native_cpu,
    native_add_cpu,
    native_add_relu_cpu,
    native_adaptive_avg_pool2d_cpu,
    native_max_pool2d_cpu,
    native_relu_cpu,
    warm_native_cpu_workers,
)


FIELDS = [
    "op",
    "shape",
    "dyop_threads",
    "torch_threads",
    "torch_ms",
    "native_ms",
    "speedup_vs_torch",
    "max_abs_error",
]


def time_call(fn, *, warmup: int, repeats: int) -> tuple[torch.Tensor, float]:
    output = fn()
    for _ in range(warmup):
        output = fn()
    start = perf_counter()
    for _ in range(repeats):
        output = fn()
    return output, (perf_counter() - start) * 1000 / repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dyop-threads", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/native_spatial_hotworkers.csv"),
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

    generator = torch.Generator().manual_seed(1001)
    rows: list[dict[str, object]] = []
    shapes = [
        ("resnet_relu", torch.randn(8, 64, 56, 56, generator=generator)),
        ("resnet_add", torch.randn(8, 64, 56, 56, generator=generator)),
        ("resnet_add_relu", torch.randn(8, 64, 56, 56, generator=generator)),
        ("resnet_maxpool", torch.randn(8, 64, 112, 112, generator=generator)),
        ("resnet_global_avgpool", torch.randn(8, 512, 7, 7, generator=generator)),
    ]
    for name, inputs in shapes:
        other = torch.randn(inputs.shape, generator=generator)
        if name == "resnet_relu":
            expected, torch_ms = time_call(
                lambda: F.relu(inputs),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            actual, native_ms = time_call(
                lambda: native_relu_cpu(inputs),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            op = "relu"
        elif name == "resnet_add":
            expected, torch_ms = time_call(
                lambda: inputs + other,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            actual, native_ms = time_call(
                lambda: native_add_cpu(inputs, other),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            op = "add"
        elif name == "resnet_add_relu":
            expected, torch_ms = time_call(
                lambda: F.relu(inputs + other),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            actual, native_ms = time_call(
                lambda: native_add_relu_cpu(inputs, other),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            op = "add_relu"
        elif name == "resnet_maxpool":
            expected, torch_ms = time_call(
                lambda: F.max_pool2d(inputs, kernel_size=3, stride=2, padding=1),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            actual, native_ms = time_call(
                lambda: native_max_pool2d_cpu(
                    inputs,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            op = "maxpool2d"
        else:
            expected, torch_ms = time_call(
                lambda: F.adaptive_avg_pool2d(inputs, (1, 1)),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            actual, native_ms = time_call(
                lambda: native_adaptive_avg_pool2d_cpu(inputs, (1, 1)),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            op = "adaptive_avgpool2d"
        rows.append(
            {
                "op": op,
                "shape": name,
                "dyop_threads": dyop_threads,
                "torch_threads": args.torch_threads,
                "torch_ms": torch_ms,
                "native_ms": native_ms,
                "speedup_vs_torch": torch_ms / native_ms,
                "max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['op']} {row['shape']}: torch={row['torch_ms']:.3f} ms "
            f"native={row['native_ms']:.3f} ms "
            f"speedup={row['speedup_vs_torch']:.3f}x "
            f"error={row['max_abs_error']:.6f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
