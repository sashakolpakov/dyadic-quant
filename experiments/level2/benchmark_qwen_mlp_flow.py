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
    dyadic_linear_packed_native_cpu,
    dyadic_qwen_mlp_packed_native_cpu,
    dyadic_qwen_mlp_stack_plan_native_cpu,
    dyadic_qwen_mlp_stack_packed_native_cpu,
    pack_native_cpu_weight,
    pack_qwen_mlp_stack_native_cpu,
    warm_native_cpu_workers,
)


def time_call(fn, *, warmup: int, repeats: int) -> tuple[torch.Tensor, float]:
    output = fn()
    for _ in range(warmup):
        output = fn()
    start = perf_counter()
    for _ in range(repeats):
        output = fn()
    return output, (perf_counter() - start) * 1000.0 / repeats


def pack_weight(weight: torch.Tensor, bits: int) -> tuple[dict[str, object], torch.Tensor]:
    encoded = encode_tensor_per_output_channel(weight, max_bits=8)
    packed = pack_native_cpu_weight(
        encoded.signs,
        encoded.magnitude_code,
        encoded.exponents,
        encoded.max_bits,
        encoded.group_size,
        bits,
    )
    return packed, encoded.decode(bits)


def benchmark_shape(
    *,
    name: str,
    rows: int,
    width: int,
    hidden: int,
    depth: int,
    bits: int,
    repeats: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(500 + rows + width + hidden)
    inputs = torch.randn(rows, width, generator=generator)
    blocks = []
    decoded_blocks = []
    for _ in range(depth):
        gate_w = torch.randn(hidden, width, generator=generator) / (width**0.5)
        up_w = torch.randn(hidden, width, generator=generator) / (width**0.5)
        down_w = torch.randn(width, hidden, generator=generator) / (hidden**0.5)
        gate_b = torch.randn(hidden, generator=generator) / 100
        up_b = torch.randn(hidden, generator=generator) / 100
        down_b = torch.randn(width, generator=generator) / 100
        gate_packed, gate_decoded = pack_weight(gate_w, bits)
        up_packed, up_decoded = pack_weight(up_w, bits)
        down_packed, down_decoded = pack_weight(down_w, bits)
        blocks.append((gate_packed, up_packed, down_packed, gate_b, up_b, down_b))
        decoded_blocks.append(
            (gate_decoded, up_decoded, down_decoded, gate_b, up_b, down_b)
        )

    def torch_flow() -> torch.Tensor:
        current = inputs
        for gate_w, up_w, down_w, gate_b, up_b, down_b in decoded_blocks:
            current = F.linear(
                F.silu(F.linear(current, gate_w, gate_b))
                * F.linear(current, up_w, up_b),
                down_w,
                down_b,
            )
        return current

    def disjoint_flow() -> torch.Tensor:
        current = inputs
        for gate_packed, up_packed, down_packed, gate_b, up_b, down_b in blocks:
            current = dyadic_linear_packed_native_cpu(
                F.silu(dyadic_linear_packed_native_cpu(current, gate_packed, gate_b))
                * dyadic_linear_packed_native_cpu(current, up_packed, up_b),
                down_packed,
                down_b,
            )
        return current

    def bundled_flow() -> torch.Tensor:
        if depth == 1:
            gate_packed, up_packed, down_packed, gate_b, up_b, down_b = blocks[0]
            return dyadic_qwen_mlp_packed_native_cpu(
                inputs,
                gate_packed,
                up_packed,
                down_packed,
                gate_b,
                up_b,
                down_b,
            )
        return dyadic_qwen_mlp_stack_packed_native_cpu(inputs, blocks)

    plan = pack_qwen_mlp_stack_native_cpu(blocks)

    def planned_flow() -> torch.Tensor:
        return dyadic_qwen_mlp_stack_plan_native_cpu(inputs, plan)

    expected, torch_ms = time_call(
        torch_flow,
        warmup=2,
        repeats=repeats,
    )
    warm_native_cpu_workers()
    disjoint, disjoint_ms = time_call(
        disjoint_flow,
        warmup=2,
        repeats=repeats,
    )
    bundled, bundled_ms = time_call(
        bundled_flow,
        warmup=2,
        repeats=repeats,
    )
    planned, planned_ms = time_call(
        planned_flow,
        warmup=2,
        repeats=repeats,
    )
    return {
        "shape": name,
        "bits": bits,
        "rows": rows,
        "width": width,
        "hidden": hidden,
        "depth": depth,
        "torch_materialized_ms": torch_ms,
        "dyop_disjoint_ms": disjoint_ms,
        "dyop_bundled_ms": bundled_ms,
        "dyop_planned_ms": planned_ms,
        "bundled_speedup_vs_disjoint": disjoint_ms / bundled_ms,
        "planned_speedup_vs_disjoint": disjoint_ms / planned_ms,
        "disjoint_speedup_vs_torch": torch_ms / disjoint_ms,
        "bundled_speedup_vs_torch": torch_ms / bundled_ms,
        "planned_speedup_vs_torch": torch_ms / planned_ms,
        "disjoint_max_abs_error": float((disjoint - expected).abs().max().item()),
        "bundled_max_abs_error": float((bundled - expected).abs().max().item()),
        "planned_max_abs_error": float((planned - expected).abs().max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/qwen_mlp_flow_benchmark.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        os.environ["DYOP_CPU_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)
    build_native_cpu(force=True)
    shapes = []
    for depth in (1, 2, 4):
        shapes.extend(
            [
                (f"qwen_mlp_gemv_d{depth}", 1, 896, 4864, depth),
                (f"qwen_mlp_seq8_d{depth}", 8, 896, 4864, depth),
                (f"qwen_mlp_seq64_d{depth}", 64, 896, 4864, depth),
            ]
        )
    shapes.append(("qwen_mlp_tiny_token_d8", 1, 256, 768, 8))
    shapes.append(("qwen_mlp_tiny_token_d24", 1, 128, 384, 24))
    rows = [
        benchmark_shape(
            name=name,
            rows=row_count,
            width=width,
            hidden=hidden,
            depth=depth,
            bits=args.bits,
            repeats=args.repeats,
        )
        for name, row_count, width, hidden, depth in shapes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['shape']}: disjoint={row['dyop_disjoint_ms']:.3f} ms "
            f"bundled={row['dyop_bundled_ms']:.3f} ms "
            f"planned={row['dyop_planned_ms']:.3f} ms "
            f"planned_speedup={row['planned_speedup_vs_disjoint']:.3f}x"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
