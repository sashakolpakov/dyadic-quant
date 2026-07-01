from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level1 import encode_model, materialize_prefix, storage_bytes
from dyadic_quant.level2 import build_level2_model, build_native_cpu, warm_native_cpu_workers
from experiments.level1.run_resnet18_dyadic import (
    fuse_resnet_batch_norm,
    replace_resnet_basic_blocks_with_native_residuals,
)


def mps_available() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def time_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> float:
    sample = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        sync_device(device)
        start = perf_counter()
        for _ in range(repeats):
            model(sample)
        sync_device(device)
    return (perf_counter() - start) * 1000.0 / repeats


def model_tensor_bytes(model: torch.nn.Module, dtype_bytes: int | None = None) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * (dtype_bytes if dtype_bytes is not None else tensor.element_size())
    return total


def load_fused_resnet(checkpoint: Path) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return fuse_resnet_batch_norm(model).eval()


def build_encoded_model(
    base_model: torch.nn.Module,
    *,
    bits: int,
    quantize_endpoints: bool,
) -> Any:
    return encode_model(
        base_model,
        max_bits=bits,
        optimize_prefix_bits=(bits,),
        exclude_names=set() if quantize_endpoints else {"conv1", "fc"},
    )


def native_worker_row(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["DYOP_CPU_THREADS"] = str(args.worker_threads)
    torch.set_num_threads(args.torch_threads)
    build_native_cpu()
    warm_native_cpu_workers()
    base_model = load_fused_resnet(args.checkpoint)
    encoded = build_encoded_model(
        base_model,
        bits=args.bits,
        quantize_endpoints=args.quantize_endpoints,
    )
    start = perf_counter()
    model, replacement = build_level2_model(
        base_model,
        encoded,
        bits=args.bits,
        dtype=torch.float32,
        linear_backend="native-cpu",
        conv_backend="native-cpu",
        spatial_backend="native-cpu",
    )
    native_blocks = replace_resnet_basic_blocks_with_native_residuals(model)
    build_ms = (perf_counter() - start) * 1000.0
    model = model.to(device=torch.device("cpu"), dtype=torch.float32).eval()
    latency_ms = time_model(
        model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    sizes = storage_bytes(base_model, encoded, bits=args.bits)
    return {
        "backend": "level2_native_dyop_cpu",
        "available": True,
        "bits": args.bits,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": args.torch_threads,
        "dyop_threads": args.worker_threads,
        "batch_size": args.batch_size,
        "latency_ms": latency_ms,
        "images_per_s": args.batch_size * 1000.0 / latency_ms,
        "conversion_ms": encoded.conversion_ms,
        "materialization_ms": 0.0,
        "level2_build_ms": build_ms,
        "total_model_bytes": sizes["total_model_bytes"],
        "incremental_plane_bytes": sizes["incremental_plane_bytes"],
        "tensor_bytes": "",
        "replaced_modules": ",".join(replacement.replaced_modules),
        "native_residual_blocks": ",".join(native_blocks),
        "note": "",
    }


def materialized_row(
    *,
    base_model: torch.nn.Module,
    encoded: Any,
    args: argparse.Namespace,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = load_fused_resnet(args.checkpoint)
    materialization_ms = materialize_prefix(model, encoded, bits=args.bits)
    model = model.to(device=device, dtype=dtype).eval()
    latency_ms = time_model(
        model,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    sizes = storage_bytes(base_model, encoded, bits=args.bits)
    dtype_name = "float16" if dtype == torch.float16 else "float32"
    dtype_bytes = 2 if dtype == torch.float16 else 4
    return {
        "backend": backend,
        "available": True,
        "bits": args.bits,
        "device": device.type,
        "dtype": dtype_name,
        "torch_threads": args.torch_threads,
        "dyop_threads": "",
        "batch_size": args.batch_size,
        "latency_ms": latency_ms,
        "images_per_s": args.batch_size * 1000.0 / latency_ms,
        "conversion_ms": encoded.conversion_ms,
        "materialization_ms": materialization_ms,
        "level2_build_ms": 0.0,
        "total_model_bytes": sizes["total_model_bytes"],
        "incremental_plane_bytes": sizes["incremental_plane_bytes"],
        "tensor_bytes": model_tensor_bytes(model, dtype_bytes=dtype_bytes),
        "replaced_modules": "",
        "native_residual_blocks": "",
        "note": "",
    }


def unavailable_mps_row(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "materialized_dyadic_mps",
        "available": False,
        "bits": args.bits,
        "device": "mps",
        "dtype": "float16",
        "torch_threads": args.torch_threads,
        "dyop_threads": "",
        "batch_size": args.batch_size,
        "latency_ms": "",
        "images_per_s": "",
        "conversion_ms": "",
        "materialization_ms": "",
        "level2_build_ms": "",
        "total_model_bytes": "",
        "incremental_plane_bytes": "",
        "tensor_bytes": "",
        "replaced_modules": "",
        "native_residual_blocks": "",
        "note": "MPS is not available in this PyTorch runtime",
    }


def run_native_subprocess(args: argparse.Namespace, threads: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--bits",
        str(args.bits),
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--torch-threads",
        str(args.torch_threads),
        "--worker-threads",
        str(threads),
    ]
    if args.quantize_endpoints:
        command.append("--quantize-endpoints")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def add_speedup_columns(rows: list[dict[str, Any]]) -> None:
    cpu_rows = [row for row in rows if row["backend"] == "materialized_dyadic_cpu"]
    mps_rows = [
        row
        for row in rows
        if row["backend"] == "materialized_dyadic_mps" and row["available"]
    ]
    cpu_latency = float(cpu_rows[0]["latency_ms"]) if cpu_rows else None
    mps_latency = float(mps_rows[0]["latency_ms"]) if mps_rows else None
    for row in rows:
        latency = row["latency_ms"]
        if row["available"] and latency != "":
            latency_f = float(latency)
            row["speedup_vs_materialized_cpu"] = (
                cpu_latency / latency_f if cpu_latency is not None else ""
            )
            row["speedup_vs_materialized_mps"] = (
                mps_latency / latency_f if mps_latency is not None else ""
            )
            row["passes_cpu_speed_gate"] = (
                row["speedup_vs_materialized_cpu"] >= 1.0
                if row["speedup_vs_materialized_cpu"] != ""
                else ""
            )
            row["passes_mps_speed_gate"] = (
                row["speedup_vs_materialized_mps"] >= 1.0
                if row["speedup_vs_materialized_mps"] != ""
                else ""
            )
        else:
            row["speedup_vs_materialized_cpu"] = ""
            row["speedup_vs_materialized_mps"] = ""
            row["passes_cpu_speed_gate"] = ""
            row["passes_mps_speed_gate"] = ""


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "available",
        "bits",
        "device",
        "dtype",
        "torch_threads",
        "dyop_threads",
        "batch_size",
        "latency_ms",
        "images_per_s",
        "speedup_vs_materialized_cpu",
        "speedup_vs_materialized_mps",
        "passes_cpu_speed_gate",
        "passes_mps_speed_gate",
        "conversion_ms",
        "materialization_ms",
        "level2_build_ms",
        "total_model_bytes",
        "incremental_plane_bytes",
        "tensor_bytes",
        "replaced_modules",
        "native_residual_blocks",
        "note",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Level 2 ResNet speed gate: native dyops vs materialized tensors."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--dyop-threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 12, 16],
        help="Worker counts to sweep. Each count runs in a fresh process.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/level2/resnet_speed_gate.csv"))
    parser.add_argument("--quantize-endpoints", action="store_true")
    parser.add_argument("--worker-threads", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise RuntimeError("--repeats must be positive")
    if args.worker_threads is not None:
        print(json.dumps(native_worker_row(args), sort_keys=True))
        return

    torch.set_num_threads(args.torch_threads)
    base_model = load_fused_resnet(args.checkpoint)
    encoded = build_encoded_model(
        base_model,
        bits=args.bits,
        quantize_endpoints=args.quantize_endpoints,
    )
    rows = [
        materialized_row(
            base_model=base_model,
            encoded=encoded,
            args=args,
            backend="materialized_dyadic_cpu",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    ]
    if mps_available():
        rows.append(
            materialized_row(
                base_model=base_model,
                encoded=encoded,
                args=args,
                backend="materialized_dyadic_mps",
                device=torch.device("mps"),
                dtype=torch.float16,
            )
        )
    else:
        rows.append(unavailable_mps_row(args))

    for threads in args.dyop_threads:
        rows.append(run_native_subprocess(args, threads))

    add_speedup_columns(rows)
    write_rows(args.output, rows)
    print(args.output)
    for row in rows:
        latency = row["latency_ms"]
        if latency == "":
            print(f"{row['backend']} unavailable: {row['note']}")
        else:
            print(
                f"{row['backend']} threads={row['dyop_threads'] or '-'} "
                f"latency_ms={float(latency):.3f} "
                f"speedup_cpu={row['speedup_vs_materialized_cpu']}"
            )


if __name__ == "__main__":
    main()
