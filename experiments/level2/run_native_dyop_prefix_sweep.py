from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.dyadic_torch import (
    encode_model,
    load_encoded_model,
    materialize_prefix,
    save_encoded_model,
    storage_bytes,
)
from dyadic_quant.level2 import build_level2_model, build_native_cpu

from experiments.level2.common import TinyDyadicNet, timed_forward, tiny_inputs


CSV_FIELDS = [
    "method",
    "execution_backend",
    "linear_backend",
    "embedding_backend",
    "conv_backend",
    "bits_per_weight",
    "conversion_ms",
    "materialization_ms",
    "level2_load_ms",
    "level2_build_ms",
    "level1_forward_ms",
    "level2_forward_ms",
    "max_abs_error_vs_level1_materialized",
    "allclose_vs_level1_materialized",
    "total_model_bytes",
    "incremental_plane_bytes",
    "weight_payload_bytes",
    "exponent_bytes",
    "other_model_bytes",
    "level2_replaced_modules",
    "level2_shared_weight_modules",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-prefix Level 2 native dyop validation from a packed "
            "Level 1 dyadic artifact."
        )
    )
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--linear-backend",
        choices=["scalar", "native-cpu"],
        default="scalar",
        help=(
            "Level 2 backend for Linear/GEMV/GEMM/output projection. "
            "native-cpu is CPU float32 only and is built explicitly."
        ),
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["scalar", "native-cpu"],
        default="scalar",
        help=(
            "Level 2 backend for embedding lookup. native-cpu is CPU int64 "
            "indices to float32 output and is built explicitly."
        ),
    )
    parser.add_argument(
        "--conv-backend",
        choices=["scalar", "native-cpu"],
        default="scalar",
        help=(
            "Level 2 backend for Conv2d. native-cpu is CPU float32 NCHW and "
            "is built explicitly."
        ),
    )
    parser.add_argument(
        "--variant-name",
        default="native_dyop_prefix_sweep",
        help="Safe prefix for CSV, metadata, and packed artifact filenames.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/level2"))
    return parser.parse_args()


def validate_prefix_bits(bits: list[int], max_bits: int) -> tuple[int, ...]:
    if max_bits < 2 or max_bits > 8:
        raise ValueError("max_bits must be in [2, 8] for packed Level 2 sweeps")
    unique = tuple(sorted(set(bits)))
    if not unique:
        raise ValueError("at least one prefix width is required")
    if any(bit < 2 or bit > max_bits for bit in unique):
        raise ValueError(f"all prefix widths must be in [2, {max_bits}]")
    return unique


def prefix_row(
    *,
    bits: int,
    linear_backend: str,
    embedding_backend: str,
    conv_backend: str,
    conversion_ms: float,
    materialization_ms: float,
    level2_load_ms: float,
    level2_build_ms: float,
    level1_forward_ms: float,
    level2_forward_ms: float,
    max_abs_error: float,
    allclose: bool,
    sizes: dict[str, int],
    replaced_modules: tuple[str, ...],
    shared_weight_modules: tuple[str, ...],
) -> dict[str, object]:
    return {
        "method": "level2_native_dyop",
        "execution_backend": "level2-native",
        "linear_backend": linear_backend,
        "embedding_backend": embedding_backend,
        "conv_backend": conv_backend,
        "bits_per_weight": bits,
        "conversion_ms": conversion_ms,
        "materialization_ms": materialization_ms,
        "level2_load_ms": level2_load_ms,
        "level2_build_ms": level2_build_ms,
        "level1_forward_ms": level1_forward_ms,
        "level2_forward_ms": level2_forward_ms,
        "max_abs_error_vs_level1_materialized": max_abs_error,
        "allclose_vs_level1_materialized": allclose,
        "total_model_bytes": sizes["total_model_bytes"],
        "incremental_plane_bytes": sizes["incremental_plane_bytes"],
        "weight_payload_bytes": sizes["weight_payload_bytes"],
        "exponent_bytes": sizes["exponent_bytes"],
        "other_model_bytes": sizes["other_model_bytes"],
        "level2_replaced_modules": ",".join(replaced_modules),
        "level2_shared_weight_modules": ",".join(shared_weight_modules),
    }


def main() -> None:
    args = parse_args()
    prefix_bits = validate_prefix_bits(args.bits, args.max_bits)
    if (
        args.linear_backend == "native-cpu"
        or args.embedding_backend == "native-cpu"
        or args.conv_backend == "native-cpu"
    ):
        build_native_cpu()
    torch.manual_seed(args.seed)
    source = TinyDyadicNet().eval()
    tokens, image = tiny_inputs(args.seed + 1)
    group_size = args.group_size or None

    encoded = encode_model(
        source,
        max_bits=args.max_bits,
        optimize_prefix_bits=prefix_bits,
        group_size=group_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packed_artifact = args.output_dir / f"{args.variant_name}.dyadic.pt"
    save_encoded_model(encoded, packed_artifact)

    rows: list[dict[str, object]] = []
    load_times_ms: list[float] = []
    for bits in prefix_bits:
        level1 = copy.deepcopy(source).eval()
        materialization_start = perf_counter()
        materialization_ms = materialize_prefix(level1, encoded, bits=bits)
        materialization_total_ms = (perf_counter() - materialization_start) * 1000
        level1_output, level1_ms = timed_forward(
            level1, tokens, image, args.repeats
        )

        load_start = perf_counter()
        loaded = load_encoded_model(packed_artifact)
        load_ms = (perf_counter() - load_start) * 1000
        load_times_ms.append(load_ms)
        build_start = perf_counter()
        level2, replacement = build_level2_model(
            source,
            loaded,
            bits=bits,
            linear_backend=args.linear_backend,
            embedding_backend=args.embedding_backend,
            conv_backend=args.conv_backend,
        )
        level2_build_ms = (perf_counter() - build_start) * 1000
        level2.eval()
        level2_output, level2_ms = timed_forward(
            level2, tokens, image, args.repeats
        )

        max_abs_error = float(
            torch.max(torch.abs(level2_output - level1_output)).item()
        )
        rows.append(
            prefix_row(
                bits=bits,
                linear_backend=args.linear_backend,
                embedding_backend=args.embedding_backend,
                conv_backend=args.conv_backend,
                conversion_ms=encoded.conversion_ms,
                materialization_ms=materialization_total_ms,
                level2_load_ms=load_ms,
                level2_build_ms=level2_build_ms,
                level1_forward_ms=level1_ms,
                level2_forward_ms=level2_ms,
                max_abs_error=max_abs_error,
                allclose=bool(
                    torch.allclose(level2_output, level1_output, rtol=1e-6, atol=1e-6)
                ),
                sizes=storage_bytes(source, encoded, bits=bits),
                replaced_modules=replacement.replaced_modules,
                shared_weight_modules=replacement.shared_weight_modules,
            )
        )

    result_path = args.output_dir / f"{args.variant_name}_results.csv"
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "level": 2,
        "variant_name": args.variant_name,
        "bits": list(prefix_bits),
        "max_bits": args.max_bits,
        "group_size": args.group_size,
        "seed": args.seed,
        "repeats": args.repeats,
        "linear_backend": args.linear_backend,
        "embedding_backend": args.embedding_backend,
        "conv_backend": args.conv_backend,
        "packed_artifact": str(packed_artifact),
        "result_path": str(result_path),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "quantized_weight_count": encoded.quantized_weight_count,
        "exponent_count": encoded.exponent_count,
        "mean_level2_load_ms": sum(load_times_ms) / len(load_times_ms),
        "all_prefixes_match_level1_materialized": all(
            bool(row["allclose_vs_level1_materialized"]) for row in rows
        ),
        "note": (
            "Each Level 2 prefix row reloads the packed dyadic artifact and "
            "executes encoded weights through native dyop modules. Level 1 "
            "materialization is used only as the comparison baseline."
        ),
    }
    metadata_path = args.output_dir / f"{args.variant_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
