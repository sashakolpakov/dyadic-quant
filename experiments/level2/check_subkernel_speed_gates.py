from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "subkernel",
    "shape",
    "source",
    "materialized_ms",
    "dyop_native_ms",
    "speedup_vs_materialized",
    "passes_speed_gate",
    "max_abs_error",
]

REQUIRED_GATE_SETS = {
    "qwen": {
        "linear_gemm_qwen_seq",
        "linear_output_projection",
        "embedding_qwen_vocab_width",
    },
    "resnet": {
        "resnet_conv3x3",
        "resnet_layer2_stride2_3x3",
        "resnet_layer3_stride2_3x3",
        "resnet_layer4_stride2_3x3",
        "resnet_downsample",
        "adaptive_avgpool2d_resnet_global",
    },
}
REQUIRED_GATE_SETS["all"] = REQUIRED_GATE_SETS["qwen"] | REQUIRED_GATE_SETS["resnet"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def add_kernel_rows(
    summary: list[dict[str, object]],
    *,
    source: Path,
    rows: list[dict[str, str]],
    op_filter: set[str] | None = None,
) -> None:
    for row in rows:
        op = row.get("op") or row.get("shape") or "unknown"
        if op_filter is not None and op not in op_filter:
            continue
        materialized = row.get("torch_materialized_ms") or row.get("torch_ms")
        native = row.get("dyop_native_ms") or row.get("native_ms")
        speedup = row.get("speedup_vs_torch")
        if materialized is None or native is None or speedup is None:
            raise RuntimeError(f"unsupported row schema in {source}: {row}")
        speedup_value = float(speedup)
        summary.append(
            {
                "subkernel": op,
                "shape": row.get("shape", ""),
                "source": str(source),
                "materialized_ms": float(materialized),
                "dyop_native_ms": float(native),
                "speedup_vs_materialized": speedup_value,
                "passes_speed_gate": speedup_value >= 1.0,
                "max_abs_error": row.get("max_abs_error", ""),
            }
        )


def add_fixed_gate_rows(
    summary: list[dict[str, object]],
    *,
    source: Path,
    rows: list[dict[str, str]],
) -> None:
    for row in rows:
        subkernel = row["subkernel"]
        materialized = float(row["materialized_gate_ms"])
        native = float(row.get("arm64_neon_ms") or row.get("x86_native_ms"))
        speedup = float(row.get("speedup_vs_arm64_gate") or row.get("speedup_vs_arm_gate"))
        summary.append(
            {
                "subkernel": subkernel,
                "shape": subkernel,
                "source": str(source),
                "materialized_ms": materialized,
                "dyop_native_ms": native,
                "speedup_vs_materialized": speedup,
                "passes_speed_gate": row["passes_fixed_gate"].lower() == "true",
                "max_abs_error": "" if row.get("correct", "true").lower() == "true" else "incorrect",
            }
        )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Level 2 dyop subkernel speed gates against materialized tensor baselines."
    )
    parser.add_argument(
        "--native-kernels",
        type=Path,
        default=Path("results/level2/native_kernels_after_neon_dot_bits6_r50.csv"),
    )
    parser.add_argument(
        "--conv2d",
        type=Path,
        default=Path("results/level2/native_conv2d_stride2_oc8_dispatch_threads12_bits6_r40.csv"),
    )
    parser.add_argument(
        "--spatial",
        type=Path,
        default=Path("results/level2/native_spatial_addrelu_hotworkers_threads16.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/subkernel_speed_gates.csv"),
    )
    parser.add_argument(
        "--fixed-gates",
        type=Path,
        help="Read a fixed-gate CSV from dyop_primitives_{arch}.cpp instead of benchmark CSVs.",
    )
    parser.add_argument(
        "--require",
        choices=tuple(REQUIRED_GATE_SETS),
        help="Require the named Level 2 gate set to pass.",
    )
    parser.add_argument("--fail-on-miss", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: list[dict[str, object]] = []
    if args.fixed_gates is not None:
        add_fixed_gate_rows(
            summary,
            source=args.fixed_gates,
            rows=read_rows(args.fixed_gates),
        )
    else:
        add_kernel_rows(
            summary,
            source=args.native_kernels,
            rows=read_rows(args.native_kernels),
            op_filter={"linear", "embedding"},
        )
        add_kernel_rows(
            summary,
            source=args.conv2d,
            rows=read_rows(args.conv2d),
        )
        add_kernel_rows(
            summary,
            source=args.spatial,
            rows=read_rows(args.spatial),
        )
    write_summary(args.output, summary)
    required = REQUIRED_GATE_SETS.get(args.require, set())
    failures = [
        row
        for row in summary
        if not row["passes_speed_gate"] and (not required or row["subkernel"] in required)
    ]
    missing = sorted(required - {str(row["subkernel"]) for row in summary})
    for row in summary:
        status = "PASS" if row["passes_speed_gate"] else "FAIL"
        print(
            f"{status} {row['subkernel']}:{row['shape']} "
            f"speedup={float(row['speedup_vs_materialized']):.3f}x"
        )
    print(f"Wrote {args.output}")
    if missing:
        for subkernel in missing:
            print(f"MISSING required gate: {subkernel}")
    if (failures or missing) and args.fail_on_miss:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
