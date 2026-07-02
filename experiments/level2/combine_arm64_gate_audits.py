from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SUBKERNELS = [
    "linear_gemm_qwen_seq",
    "linear_output_projection",
    "embedding_qwen_vocab_width",
    "resnet_conv3x3",
    "resnet_layer2_stride2_3x3",
    "resnet_layer3_stride2_3x3",
    "resnet_layer4_stride2_3x3",
    "resnet_downsample",
    "adaptive_avgpool2d_resnet_global",
]


@dataclass(frozen=True)
class GateRow:
    backend: str
    subkernel: str
    gate_ms: float
    native_ms: float
    speedup: float
    threads: str
    correct: bool
    op_tree: str

    @property
    def passes(self) -> bool:
        return self.correct and self.native_ms < self.gate_ms


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def read_backend_rows(path: Path, backend: str) -> dict[str, GateRow]:
    rows: dict[str, GateRow] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            native = (
                row.get("arm64_native_ms")
                or row.get("arm64_neon_ms")
                or row.get("arm64_amx_ms")
            )
            if native is None:
                raise RuntimeError(f"{path} has no ARM64 native timing column")
            subkernel = row["subkernel"]
            gate_ms = float(row["materialized_gate_ms"])
            native_ms = float(native)
            rows[subkernel] = GateRow(
                backend=backend,
                subkernel=subkernel,
                gate_ms=gate_ms,
                native_ms=native_ms,
                speedup=gate_ms / native_ms,
                threads=row.get("best_threads", ""),
                correct=parse_bool(row.get("correct", "false")),
                op_tree=row.get("op_tree", ""),
            )
    return rows


def choose_row(candidates: list[GateRow]) -> GateRow | None:
    correct = [row for row in candidates if row.correct]
    if not correct:
        return None
    return min(correct, key=lambda row: row.native_ms)


def write_combined(
    output: Path,
    detailed_output: Path,
    neon_rows: dict[str, GateRow],
    amx_rows: dict[str, GateRow],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    detailed_output.parent.mkdir(parents=True, exist_ok=True)

    fixed_fields = [
        "subkernel",
        "materialized_gate_ms",
        "arm64_native_ms",
        "speedup_vs_arm64_gate",
        "best_threads",
        "passes_fixed_gate",
        "correct",
        "selected_backend",
        "op_tree",
    ]
    detail_fields = [
        *fixed_fields,
        "neon_ms",
        "neon_speedup",
        "neon_passes",
        "amx_ms",
        "amx_speedup",
        "amx_passes",
    ]

    with output.open("w", newline="") as fixed_handle, detailed_output.open("w", newline="") as detail_handle:
        fixed_writer = csv.DictWriter(fixed_handle, fieldnames=fixed_fields)
        detail_writer = csv.DictWriter(detail_handle, fieldnames=detail_fields)
        fixed_writer.writeheader()
        detail_writer.writeheader()

        for subkernel in REQUIRED_SUBKERNELS:
            candidates = [
                row
                for row in (neon_rows.get(subkernel), amx_rows.get(subkernel))
                if row is not None
            ]
            selected = choose_row(candidates)
            if selected is None:
                raise RuntimeError(f"no correct backend row for {subkernel}")

            fixed_row = {
                "subkernel": subkernel,
                "materialized_gate_ms": f"{selected.gate_ms:.6f}",
                "arm64_native_ms": f"{selected.native_ms:.6f}",
                "speedup_vs_arm64_gate": f"{selected.speedup:.6f}",
                "best_threads": selected.threads,
                "passes_fixed_gate": str(selected.passes).lower(),
                "correct": str(selected.correct).lower(),
                "selected_backend": selected.backend,
                "op_tree": selected.op_tree,
            }
            detail_row = {
                **fixed_row,
                **backend_detail("neon", neon_rows.get(subkernel)),
                **backend_detail("amx", amx_rows.get(subkernel)),
            }
            fixed_writer.writerow(fixed_row)
            detail_writer.writerow(detail_row)


def backend_detail(prefix: str, row: GateRow | None) -> dict[str, str]:
    if row is None:
        return {
            f"{prefix}_ms": "",
            f"{prefix}_speedup": "",
            f"{prefix}_passes": "false",
        }
    return {
        f"{prefix}_ms": f"{row.native_ms:.6f}",
        f"{prefix}_speedup": f"{row.speedup:.6f}",
        f"{prefix}_passes": str(row.passes).lower(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ARM64 NEON and AMX fixed-gate audits by selecting the fastest correct backend per subkernel."
    )
    parser.add_argument("--neon", type=Path, required=True)
    parser.add_argument("--amx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detailed-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_combined(
        output=args.output,
        detailed_output=args.detailed_output,
        neon_rows=read_backend_rows(args.neon, "neon"),
        amx_rows=read_backend_rows(args.amx, "amx"),
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.detailed_output}")


if __name__ == "__main__":
    main()
