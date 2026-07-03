from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


BACKENDS: tuple[tuple[str, str, str], ...] = (
    ("native_linear", "torch", "torch"),
    ("native_linear_mlp_plan", "native-cpu-plan", "torch"),
    ("native_linear_rmsnorm", "torch", "native-cpu"),
    ("native_linear_mlp_plan_rmsnorm", "native-cpu-plan", "native-cpu"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Qwen Level 2 native depth backends and summarize full-forward "
            "timings. This is for testing Axiom-style native islands against the "
            "actual model depth path."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--load-dyadic", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[8, 64])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--backend",
        action="append",
        choices=[name for name, _, _ in BACKENDS],
        help="Backend label to run. May be repeated. Defaults to all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/qwen_depth_backend_sweep.csv"),
    )
    parser.add_argument(
        "--keep-raw-dir",
        type=Path,
        help="Optional directory to keep raw profile_qwen_depth CSVs.",
    )
    return parser.parse_args()


def selected_backends(labels: list[str] | None) -> tuple[tuple[str, str, str], ...]:
    if not labels:
        return BACKENDS
    wanted = set(labels)
    return tuple(row for row in BACKENDS if row[0] in wanted)


def run_profile(args: argparse.Namespace, label: str, mlp_backend: str, norm_backend: str, raw_csv: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "experiments/level2/profile_qwen_depth.py"),
        "--model-dir",
        str(args.model_dir),
        "--bits",
        str(args.bits),
        "--sequence-lengths",
        *[str(value) for value in args.sequence_lengths],
        "--batch-sizes",
        *[str(value) for value in args.batch_sizes],
        "--repeats",
        str(args.repeats),
        "--skip-module-timing",
        "--qwen-mlp-backend",
        mlp_backend,
        "--qwen-norm-backend",
        norm_backend,
        "--load-dyadic",
        str(args.load_dyadic),
        "--output",
        str(raw_csv),
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    print(f"Running {label}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def collect_rows(raw_csv: Path, label: str, mlp_backend: str, norm_backend: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with raw_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("scope") != "full_forward":
                continue
            repeats = max(1, int(row["calls"]))
            total_ms = float(row["total_ms"])
            rows.append(
                {
                    "backend_label": label,
                    "qwen_mlp_backend": mlp_backend,
                    "qwen_norm_backend": norm_backend,
                    "batch_size": row["batch_size"],
                    "sequence_length": row["sequence_length"],
                    "repeats": row["repeats"],
                    "calls": row["calls"],
                    "total_ms": f"{total_ms:.6f}",
                    "avg_forward_ms": f"{total_ms / repeats:.6f}",
                    "min_forward_ms": f"{float(row['min_us']) / 1000.0:.6f}",
                    "max_forward_ms": f"{float(row['max_us']) / 1000.0:.6f}",
                    "raw_csv": str(raw_csv),
                }
            )
    return rows


def write_summary(output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend_label",
        "qwen_mlp_backend",
        "qwen_norm_backend",
        "batch_size",
        "sequence_length",
        "repeats",
        "calls",
        "total_ms",
        "avg_forward_ms",
        "min_forward_ms",
        "max_forward_ms",
        "raw_csv",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    backends = selected_backends(args.backend)
    if args.keep_raw_dir is not None:
        args.keep_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_dir_context = None
        raw_dir = args.keep_raw_dir
    else:
        raw_dir_context = tempfile.TemporaryDirectory(prefix="qwen-depth-sweep-")
        raw_dir = Path(raw_dir_context.name)
    try:
        for label, mlp_backend, norm_backend in backends:
            raw_csv = raw_dir / f"{label}.csv"
            run_profile(args, label, mlp_backend, norm_backend, raw_csv)
            rows.extend(collect_rows(raw_csv, label, mlp_backend, norm_backend))
    finally:
        if raw_dir_context is not None:
            raw_dir_context.cleanup()
    write_summary(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
