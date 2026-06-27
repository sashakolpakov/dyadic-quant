"""Sweep the dyadic exponent group size and summarize quality vs effective bits.

For each group size, the source BF16 Qwen checkpoint is encoded once at maximum
depth and evaluated at every prefix width (perplexity, next-token agreement with
the source, ARC-Easy likelihood). The summary reports effective bits/weight so
the block-wise dyadic variants can be compared to the GGUF controls on a
bits-matched basis rather than by nominal label.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=[0, 64, 32, 16],
        help="0 means one exponent per output channel (per-channel baseline).",
    )
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--embedding-bits", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--arc-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    variant_names: list[tuple[int, str]] = []
    for group_size in args.group_sizes:
        suffix = "g0" if group_size == 0 else f"g{group_size}"
        if args.embedding_bits:
            suffix += f"_emb{args.embedding_bits}"
        variant = f"qwen25_dyadic_{suffix}"
        variant_names.append((group_size, variant))
        run(
            [
                python,
                str(root / "experiments/run_qwen_dyadic.py"),
                "--model-dir",
                str(args.source_dir.resolve()),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--dtype",
                "bfloat16",
                "--bits",
                *[str(b) for b in args.bits],
                "--group-size",
                str(group_size),
                "--embedding-bits",
                str(args.embedding_bits),
                "--max-tokens",
                str(args.max_tokens),
                "--sequence-length",
                str(args.sequence_length),
                "--arc-limit",
                str(args.arc_limit),
                "--variant-name",
                variant,
                "--output-dir",
                str(output),
            ]
        )

    rows: list[dict[str, object]] = []
    for group_size, variant in variant_names:
        frame = pd.read_csv(output / f"{variant}_results.csv")
        for _, row in frame.iterrows():
            if row["bits_per_weight"] == 16:
                continue  # the source reference row is identical across runs
            rows.append(
                {
                    "group_size": "per_channel" if group_size == 0 else group_size,
                    "bits_per_weight": int(row["bits_per_weight"]),
                    "effective_bits_per_weight": row["effective_bits_per_weight"],
                    "perplexity": row["perplexity"],
                    "next_token_agreement": row["next_token_agreement"],
                    "arc_easy_likelihood": row["arc_easy_accuracy"],
                }
            )
    summary = pd.DataFrame(rows).sort_values(
        ["bits_per_weight", "group_size"], key=lambda s: s.map(str)
    )
    summary_path = output / "qwen25_group_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
