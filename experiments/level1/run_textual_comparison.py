"""Drive the full textual-output comparison across all controlled variants.

Generates greedy text from the BF16 source, the per-channel dyadic prefixes
(4/5/6/8-bit), and each dequantized GGUF control (Q4_K_M, Q6_K, Q8_0) using one
common Transformers backend, so the comparison isolates the effect of the
*weights* rather than the execution backend. It then scores every variant's text
against the BF16 source with lexical, embedding-cosine, and LLM-judge metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/level1"))
    parser.add_argument("--arc-count", type=int, default=20)
    parser.add_argument("--wikitext-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument(
        "--group-sizes",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional extra Level 1 grouped dyadic passes. A value of 32 "
            "produces variants named dyadic_g32_<bits>."
        ),
    )
    parser.add_argument(
        "--control-lineage",
        type=Path,
        help=(
            "GGUF control lineage JSON. Defaults to qwen25_control_lineage.json "
            "inside the output directory."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional judge model override; empty uses the session default.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["claude", "ollama"],
        default="claude",
    )
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse an existing generations file and only run the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    python = sys.executable
    output = args.output_dir.resolve()
    generations = output / "qwen25_generations.json"
    common_gen = [
        "--model-dir",
        str(args.source_dir.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--arc-count",
        str(args.arc_count),
        "--wikitext-count",
        str(args.wikitext_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--generations-file",
        str(generations),
    ]

    if not args.skip_generation:
        if generations.exists():
            generations.unlink()  # start a fresh, consistent prompt set
        # Source + dyadic prefixes (one model load).
        run(
            [
                python,
                str(root / "experiments/level1/run_textual_generation.py"),
                *common_gen,
                "--variant",
                "bf16_source",
                "--dyadic-prefix",
                "dyadic",
                "--bits",
                *[str(bits) for bits in args.bits],
            ]
        )
        for group_size in args.group_sizes:
            run(
                [
                    python,
                    str(root / "experiments/level1/run_textual_generation.py"),
                    *common_gen,
                    "--variant",
                    "bf16_source",
                    "--skip-reference-generation",
                    "--dyadic-prefix",
                    f"dyadic_g{group_size}",
                    "--group-size",
                    str(group_size),
                    "--bits",
                    *[str(bits) for bits in args.bits],
                ]
            )
        # Dequantized GGUF controls.
        lineage_path = args.control_lineage or (output / "qwen25_control_lineage.json")
        if not lineage_path.exists():
            raise RuntimeError(
                f"GGUF control lineage not found: {lineage_path}. "
                "Pass --control-lineage for separated rerun directories."
            )
        lineage = json.loads(lineage_path.read_text())
        for key, variant in lineage["variants"].items():
            run(
                [
                    python,
                    str(root / "experiments/level1/run_textual_generation.py"),
                    *common_gen,
                    "--gguf-file",
                    variant["file"],
                    "--reference-only",
                    "--variant",
                    key,
                ]
            )

    compare = [
        python,
        str(root / "experiments/level1/compare_generations.py"),
        "--generations-file",
        str(generations),
        "--output-dir",
        str(output),
        "--judge-model",
        args.judge_model,
        "--judge-backend",
        args.judge_backend,
        "--judge-timeout",
        str(args.judge_timeout),
    ]
    if args.no_judge:
        compare.append("--no-judge")
    run(compare)


if __name__ == "__main__":
    main()
