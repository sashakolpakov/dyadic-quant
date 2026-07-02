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
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        help="llama.cpp checkout used to build the GGUF controls; "
        "required unless --skip-build.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--arc-limit", type=int, default=100)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir.resolve()
    data_dir = args.data_dir.resolve()

    if not args.skip_build:
        if args.llama_cpp_dir is None:
            raise SystemExit("--llama-cpp-dir is required unless --skip-build")
        run(
            [
                python,
                str(root / "experiments/build_qwen_gguf_controls.py"),
                "--source-dir",
                str(source_dir),
                "--llama-cpp-dir",
                str(args.llama_cpp_dir.resolve()),
                "--output-dir",
                str(output),
            ]
        )

    reference_predictions = output / "qwen25_bf16_reference_predictions.pt"
    common = [
        "--model-dir",
        str(source_dir),
        "--data-dir",
        str(data_dir),
        "--dtype",
        "bfloat16",
        "--max-tokens",
        str(args.max_tokens),
        "--sequence-length",
        str(args.sequence_length),
        "--arc-limit",
        str(args.arc_limit),
        "--output-dir",
        str(output),
    ]
    run(
        [
            python,
            str(root / "experiments/level1/run_qwen_dyadic.py"),
            *common,
            "--bits",
            "4",
            "5",
            "6",
            "8",
            "--variant-name",
            "qwen25_bf16_dyadic",
            "--save-reference-predictions",
            str(reference_predictions),
        ]
    )

    lineage = json.loads((output / "qwen25_control_lineage.json").read_text())

    # Native-Ollama baseline for the BF16 source, so the summary carries native
    # throughput and generation accuracy for the reference itself.
    run(
        [
            python,
            str(root / "experiments/run_ollama_llm.py"),
            "--model",
            lineage["bf16_gguf"]["ollama_model"],
            "--data-dir",
            str(data_dir),
            "--arc-limit",
            str(args.arc_limit),
            "--result-prefix",
            "ollama_qwen25_bf16_source",
            "--output-dir",
            str(output),
        ]
    )

    for key, variant in lineage["variants"].items():
        run(
            [
                python,
                str(root / "experiments/level1/run_qwen_dyadic.py"),
                *common,
                "--gguf-file",
                variant["file"],
                "--reference-only",
                "--variant-name",
                f"qwen25_{key}",
                "--reference-predictions",
                str(reference_predictions),
            ]
        )
        run(
            [
                python,
                str(root / "experiments/run_ollama_llm.py"),
                "--model",
                variant["ollama_model"],
                "--data-dir",
                str(data_dir),
                "--arc-limit",
                str(args.arc_limit),
                "--result-prefix",
                f"ollama_qwen25_{key}",
                "--output-dir",
                str(output),
            ]
        )

    run(
        [
            python,
            str(root / "experiments/summarize_qwen_controlled.py"),
            "--results-dir",
            str(output),
        ]
    )


if __name__ == "__main__":
    main()
