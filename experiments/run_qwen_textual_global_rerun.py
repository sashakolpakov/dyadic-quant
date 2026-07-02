"""Run a provenance-clean Qwen textual comparison for Level 1 and Level 2.

The output tree is deliberately split by level:

* ``level1_materialized`` contains Level 1 full-tensor materialized outputs.
* ``level2_native_dyop`` contains Level 2 native dyop outputs.
* ``audit`` contains the cross-level comparability report.

Both levels are judged with the same explicit Claude model and the same local
embedding model, and Level 2 is seeded from the exact Level 1 BF16 source text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/level1/qwen25_textual_global_rerun"),
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help=(
            "Explicit judge model. Use a concrete Claude model name or a local "
            "Ollama model such as gemma4:e2b; do not leave this implicit."
        ),
    )
    parser.add_argument(
        "--judge-backend",
        choices=["claude", "ollama"],
        default="claude",
    )
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--control-lineage",
        type=Path,
        default=Path("results/level1/qwen25_control_lineage.json"),
    )
    parser.add_argument(
        "--load-dyadic",
        type=Path,
        default=Path("data/checkpoints/Qwen2.5-0.5B-Instruct-progressive-dyadic-8bit.pt"),
    )
    parser.add_argument(
        "--level1-generations-file",
        type=Path,
        help=(
            "Existing Level 1 generations JSON to copy into the rerun tree "
            "before recomputing metrics. Useful when MPS is unavailable."
        ),
    )
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument(
        "--level1-group-sizes",
        nargs="*",
        type=int,
        default=[32],
        help="Extra Level 1 grouped dyadic variants, e.g. 32 -> dyadic_g32_<bits>.",
    )
    parser.add_argument("--arc-count", type=int, default=20)
    parser.add_argument("--wikitext-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dyop-cpu-threads", type=int, default=16)
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--repair-passes", type=int, default=3)
    parser.add_argument("--repair-timeout", type=float, default=480.0)
    parser.add_argument("--timeout-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--skip-level1-generation",
        action="store_true",
        help="Reuse output-root/level1_materialized/qwen25_generations.json.",
    )
    parser.add_argument(
        "--skip-level2-generation",
        action="store_true",
        help="Reuse output-root/level2_native_dyop/qwen25_dyop_generations.json.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Only run generation steps; skip judge/comparison/audit.",
    )
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def validate_reused_level1_generations(
    generations_file: Path, *, arc_count: int, wikitext_count: int
) -> None:
    document = json.loads(generations_file.read_text())
    prompts = document.get("prompts", {})
    actual = {
        "arc": len(prompts.get("arc", [])),
        "wikitext": len(prompts.get("wikitext", [])),
    }
    expected = {"arc": arc_count, "wikitext": wikitext_count}
    if actual != expected:
        raise RuntimeError(
            "reused Level 1 generations prompt counts do not match requested "
            f"counts: actual={actual}, expected={expected}"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output_root = args.output_root.resolve()
    level1_dir = output_root / "level1_materialized"
    level2_dir = output_root / "level2_native_dyop"
    audit_dir = output_root / "audit"
    level1_generations = level1_dir / "qwen25_generations.json"
    level2_generations = level2_dir / "qwen25_dyop_generations.json"

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(output_root),
        "judge_model": args.judge_model,
        "judge_backend": args.judge_backend,
        "embed_model": args.embed_model,
        "control_lineage": str(args.control_lineage.resolve()),
        "load_dyadic": str(args.load_dyadic.resolve()),
        "level1_generations_file": (
            str(args.level1_generations_file.resolve())
            if args.level1_generations_file
            else None
        ),
        "bits": args.bits,
        "level1_group_sizes": args.level1_group_sizes,
        "arc_count": args.arc_count,
        "wikitext_count": args.wikitext_count,
        "max_new_tokens": args.max_new_tokens,
        "max_tokens": args.max_tokens,
        "dyop_cpu_threads": args.dyop_cpu_threads,
        "judge_timeout": args.judge_timeout,
        "repair_passes": args.repair_passes,
        "repair_timeout": args.repair_timeout,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    level1_command = [
        python,
        str(root / "experiments/level1/run_textual_comparison.py"),
        "--source-dir",
        str(args.source_dir.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-dir",
        str(level1_dir),
        "--arc-count",
        str(args.arc_count),
        "--wikitext-count",
        str(args.wikitext_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--bits",
        *[str(bits) for bits in args.bits],
        "--control-lineage",
        str(args.control_lineage.resolve()),
        "--judge-model",
        args.judge_model,
        "--judge-backend",
        args.judge_backend,
        "--judge-timeout",
        str(args.judge_timeout),
    ]
    if args.level1_group_sizes:
        level1_command.extend(
            ["--group-sizes", *[str(size) for size in args.level1_group_sizes]]
        )
    copied_level1_generations = False
    if args.level1_generations_file is not None:
        validate_reused_level1_generations(
            args.level1_generations_file,
            arc_count=args.arc_count,
            wikitext_count=args.wikitext_count,
        )
        level1_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.level1_generations_file, level1_generations)
        copied_level1_generations = True
        level1_command.append("--skip-generation")
    elif args.skip_level1_generation:
        level1_command.append("--skip-generation")
    if args.skip_comparison and copied_level1_generations:
        print(f"Copied {args.level1_generations_file} -> {level1_generations}")
    else:
        run(level1_command)

    if not args.skip_level2_generation:
        run(
            [
                python,
                str(root / "experiments/level2/seed_qwen_textual_reference.py"),
                "--source-generations",
                str(level1_generations),
                "--source-variant",
                "bf16_source",
                "--output",
                str(level2_generations),
            ]
        )
        dyop_env = os.environ.copy()
        dyop_env["DYOP_CPU_THREADS"] = str(args.dyop_cpu_threads)
        run(
            [
                python,
                str(root / "experiments/level2/run_qwen_textual_generation.py"),
                "--model-dir",
                str(args.source_dir.resolve()),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--bits",
                *[str(bits) for bits in args.bits],
                "--arc-count",
                str(args.arc_count),
                "--wikitext-count",
                str(args.wikitext_count),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--max-tokens",
                str(args.max_tokens),
                "--source-variant",
                "bf16_source",
                "--dyop-prefix",
                "dyop_native",
                "--skip-source-generation",
                "--load-dyadic",
                str(args.load_dyadic.resolve()),
                "--generations-file",
                str(level2_generations),
            ],
            env=dyop_env,
        )

    if args.skip_comparison:
        return

    run(
        [
            python,
            str(root / "experiments/level1/compare_generations.py"),
            "--generations-file",
            str(level2_generations),
            "--source-variant",
            "bf16_source",
            "--embed-model",
            args.embed_model,
            "--judge-model",
            args.judge_model,
            "--judge-backend",
            args.judge_backend,
            "--judge-timeout",
            str(args.judge_timeout),
            "--output-dir",
            str(level2_dir),
        ]
    )
    repair_completed = True
    try:
        run(
            [
                python,
                str(root / "experiments/level2/complete_missing_judges.py"),
                "--results-root",
                str(output_root),
                "--judge-model",
                args.judge_model,
                "--judge-backend",
                args.judge_backend,
                "--judge-timeout",
                str(args.repair_timeout),
                "--passes",
                str(args.repair_passes),
                "--timeout-multiplier",
                str(args.timeout_multiplier),
            ]
        )
    except subprocess.CalledProcessError as error:
        repair_completed = False
        print(f"Judge repair did not complete: {error}", flush=True)
    run(
        [
            python,
            str(root / "experiments/level2/compare_qwen_level1_level2_textual.py"),
            "--level1-generations",
            str(level1_generations),
            "--level1-summary",
            str(level1_dir / "qwen25_textual_summary.csv"),
            "--level1-comparison",
            str(level1_dir / "qwen25_textual_comparison.csv"),
            "--level1-metadata",
            str(level1_dir / "qwen25_textual_metadata.json"),
            "--level2-generations",
            str(level2_generations),
            "--level2-summary",
            str(level2_dir / "qwen25_textual_summary.csv"),
            "--level2-comparison",
            str(level2_dir / "qwen25_textual_comparison.csv"),
            "--level2-metadata",
            str(level2_dir / "qwen25_textual_metadata.json"),
            "--output-dir",
            str(audit_dir),
        ]
    )
    print(f"Wrote global rerun artifacts under {output_root}")
    if not repair_completed:
        raise SystemExit("global rerun finished with incomplete judge repair")


if __name__ == "__main__":
    main()
