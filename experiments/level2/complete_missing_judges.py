from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.textgen import exact_match
from experiments.compare_generations import (
    chunk_items,
    judge_prompt,
    normalize_judge_verdicts,
    run_judge,
)


def is_judge_limit_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "api_error_status=429" in text or "session limit" in text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find Level 2 textual comparison reports with blank judge verdicts "
            "and retry only the missing prompt groups."
        )
    )
    parser.add_argument(
        "--generations-file",
        type=Path,
        help=(
            "Generations JSON used to reconstruct judge prompts. If omitted, "
            "infer it per report from metadata or known textual generation files."
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        action="append",
        help=(
            "Specific comparison CSV to repair. May be passed more than once. "
            "If omitted, scan --results-root for incomplete reports."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/level2"),
        help="Root scanned for qwen25_textual_comparison.csv reports.",
    )
    parser.add_argument(
        "--source-variant",
        help="Override source variant. If omitted, infer it per report.",
    )
    parser.add_argument(
        "--judge-model",
        help=(
            "Override judge model. If omitted, reuse report metadata when "
            "available; otherwise use the Claude CLI default."
        ),
    )
    parser.add_argument(
        "--judge-backend",
        choices=["claude", "ollama"],
        help=(
            "Override judge backend. If omitted, reuse report metadata when "
            "available; otherwise use claude."
        ),
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=300.0,
        help="Initial per-prompt judge timeout in seconds.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Retry passes over remaining blanks.",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.5,
        help="Timeout multiplier after each retry pass.",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=0,
        help=(
            "Maximum variants per judge call. 0 judges all missing variants for "
            "a prompt in one call; use 1 for weaker local judges."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list incomplete reports and missing rows.",
    )
    return parser.parse_args()


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    for (family, variant), group in frame.groupby(["family", "variant"]):
        judged = group["judge_equivalent"].dropna()
        summary_rows.append(
            {
                "family": family,
                "variant": variant,
                "prompts": len(group),
                "exact_match_rate": group["exact_match"].mean(),
                "mean_edit_ratio": group["edit_ratio"].mean(),
                "mean_token_jaccard": group["token_jaccard"].mean(),
                "mean_cosine": group["cosine"].mean(),
                "judge_equivalent_rate": judged.mean() if len(judged) else None,
                "judged_prompts": len(judged),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "qwen25_textual_summary.csv", index=False
    )


def candidate_reports(args: argparse.Namespace) -> list[Path]:
    if args.comparison_csv:
        return [path for path in args.comparison_csv]
    return sorted(args.results_root.rglob("qwen25_textual_comparison.csv"))


def missing_count(path: Path) -> int:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if "judge_equivalent" not in frame.columns:
        return 0
    return int(frame["judge_equivalent"].isna().sum())


def incomplete_reports(args: argparse.Namespace) -> list[Path]:
    return [path for path in candidate_reports(args) if missing_count(path) > 0]


def read_metadata(report: Path) -> dict:
    path = report.parent / "qwen25_textual_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def generation_candidates(report: Path, args: argparse.Namespace, metadata: dict) -> list[Path]:
    candidates: list[Path] = []
    if args.generations_file is not None:
        candidates.append(args.generations_file)
    if metadata.get("generations_file"):
        candidates.append(Path(str(metadata["generations_file"])))
    candidates.extend(
        [
            report.parent / "qwen25_generations.json",
            report.parent / "qwen25_dyop_generations.json",
            Path("results/level2/qwen25_dyop_generations.json"),
            Path("results/level2/qwen25_dyop_generation_smoke.json"),
            Path("results/qwen25_generations.json"),
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = resolved.resolve() if resolved.exists() else resolved
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def source_candidates(args: argparse.Namespace, metadata: dict) -> list[str]:
    candidates = [
        args.source_variant,
        metadata.get("source_variant"),
        "bf16_source",
        "float32_source",
    ]
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(str(candidate))
    return unique


def report_requirements(report: Path) -> tuple[set[str], set[tuple[str, str]]]:
    frame = pd.read_csv(report)
    variants = {str(value) for value in frame["variant"].dropna().unique()}
    prompt_keys = {
        (str(row.family), str(row.prompt_id))
        for row in frame[["family", "prompt_id"]].itertuples(index=False)
    }
    return variants, prompt_keys


def candidate_matches_report(
    generations_file: Path,
    source: str,
    required_variants: set[str],
    required_prompts: set[tuple[str, str]],
) -> bool:
    if not generations_file.exists():
        return False
    try:
        document = json.loads(generations_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    generations = document.get("generations", {})
    if source not in generations:
        return False
    if not required_variants.issubset(generations):
        return False
    for variant in required_variants | {source}:
        for family, prompt_id in required_prompts:
            if prompt_id not in generations.get(variant, {}).get(family, {}):
                return False
    return True


def infer_report_context(
    report: Path, args: argparse.Namespace
) -> tuple[Path, str, str, str]:
    metadata = read_metadata(report)
    required_variants, required_prompts = report_requirements(report)
    for generations_file in generation_candidates(report, args, metadata):
        for source in source_candidates(args, metadata):
            if candidate_matches_report(
                generations_file,
                source,
                required_variants,
                required_prompts,
            ):
                judge_model = (
                    args.judge_model
                    if args.judge_model is not None
                    else str(metadata.get("judge_model_arg") or "")
                )
                judge_backend = (
                    args.judge_backend
                    if args.judge_backend is not None
                    else str(metadata.get("judge_backend") or "claude")
                )
                return generations_file, source, judge_backend, judge_model
    raise RuntimeError(
        f"could not infer generations/source for {report}; pass "
        "--generations-file and --source-variant explicitly"
    )


def load_generation_context(
    generations_file: Path, source: str
) -> tuple[dict, dict, dict[tuple[str, str], str]]:
    document = json.loads(generations_file.read_text())
    prompts = document["prompts"]
    generations = document["generations"]
    if source not in generations:
        raise RuntimeError(f"source variant '{source}' not in {generations_file}")

    instruction_by: dict[tuple[str, str], str] = {}
    for family, items in prompts.items():
        for item in items:
            instruction_by[(family, item["id"])] = item["prompt"]
    return prompts, generations, instruction_by


def write_metadata(
    *,
    comparison_csv: Path,
    generations_file: Path,
    source: str,
    judge_backend: str,
    judge_model: str,
    judge_timeout: float,
    judge_batch_size: int,
    remaining: int,
) -> None:
    metadata_path = comparison_csv.parent / "qwen25_textual_metadata.json"
    metadata = read_metadata(comparison_csv)
    metadata.update(
        {
            "generations_file": str(generations_file),
            "source_variant": source,
            "judge_backend": judge_backend,
            "judge_model": judge_model or f"{judge_backend}_default",
            "judge_model_arg": judge_model,
            "judge_timeout": judge_timeout,
            "judge_batch_size": judge_batch_size,
            "judge_enabled": True,
            "output_dir": str(comparison_csv.parent),
            "missing_judge_equivalent": remaining,
            "last_completed_by": "experiments/level2/complete_missing_judges.py",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def complete_report(
    *,
    comparison_csv: Path,
    generations_file: Path,
    generations: dict,
    instruction_by: dict[tuple[str, str], str],
    source: str,
    judge_backend: str,
    judge_model: str,
    judge_timeout: float,
    judge_batch_size: int,
) -> int:
    frame = pd.read_csv(comparison_csv)
    missing = frame[frame["judge_equivalent"].isna()]
    if missing.empty:
        write_summary(frame, comparison_csv.parent)
        write_metadata(
            comparison_csv=comparison_csv,
            generations_file=generations_file,
            source=source,
            judge_backend=judge_backend,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
            judge_batch_size=judge_batch_size,
            remaining=0,
        )
        return 0

    for (family, prompt_id), group in missing.groupby(["family", "prompt_id"]):
        reference = generations[source][family][prompt_id]
        variants: dict[str, str] = {}
        for variant in group["variant"]:
            text = generations[variant][family][prompt_id]
            if exact_match(reference, text):
                mask = (
                    (frame["family"] == family)
                    & (frame["prompt_id"] == prompt_id)
                    & (frame["variant"] == variant)
                )
                frame.loc[mask, "judge_equivalent"] = True
                frame.loc[mask, "judge_reason"] = "identical text"
            else:
                variants[str(variant)] = text
        if variants:
            print(
                f"{comparison_csv}: {family}/{prompt_id}: "
                f"judging {len(variants)} missing variants "
                f"(timeout={judge_timeout:.0f}s)",
                flush=True,
            )
            try:
                verdicts = {}
                for batch in chunk_items(variants, judge_batch_size):
                    batch_verdicts = run_judge(
                        judge_prompt(
                            instruction=instruction_by[(family, prompt_id)],
                            reference=reference,
                            variants=batch,
                        ),
                        backend=judge_backend,
                        model=judge_model,
                        timeout=judge_timeout,
                    )
                    verdicts.update(
                        normalize_judge_verdicts(
                            batch_verdicts,
                            set(batch),
                        )
                    )
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                if is_judge_limit_error(error):
                    frame.to_csv(comparison_csv, index=False)
                    raise RuntimeError(
                        "judge API limit reached; stop and rerun after reset"
                    ) from error
                print(f"  judge still missing for {family}/{prompt_id}: {error}")
                frame.to_csv(comparison_csv, index=False)
                continue
            for variant, verdict in verdicts.items():
                mask = (
                    (frame["family"] == family)
                    & (frame["prompt_id"] == prompt_id)
                    & (frame["variant"] == variant)
                )
                if not mask.any():
                    raise RuntimeError(f"judge returned unexpected variant {variant!r}")
                frame.loc[mask, "judge_equivalent"] = bool(verdict["equivalent"])
                frame.loc[mask, "judge_reason"] = str(verdict.get("reason", ""))
        frame.to_csv(comparison_csv, index=False)

    write_summary(frame, comparison_csv.parent)
    remaining = int(frame["judge_equivalent"].isna().sum())
    write_metadata(
        comparison_csv=comparison_csv,
        generations_file=generations_file,
        source=source,
        judge_backend=judge_backend,
        judge_model=judge_model,
        judge_timeout=judge_timeout,
        judge_batch_size=judge_batch_size,
        remaining=remaining,
    )
    print(f"{comparison_csv}: remaining={remaining}")
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {comparison_csv.parent / 'qwen25_textual_summary.csv'}")
    return remaining


def main() -> None:
    args = parse_args()
    reports = incomplete_reports(args)
    if not reports:
        print("No incomplete judge reports found.")
        return
    print("Incomplete judge reports:")
    for report in reports:
        try:
            generations_file, source, judge_backend, judge_model = infer_report_context(
                report, args
            )
            model_label = judge_model or "claude_cli_default"
            print(
                f"  {report}: missing={missing_count(report)} "
                f"generations={generations_file} source={source} "
                f"judge={judge_backend}:{model_label}"
            )
        except RuntimeError as error:
            print(f"  {report}: missing={missing_count(report)} ({error})")
    if args.dry_run:
        return

    timeout = args.judge_timeout
    remaining_by_report = {path: missing_count(path) for path in reports}
    context_cache: dict[tuple[Path, str], tuple[dict, dict[tuple[str, str], str]]] = {}
    for attempt in range(max(1, args.passes)):
        active = [path for path, count in remaining_by_report.items() if count > 0]
        if not active:
            break
        print(f"Retry pass {attempt + 1}/{args.passes} with timeout={timeout:.0f}s")
        for report in active:
            generations_file, source, judge_backend, judge_model = infer_report_context(
                report, args
            )
            cache_key = (generations_file, source)
            if cache_key not in context_cache:
                _, generations, instruction_by = load_generation_context(
                    generations_file, source
                )
                context_cache[cache_key] = (generations, instruction_by)
            generations, instruction_by = context_cache[cache_key]
            remaining_by_report[report] = complete_report(
                comparison_csv=report,
                generations_file=generations_file,
                generations=generations,
                instruction_by=instruction_by,
                source=source,
                judge_backend=judge_backend,
                judge_model=judge_model,
                judge_timeout=timeout,
                judge_batch_size=args.judge_batch_size,
            )
        timeout *= args.timeout_multiplier

    total_remaining = sum(remaining_by_report.values())
    if total_remaining:
        for report, count in remaining_by_report.items():
            if count:
                print(f"Still incomplete: {report}: missing={count}")
        raise SystemExit(1)
    print("All missing judge verdicts completed.")


if __name__ == "__main__":
    main()
