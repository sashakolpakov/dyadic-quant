from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit comparability between Level 1 materialized Qwen textual "
            "metrics and Level 2 native-dyop textual metrics."
        )
    )
    parser.add_argument(
        "--level1-generations",
        type=Path,
        default=Path("results/qwen25_generations.json"),
    )
    parser.add_argument(
        "--level2-generations",
        type=Path,
        default=Path("results/level2/qwen25_dyop_generations.json"),
    )
    parser.add_argument(
        "--level1-summary",
        type=Path,
        default=Path("results/qwen25_textual_summary.csv"),
    )
    parser.add_argument(
        "--level1-comparison",
        type=Path,
        default=Path("results/qwen25_textual_comparison.csv"),
    )
    parser.add_argument(
        "--level1-metadata",
        type=Path,
        default=Path("results/qwen25_textual_metadata.json"),
    )
    parser.add_argument(
        "--level2-summary",
        type=Path,
        default=Path("results/level2/qwen_textual_full_native6_8/qwen25_textual_summary.csv"),
    )
    parser.add_argument(
        "--level2-comparison",
        type=Path,
        default=Path("results/level2/qwen_textual_full_native6_8/qwen25_textual_comparison.csv"),
    )
    parser.add_argument(
        "--level2-metadata",
        type=Path,
        default=Path("results/level2/qwen_textual_full_native6_8/qwen25_textual_metadata.json"),
    )
    parser.add_argument("--source-variant", default="bf16_source")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/level2/qwen_textual_level1_level2_audit"),
    )
    return parser.parse_args()


def variant_bits(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name.removeprefix(prefix)
    try:
        return int(suffix)
    except ValueError:
        return None


def comparable_rows(level1: pd.DataFrame, level2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    level1_variants = {
        "per_channel": "dyadic_",
        "group32": "dyadic_g32_",
    }
    for _, l2_row in level2.iterrows():
        bits = variant_bits(str(l2_row["variant"]), "dyop_native_")
        if bits is None:
            continue
        for baseline_name, prefix in level1_variants.items():
            l1_variant = f"{prefix}{bits}"
            match = level1[
                (level1["family"] == l2_row["family"])
                & (level1["variant"] == l1_variant)
            ]
            if match.empty:
                continue
            l1_row = match.iloc[0]
            rows.append(
                {
                    "family": l2_row["family"],
                    "bits": bits,
                    "level1_baseline": baseline_name,
                    "level1_variant": l1_variant,
                    "level2_variant": l2_row["variant"],
                    "level1_prompts": int(l1_row["prompts"]),
                    "level2_prompts": int(l2_row["prompts"]),
                    "level1_judged_prompts": int(l1_row["judged_prompts"]),
                    "level2_judged_prompts": int(l2_row["judged_prompts"]),
                    "level1_mean_cosine": float(l1_row["mean_cosine"]),
                    "level2_mean_cosine": float(l2_row["mean_cosine"]),
                    "delta_mean_cosine_l2_minus_l1": float(
                        l2_row["mean_cosine"] - l1_row["mean_cosine"]
                    ),
                    "level1_judge_equivalent_rate": float(
                        l1_row["judge_equivalent_rate"]
                    ),
                    "level2_judge_equivalent_rate": float(
                        l2_row["judge_equivalent_rate"]
                    ),
                    "delta_judge_rate_l2_minus_l1": float(
                        l2_row["judge_equivalent_rate"]
                        - l1_row["judge_equivalent_rate"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def family_strength(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, group in summary.groupby("family"):
        judged = group[group["judged_prompts"] > 0]
        rows.append(
            {
                "family": family,
                "variants": len(group),
                "mean_cosine_mean": float(group["mean_cosine"].mean()),
                "mean_cosine_max": float(group["mean_cosine"].max()),
                "judge_rate_mean": float(judged["judge_equivalent_rate"].mean()),
                "judge_rate_max": float(judged["judge_equivalent_rate"].max()),
                "judge_rate_min": float(judged["judge_equivalent_rate"].min()),
            }
        )
    return pd.DataFrame(rows)


def missing_judges(path: Path) -> int | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "judge_equivalent" not in frame.columns:
        return None
    return int(frame["judge_equivalent"].isna().sum())


def read_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    level1_generations = json.loads(args.level1_generations.read_text())
    level2_generations = json.loads(args.level2_generations.read_text())
    prompts_equal = level1_generations["prompts"] == level2_generations["prompts"]
    source_equal = (
        level1_generations["generations"].get(args.source_variant)
        == level2_generations["generations"].get(args.source_variant)
    )

    level1_summary = pd.read_csv(args.level1_summary)
    level2_summary = pd.read_csv(args.level2_summary)
    level1_metadata = read_metadata(args.level1_metadata)
    level2_metadata = read_metadata(args.level2_metadata)
    judge_metadata_comparable = (
        level1_metadata is not None
        and level2_metadata is not None
        and level1_metadata.get("judge_backend") == level2_metadata.get("judge_backend")
        and level1_metadata.get("judge_model") == level2_metadata.get("judge_model")
        and level1_metadata.get("embed_model") == level2_metadata.get("embed_model")
        and bool(level1_metadata.get("judge_enabled"))
        == bool(level2_metadata.get("judge_enabled"))
    )
    comparisons = comparable_rows(level1_summary, level2_summary)
    level1_family = family_strength(level1_summary)
    level2_family = family_strength(level2_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_path = args.output_dir / "qwen_l1_l2_textual_deltas.csv"
    l1_family_path = args.output_dir / "qwen_level1_family_strength.csv"
    l2_family_path = args.output_dir / "qwen_level2_family_strength.csv"
    comparisons.to_csv(comparisons_path, index=False)
    level1_family.to_csv(l1_family_path, index=False)
    level2_family.to_csv(l2_family_path, index=False)

    audit = {
        "level1_generations": str(args.level1_generations),
        "level2_generations": str(args.level2_generations),
        "level1_summary": str(args.level1_summary),
        "level2_summary": str(args.level2_summary),
        "level1_comparison": str(args.level1_comparison),
        "level2_comparison": str(args.level2_comparison),
        "level1_metadata": str(args.level1_metadata),
        "level2_metadata": str(args.level2_metadata),
        "source_variant": args.source_variant,
        "prompts_equal": prompts_equal,
        "source_generations_equal": source_equal,
        "level1_missing_judges": missing_judges(args.level1_comparison),
        "level2_missing_judges": missing_judges(args.level2_comparison),
        "level1_judge_model": (
            level1_metadata.get("judge_model") if level1_metadata else None
        ),
        "level2_judge_model": (
            level2_metadata.get("judge_model") if level2_metadata else None
        ),
        "level1_judge_backend": (
            level1_metadata.get("judge_backend") if level1_metadata else None
        ),
        "level2_judge_backend": (
            level2_metadata.get("judge_backend") if level2_metadata else None
        ),
        "level1_embed_model": (
            level1_metadata.get("embed_model") if level1_metadata else None
        ),
        "level2_embed_model": (
            level2_metadata.get("embed_model") if level2_metadata else None
        ),
        "judge_metadata_comparable": judge_metadata_comparable,
        "notes": [
            (
                "ARC is the stronger free-generation LLM-judge family here: "
                "the source task asks for a constrained answer and explanation."
            ),
            (
                "WikiText is an unconstrained continuation task for an instruct "
                "model; low judge-equivalence can reflect early free-running "
                "divergence even when cosine remains moderate."
            ),
        ],
        "artifacts": {
            "deltas": str(comparisons_path),
            "level1_family_strength": str(l1_family_path),
            "level2_family_strength": str(l2_family_path),
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not prompts_equal:
        raise SystemExit("Level 1 and Level 2 prompt sets differ")
    if not source_equal:
        raise SystemExit("Level 1 and Level 2 source generations differ")
    if not judge_metadata_comparable:
        raise SystemExit("Level 1 and Level 2 judge/embed metadata are not comparable")

    print(comparisons.to_string(index=False))
    print(f"Wrote {comparisons_path}")
    print(f"Wrote {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
