from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Level 2 native dyop evidence from a replication run."
        )
    )
    parser.add_argument("--level2-dir", type=Path, required=True)
    parser.add_argument("--level1-dir", type=Path)
    parser.add_argument("--qwen-results", type=Path)
    parser.add_argument("--textual-summary", type=Path)
    parser.add_argument("--textual-comparison", type=Path)
    parser.add_argument("--qwen-kernels", type=Path)
    parser.add_argument("--textual-audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--min-qwen-agreement", type=float)
    parser.add_argument("--max-qwen-perplexity-ratio", type=float)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def scalar(row: pd.Series, name: str) -> Any:
    value = row.get(name)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def finite_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def native_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "execution_backend" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["execution_backend"] == "level2-native"]


def reference_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    if "execution_backend" in frame.columns:
        source = frame[frame["execution_backend"].astype(str).str.contains("source|torch", regex=True)]
        if not source.empty:
            return source.iloc[0]
    return frame.iloc[0]


def summarize_qwen(
    frame: pd.DataFrame | None,
    bits: list[int],
    *,
    min_agreement: float | None,
    max_perplexity_ratio: float | None,
) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if frame is None:
        return pd.DataFrame(rows), ["missing qwen native quality results"]
    ref = reference_row(frame)
    source_bytes = float(ref["total_model_bytes"]) if ref is not None else None
    source_perplexity = finite_number(ref.get("perplexity")) if ref is not None else None
    source_arc = finite_number(ref.get("arc_easy_accuracy")) if ref is not None else None
    native = native_rows(frame)
    for bit in bits:
        match = native[native["bits_per_weight"] == bit]
        if match.empty:
            issues.append(f"missing qwen native row for {bit} bit")
            continue
        row = match.iloc[0]
        total_bytes = finite_number(row["total_model_bytes"])
        perplexity = finite_number(row.get("perplexity"))
        agreement = finite_number(row.get("next_token_agreement"))
        arc = finite_number(row.get("arc_easy_accuracy"))
        if total_bytes is None or total_bytes <= 0:
            issues.append(f"qwen {bit} bit has invalid total_model_bytes")
            continue
        if perplexity is None:
            issues.append(f"qwen {bit} bit has invalid perplexity")
        if agreement is None:
            issues.append(f"qwen {bit} bit has invalid next_token_agreement")
        elif min_agreement is not None and agreement < min_agreement:
            issues.append(
                f"qwen {bit} bit agreement {agreement:.4f} below {min_agreement:.4f}"
            )
        perplexity_ratio = (
            perplexity / source_perplexity
            if perplexity is not None and source_perplexity
            else None
        )
        if (
            max_perplexity_ratio is not None
            and perplexity_ratio is not None
            and perplexity_ratio > max_perplexity_ratio
        ):
            issues.append(
                f"qwen {bit} bit perplexity ratio {perplexity_ratio:.4f} "
                f"above {max_perplexity_ratio:.4f}"
            )
        rows.append(
            {
                "model": "qwen",
                "bits": bit,
                "native_backend": scalar(row, "execution_backend"),
                "linear_backend": scalar(row, "level2_linear_backend"),
                "embedding_backend": scalar(row, "level2_embedding_backend"),
                "total_model_bytes": total_bytes,
                "compression_vs_source": (
                    source_bytes / total_bytes if source_bytes else None
                ),
                "effective_bits_per_weight": scalar(row, "effective_bits_per_weight"),
                "perplexity": perplexity,
                "perplexity_ratio_vs_reference": perplexity_ratio,
                "next_token_agreement": agreement,
                "arc_easy_accuracy": arc,
                "arc_easy_delta_vs_reference": (
                    arc - source_arc if arc is not None and source_arc is not None else None
                ),
                "evaluated_tokens": scalar(row, "evaluated_tokens"),
                "wikitext_tokens_per_s": scalar(row, "wikitext_tokens_per_s"),
            }
        )
    if not native.empty:
        backends = set(native.get("level2_linear_backend", pd.Series(dtype=str)).dropna())
        if "native-cpu" not in backends:
            issues.append("qwen native rows do not use native-cpu linear backend")
    return pd.DataFrame(rows), issues


def summarize_textual(
    summary: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    bits: list[int],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    issues: list[str] = []
    if summary is None:
        return pd.DataFrame(), ["missing qwen textual summary"], {}
    rows = summary.copy()
    rows = rows[rows["variant"].astype(str).str.startswith("dyop_native_")]
    if rows.empty:
        issues.append("missing dyop_native textual variants")
    variants = set(rows["variant"].astype(str).unique()) if not rows.empty else set()
    for bit in bits:
        expected = f"dyop_native_{bit}"
        if expected not in variants:
            issues.append(f"missing textual metrics for {expected}")
    families = set(rows["family"].astype(str).unique()) if not rows.empty else set()
    for family in ("arc", "wikitext"):
        if family not in families:
            issues.append(f"missing textual family: {family}")
    if comparison is None:
        issues.append("missing qwen textual comparison")
        missing_judges = None
    elif "judge_equivalent" in comparison.columns:
        missing_judges = int(comparison["judge_equivalent"].isna().sum())
        if missing_judges:
            issues.append(f"textual comparison has {missing_judges} missing judge rows")
    else:
        missing_judges = None
        issues.append("textual comparison has no judge_equivalent column")
    evidence = {
        "missing_judge_rows": missing_judges,
        "variants": sorted(variants),
        "families": sorted(families),
    }
    return rows, issues, evidence


def summarize_kernels(frame: pd.DataFrame | None, model: str) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    if frame is None:
        return pd.DataFrame(), [f"missing {model} native kernel results"]
    rows = frame.copy()
    if "speedup_vs_torch" in rows.columns:
        rows["passes_torch_speed"] = rows["speedup_vs_torch"] >= 1.0
        slow = rows[~rows["passes_torch_speed"]]
        if not slow.empty:
            names = ", ".join(str(value) for value in slow.get("shape", slow.index).tolist())
            issues.append(f"{model} native kernels below torch speed: {names}")
    return rows, issues


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Level 2 Native Evidence Audit",
        "",
        f"Status: {'PASS' if payload['status']['passes'] else 'INCOMPLETE'}",
        "",
        "## Issues",
        "",
    ]
    if payload["issues"]:
        lines.extend(f"- {issue}" for issue in payload["issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Inputs", ""])
    lines.extend(f"- {name}: {value}" for name, value in payload["inputs"].items())
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    level2 = args.level2_dir
    output = args.output_dir or (level2 / "evidence")
    output.mkdir(parents=True, exist_ok=True)

    qwen_results = args.qwen_results or first_existing(
        level2,
        [
            "qwen25_level2_native_cpu_results.csv",
            "qwen25_native_dyop_quality_current_results.csv",
            "*qwen*native*results.csv",
        ],
    )
    textual_summary = args.textual_summary or first_existing(
        level2,
        ["qwen25_textual_summary.csv"],
    )
    textual_comparison = args.textual_comparison or first_existing(
        level2,
        ["qwen25_textual_comparison.csv"],
    )
    qwen_kernels = args.qwen_kernels or first_existing(
        level2,
        ["qwen_native_kernels.csv", "*linear*.csv", "*native*kernels*.csv"],
    )
    textual_audit = args.textual_audit or first_existing(level2, ["audit.json"])

    issues: list[str] = []
    qwen, qwen_issues = summarize_qwen(
        read_csv(qwen_results),
        args.bits,
        min_agreement=args.min_qwen_agreement,
        max_perplexity_ratio=args.max_qwen_perplexity_ratio,
    )
    textual, textual_issues, textual_evidence = summarize_textual(
        read_csv(textual_summary), read_csv(textual_comparison), args.bits
    )
    qwen_kernel, qwen_kernel_issues = summarize_kernels(
        read_csv(qwen_kernels), "qwen"
    )
    issues.extend(qwen_issues)
    issues.extend(textual_issues)
    issues.extend(qwen_kernel_issues)

    audit_doc = read_json(textual_audit)
    if audit_doc is not None:
        if not audit_doc.get("prompts_equal", True):
            issues.append("level1 and level2 textual prompts differ")
        if not audit_doc.get("source_generations_equal", True):
            issues.append("level1 and level2 source generations differ")
        if not audit_doc.get("judge_metadata_comparable", True):
            issues.append("level1 and level2 textual judge metadata differ")

    qwen.to_csv(output / "qwen_native_evidence.csv", index=False)
    textual.to_csv(output / "qwen_textual_native_evidence.csv", index=False)
    qwen_kernel.to_csv(output / "qwen_kernel_evidence.csv", index=False)

    payload = {
        "status": {
            "passes": not issues,
            "strict": bool(args.strict),
        },
        "issues": issues,
        "inputs": {
            "level2_dir": str(level2),
            "level1_dir": str(args.level1_dir) if args.level1_dir else None,
            "qwen_results": str(qwen_results) if qwen_results else None,
            "textual_summary": str(textual_summary) if textual_summary else None,
            "textual_comparison": (
                str(textual_comparison) if textual_comparison else None
            ),
            "qwen_kernels": str(qwen_kernels) if qwen_kernels else None,
            "textual_audit": str(textual_audit) if textual_audit else None,
        },
        "requested_bits": args.bits,
        "thresholds": {
            "min_qwen_agreement": args.min_qwen_agreement,
            "max_qwen_perplexity_ratio": args.max_qwen_perplexity_ratio,
        },
        "textual": textual_evidence,
        "outputs": {
            "qwen": str(output / "qwen_native_evidence.csv"),
            "textual": str(output / "qwen_textual_native_evidence.csv"),
            "qwen_kernels": str(output / "qwen_kernel_evidence.csv"),
            "markdown": str(output / "native_evidence_audit.md"),
        },
    }
    (output / "native_evidence_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_markdown(output / "native_evidence_audit.md", payload)
    print(json.dumps(payload, indent=2))
    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
