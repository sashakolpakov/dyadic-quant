"""Compare variant generations against the BF16 source, textually and semantically.

For every prompt, each non-source variant's generated text is compared to the
BF16 source's text along four axes:

* exact_match  - byte-identical after stripping (cheap local signal)
* edit_ratio   - normalized character similarity (cheap local signal)
* token_jaccard- whitespace-token set overlap (cheap local signal)
* cosine       - embedding cosine via a local Ollama nomic-embed-text model
* judge_*      - same-meaning verdict from a headless Claude/Ollama judge

The lexical signals catch verbatim drift; cosine and the judge address the case
the user cares about: outputs that differ on the surface but mean the same thing.
The judge evaluates all variants for one prompt in a single call against the
source, both to cut cost and to give the judge a consistent frame of reference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dyadic_quant.textgen import (
    cosine_similarity,
    edit_ratio,
    exact_match,
    token_jaccard,
)


def embed(text: str, *, model: str) -> list[float]:
    payload = json.dumps({"model": model, "prompt": text}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["embedding"]


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in judge response: {text[:200]!r}")
    return json.loads(candidate[start : end + 1])


def judge_prompt(
    *, instruction: str, reference: str, variants: dict[str, str]
) -> str:
    blocks = "\n\n".join(
        f"VARIANT {name}:\n{text}" for name, text in variants.items()
    )
    names = ", ".join(variants)
    return (
        "You are judging whether quantized model outputs preserve the MEANING "
        "of a reference output. Two texts are equivalent if a careful reader "
        "would draw the same conclusions and factual content from them, even if "
        "the wording, length, or formatting differs. Disfluency or truncation "
        "alone does not break equivalence; a different answer, a contradicted "
        "fact, or degenerate/empty text does.\n\n"
        f"PROMPT GIVEN TO THE MODELS:\n{instruction}\n\n"
        f"REFERENCE OUTPUT:\n{reference}\n\n"
        f"{blocks}\n\n"
        f"For each of these variants [{names}], decide if it is meaning-"
        "equivalent to the REFERENCE. Respond with ONLY a JSON object mapping "
        'each variant name to {"equivalent": true|false, "reason": "<short>"}. '
        "No other text."
    )


def chunk_items(items: dict[str, str], size: int) -> list[dict[str, str]]:
    if size <= 0:
        return [items]
    pairs = list(items.items())
    return [dict(pairs[index : index + size]) for index in range(0, len(pairs), size)]


def normalize_judge_verdicts(
    verdicts: dict, expected_variants: set[str]
) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for raw_name, verdict in verdicts.items():
        name = str(raw_name).strip()
        if name.startswith("VARIANT "):
            name = name.removeprefix("VARIANT ").strip()
        if name not in expected_variants and len(expected_variants) == 1:
            name = next(iter(expected_variants))
        if name in expected_variants and isinstance(verdict, dict):
            normalized[name] = verdict
        elif name in expected_variants and isinstance(verdict, bool):
            normalized[name] = {"equivalent": verdict, "reason": ""}
        elif name in expected_variants and isinstance(verdict, str):
            lowered = verdict.strip().lower()
            if lowered in {"true", "false"}:
                normalized[name] = {
                    "equivalent": lowered == "true",
                    "reason": str(verdicts.get("reason", "")),
                }
    if not normalized and len(expected_variants) == 1:
        expected = next(iter(expected_variants))
        equivalent: bool | None = None
        reason = ""
        for raw_name, verdict in verdicts.items():
            if str(raw_name).strip().lower() == "reason":
                reason = str(verdict)
            if isinstance(verdict, bool):
                equivalent = verdict
            elif isinstance(verdict, str) and verdict.strip().lower() in {"true", "false"}:
                equivalent = verdict.strip().lower() == "true"
            elif isinstance(raw_name, str) and raw_name.strip().lower() in {"true", "false"}:
                equivalent = raw_name.strip().lower() == "true"
                reason = str(verdict)
        if equivalent is not None:
            normalized[expected] = {"equivalent": equivalent, "reason": reason}
    return normalized


def run_claude_judge(prompt: str, *, model: str | None, timeout: float) -> dict:
    # --max-turns 1 keeps the headless agent from entering a tool-use loop and
    # forces a single direct answer. The session default model answers cleanly;
    # an explicit --model can be slower and more prone to agentic refusals.
    command = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
    if model:
        command += ["--model", model]
    # Firing many calls back-to-back occasionally hits transient API errors or
    # rate limits; retry with backoff before giving up on a prompt.
    last_error: Exception | None = None
    for attempt in range(4):
        if attempt:
            time.sleep(2 * attempt)
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            last_error = RuntimeError(f"judge timed out after {timeout:.0f}s")
            continue
        if completed.returncode != 0:
            try:
                envelope = json.loads(completed.stdout)
                status = envelope.get("api_error_status")
                result = str(envelope.get("result", ""))[:200]
                last_error = RuntimeError(
                    f"claude exited {completed.returncode}: "
                    f"api_error_status={status} result={result}"
                )
                continue
            except (ValueError, TypeError):
                pass
            last_error = RuntimeError(
                f"claude exited {completed.returncode}: {completed.stderr[:200]}"
            )
            continue
        try:
            envelope = json.loads(completed.stdout)
            if envelope.get("is_error"):
                last_error = RuntimeError(
                    f"judge api error: {envelope.get('api_error_status')}"
                )
                continue
            return extract_json(envelope["result"])
        except (ValueError, KeyError) as error:
            last_error = error
            continue
    raise last_error if last_error else RuntimeError("judge failed")


def run_ollama_judge(prompt: str, *, model: str, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        document = json.loads(response.read())
    if "response" not in document:
        raise RuntimeError(f"ollama response missing text: {document}")
    return extract_json(document["response"])


def run_judge(
    prompt: str, *, backend: str, model: str | None, timeout: float
) -> dict:
    if backend == "claude":
        return run_claude_judge(prompt, model=model, timeout=timeout)
    if backend == "ollama":
        if not model:
            raise RuntimeError("--judge-model is required for --judge-backend ollama")
        return run_ollama_judge(prompt, model=model, timeout=timeout)
    raise RuntimeError(f"unknown judge backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations-file",
        type=Path,
        default=Path("results/qwen25_generations.json"),
    )
    parser.add_argument("--source-variant", default="bf16_source")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional model override for the judge; empty uses the session default.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["claude", "ollama"],
        default="claude",
        help="LLM judge backend. Ollama uses the local /api/generate endpoint.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (compute only lexical + cosine metrics).",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=180.0,
        help="Per-prompt Claude judge timeout in seconds.",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=0,
        help=(
            "Maximum variants per judge call. 0 judges all variants for a prompt "
            "in one call; use 1 for weaker local judges."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of non-source variants to compare.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.generations_file.read_text())
    prompts = document["prompts"]
    generations = document["generations"]
    source = args.source_variant
    if source not in generations:
        raise RuntimeError(f"source variant '{source}' not in generations file")
    variants = [name for name in generations if name != source]
    if args.variants is not None:
        missing = sorted(set(args.variants) - set(variants))
        if missing:
            raise RuntimeError(f"requested variants missing from generations: {missing}")
        variants = [name for name in variants if name in set(args.variants)]

    embedding_cache: dict[str, list[float]] = {}

    def embedding(text: str) -> list[float]:
        if not text.strip():
            # A degenerate empty generation has no embedding; treat it as
            # maximally dissimilar (cosine 0 via a zero vector).
            return []
        if text not in embedding_cache:
            embedding_cache[text] = embed(text, model=args.embed_model)
        return embedding_cache[text]

    # Index prompt instruction text by (family, id) for the judge frame.
    instruction_by: dict[tuple[str, str], str] = {}
    for family, items in prompts.items():
        for item in items:
            instruction_by[(family, item["id"])] = item["prompt"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_row_path = args.output_dir / "qwen25_textual_comparison.csv"
    rows: list[dict[str, object]] = []
    for family in prompts:
        prompt_ids = [item["id"] for item in prompts[family]]
        for prompt_id in prompt_ids:
            reference = generations[source][family][prompt_id]
            present = {
                name: generations[name][family][prompt_id]
                for name in variants
                if prompt_id in generations[name].get(family, {})
            }
            verdicts: dict[str, dict] = {}
            # Only ask the judge about variants that actually differ; identical
            # text is trivially equivalent and would waste a judge call.
            differing = {
                name: text
                for name, text in present.items()
                if not exact_match(reference, text)
            }
            if not args.no_judge and differing:
                try:
                    print(
                        f"{family}/{prompt_id}: judging {len(differing)} variants",
                        flush=True,
                    )
                    for batch in chunk_items(differing, args.judge_batch_size):
                        batch_verdicts = run_judge(
                            judge_prompt(
                                instruction=instruction_by[(family, prompt_id)],
                                reference=reference,
                                variants=batch,
                            ),
                            backend=args.judge_backend,
                            model=args.judge_model,
                            timeout=args.judge_timeout,
                        )
                        verdicts.update(
                            normalize_judge_verdicts(
                                batch_verdicts,
                                set(batch),
                            )
                        )
                except (subprocess.SubprocessError, ValueError, KeyError, RuntimeError) as error:
                    print(f"  judge failed for {family}/{prompt_id}: {error}")
                    verdicts = {}
            for name, text in present.items():
                identical = exact_match(reference, text)
                verdict = verdicts.get(name, {})
                rows.append(
                    {
                        "family": family,
                        "prompt_id": prompt_id,
                        "variant": name,
                        "exact_match": identical,
                        "edit_ratio": edit_ratio(reference, text),
                        "token_jaccard": token_jaccard(reference, text),
                        "cosine": cosine_similarity(
                            embedding(reference), embedding(text)
                        ),
                        "judge_equivalent": (
                            True
                            if identical
                            else verdict.get("equivalent")
                            if not args.no_judge
                            else None
                        ),
                        "judge_reason": (
                            "identical text"
                            if identical
                            else verdict.get("reason", "")
                            if not args.no_judge
                            else ""
                        ),
                    }
                )
            pd.DataFrame(rows).to_csv(per_row_path, index=False)
            print(f"{family}/{prompt_id}: compared {len(present)} variants")

    frame = pd.DataFrame(rows)
    frame.to_csv(per_row_path, index=False)

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
                "judge_equivalent_rate": (
                    judged.mean() if len(judged) else None
                ),
                "judged_prompts": len(judged),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "qwen25_textual_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "generations_file": str(args.generations_file),
        "source_variant": args.source_variant,
        "variants": variants,
        "embed_model": args.embed_model,
        "judge_backend": args.judge_backend,
        "judge_model": (
            args.judge_model
            if args.judge_model
            else f"{args.judge_backend}_default"
        ),
        "judge_model_arg": args.judge_model,
        "judge_timeout": args.judge_timeout,
        "judge_batch_size": args.judge_batch_size,
        "judge_enabled": not args.no_judge,
        "output_dir": str(args.output_dir),
        "row_count": len(frame),
        "missing_judge_equivalent": (
            int(frame["judge_equivalent"].isna().sum())
            if "judge_equivalent" in frame
            else None
        ),
    }
    metadata_path = args.output_dir / "qwen25_textual_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {per_row_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
