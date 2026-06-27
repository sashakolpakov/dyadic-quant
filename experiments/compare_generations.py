"""Compare variant generations against the BF16 source, textually and semantically.

For every prompt, each non-source variant's generated text is compared to the
BF16 source's text along four axes:

* exact_match  - byte-identical after stripping (cheap local signal)
* edit_ratio   - normalized character similarity (cheap local signal)
* token_jaccard- whitespace-token set overlap (cheap local signal)
* cosine       - embedding cosine via a local Ollama nomic-embed-text model
* judge_*      - same-meaning verdict from a headless Claude Code judge

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

from padic_quant.textgen import (
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


def run_judge(prompt: str, *, model: str | None) -> dict:
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
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if completed.returncode != 0:
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
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (compute only lexical + cosine metrics).",
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
                    verdicts = run_judge(
                        judge_prompt(
                            instruction=instruction_by[(family, prompt_id)],
                            reference=reference,
                            variants=differing,
                        ),
                        model=args.judge_model,
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
            print(f"{family}/{prompt_id}: compared {len(present)} variants")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    per_row_path = args.output_dir / "qwen25_textual_comparison.csv"
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
    print(summary.to_string(index=False))
    print(f"Wrote {per_row_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
