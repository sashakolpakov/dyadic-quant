from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from time import perf_counter

import pandas as pd


def request_generate(model: str, prompt: str, *, num_predict: int) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 7,
                "num_predict": num_predict,
            },
            "keep_alive": "10m",
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def format_question(question: dict[str, object]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = "\n".join(
        f"{letters[index]}. {choice}"
        for index, choice in enumerate(question["choices"])
    )
    return (
        "Select the correct answer. Output only one capital letter, with no "
        f"explanation.\nQuestion: {question['question']}\n{choices}\nAnswer:"
    )


def parse_letter(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--arc-limit", type=int, default=50)
    parser.add_argument(
        "--result-prefix",
        default="ollama_qwen05b",
        help="Safe filename prefix for CSV and summary output.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = json.loads((args.data_dir / "arc_easy.json").read_text())[
        : args.arc_limit
    ]
    # Warm up and load the model onto Metal.
    request_generate(args.model, "Reply with OK.", num_predict=2)

    rows = []
    correct = 0
    valid = 0
    prompt_tokens = 0
    generated_tokens = 0
    prompt_ns = 0
    generation_ns = 0
    start = perf_counter()
    for question in questions:
        result = request_generate(
            args.model, format_question(question), num_predict=4
        )
        predicted = parse_letter(result["response"])
        expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(question["answer_index"])]
        valid += int(predicted is not None)
        correct += int(predicted == expected)
        prompt_tokens += int(result.get("prompt_eval_count", 0))
        generated_tokens += int(result.get("eval_count", 0))
        prompt_ns += int(result.get("prompt_eval_duration", 0))
        generation_ns += int(result.get("eval_duration", 0))
        rows.append(
            {
                "id": question["id"],
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "response": result["response"],
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "generated_tokens": result.get("eval_count", 0),
                "prompt_duration_ns": result.get("prompt_eval_duration", 0),
                "generation_duration_ns": result.get("eval_duration", 0),
            }
        )
    wall_s = perf_counter() - start
    summary = {
        "model": args.model,
        "questions": len(questions),
        "arc_easy_generation_accuracy": correct / len(questions),
        "valid_letter_rate": valid / len(questions),
        "prompt_tokens_per_s": prompt_tokens / (prompt_ns / 1e9),
        "generation_tokens_per_s": generated_tokens / (generation_ns / 1e9),
        "wall_elapsed_s": wall_s,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / f"{args.result_prefix}_arc_results.csv", index=False
    )
    (args.output_dir / f"{args.result_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
