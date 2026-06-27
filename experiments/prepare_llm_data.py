from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/llm_eval"))
    parser.add_argument("--arc-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wikitext = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    text = "\n\n".join(row["text"] for row in wikitext if row["text"].strip())
    (args.output_dir / "wikitext2_test.txt").write_text(text)

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    records = []
    for row in arc.select(range(min(args.arc_limit, len(arc)))):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer_index = labels.index(row["answerKey"])
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "choices": texts,
                "answer_index": answer_index,
                "answer_label": labels[answer_index],
            }
        )
    (args.output_dir / "arc_easy.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(f"Wrote WikiText-2 and {len(records)} ARC-Easy questions")


if __name__ == "__main__":
    main()

