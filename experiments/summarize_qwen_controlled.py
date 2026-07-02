from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/level1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.results_dir
    lineage = json.loads(
        (directory / "qwen25_control_lineage.json").read_text()
    )
    rows: list[dict[str, object]] = []

    def ollama_metrics(prefix: str) -> dict[str, object]:
        path = directory / f"{prefix}_summary.json"
        if not path.exists():
            return {}
        summary = json.loads(path.read_text())
        return {
            "ollama_arc_generation_accuracy": summary[
                "arc_easy_generation_accuracy"
            ],
            "ollama_valid_letter_rate": summary["valid_letter_rate"],
            "ollama_prompt_tokens_per_s": summary["prompt_tokens_per_s"],
            "ollama_generation_tokens_per_s": summary[
                "generation_tokens_per_s"
            ],
        }

    dyadic = pd.read_csv(directory / "qwen25_bf16_dyadic_results.csv")
    for _, row in dyadic.iterrows():
        bits = int(row["bits_per_weight"])
        rows.append(
            {
                "variant": "bf16_source" if bits == 16 else f"dyadic_{bits}",
                "family": "source" if bits == 16 else "progressive_dyadic",
                "bits_per_weight": bits,
                "storage_bytes": (
                    lineage["source"]["size_bytes"]
                    if bits == 16
                    else int(row["total_model_bytes"])
                ),
                "perplexity": row["perplexity"],
                "next_token_agreement_with_source": row[
                    "next_token_agreement"
                ],
                "arc_easy_likelihood_accuracy": row["arc_easy_accuracy"],
                "transformers_generation_tokens_per_s": row[
                    "generation_tokens_per_s"
                ],
                # The BF16 source carries a native-Ollama baseline; the
                # dequantized dyadic prefixes have no native backend yet.
                **(
                    ollama_metrics("ollama_qwen25_bf16_source")
                    if bits == 16
                    else {}
                ),
            }
        )

    mapping = {
        "q4_k_m": "q4_k_m",
        "q6_k": "q6_k",
        "q8_0": "q8_0",
    }
    for key, variant_name in mapping.items():
        frame = pd.read_csv(directory / f"qwen25_{key}_results.csv")
        row = frame.iloc[0]
        rows.append(
            {
                "variant": variant_name,
                "family": "gguf",
                "bits_per_weight": None,
                "storage_bytes": lineage["variants"][key]["size_bytes"],
                "perplexity": row["perplexity"],
                "next_token_agreement_with_source": row[
                    "next_token_agreement"
                ],
                "arc_easy_likelihood_accuracy": row["arc_easy_accuracy"],
                "transformers_generation_tokens_per_s": row[
                    "generation_tokens_per_s"
                ],
                **ollama_metrics(f"ollama_qwen25_{key}"),
            }
        )

    frame = pd.DataFrame(rows)
    output = directory / "qwen25_controlled_comparison.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
