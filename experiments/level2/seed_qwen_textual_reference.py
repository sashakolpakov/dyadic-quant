from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a Level 2 Qwen generations file with only the shared prompt "
            "set and source-reference generations from an existing Level 1 "
            "textual artifact."
        )
    )
    parser.add_argument(
        "--source-generations",
        type=Path,
        default=Path("results/level1/qwen25_generations.json"),
    )
    parser.add_argument(
        "--source-variant",
        default="bf16_source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/qwen25_dyop_generations.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_generations.read_text())
    if args.source_variant not in source["generations"]:
        raise RuntimeError(f"missing source variant: {args.source_variant}")
    seeded = {
        "prompts": source["prompts"],
        "generations": {
            args.source_variant: source["generations"][args.source_variant],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(seeded, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        f"Seeded {args.output} with {args.source_variant} and "
        f"{sum(len(items) for items in seeded['prompts'].values())} prompts"
    )


if __name__ == "__main__":
    main()
