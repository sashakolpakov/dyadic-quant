"""Generate Qwen textual outputs through Level 2 native dyop modules.

This is the Level 2 counterpart to ``experiments/run_textual_generation.py``.
It keeps the textual artifact separate from Level 1 materialized generations:
the source model may be generated once for comparison, while dyadic variants
are built with ``build_level2_model`` from packed sign/magnitude tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.dyadic_torch import encode_model, load_encoded_model, save_encoded_model
from dyadic_quant.level2 import build_level2_model, build_native_cpu
from dyadic_quant.textgen import (
    build_arc_prompts,
    build_wikitext_prompts,
    generate_texts,
    merge_generations,
)
from experiments.run_qwen_dyadic import load_wikitext_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--embedding-bits", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--arc-count", type=int, default=20)
    parser.add_argument("--wikitext-count", type=int, default=10)
    parser.add_argument("--wikitext-prefix-tokens", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--source-variant", default="float32_source")
    parser.add_argument("--dyop-prefix", default="dyop_native")
    parser.add_argument(
        "--skip-source-generation",
        action="store_true",
        help="Only append Level 2 dyop variants to an existing generations file.",
    )
    parser.add_argument(
        "--load-dyadic",
        type=Path,
        help="Load a packed Level 1 dyadic artifact instead of encoding the source.",
    )
    parser.add_argument(
        "--save-dyadic",
        type=Path,
        help="Write the packed Level 1 dyadic artifact used by this Level 2 run.",
    )
    parser.add_argument(
        "--linear-backend",
        choices=["native-cpu"],
        default="native-cpu",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["native-cpu"],
        default="native-cpu",
    )
    parser.add_argument(
        "--generations-file",
        type=Path,
        default=Path("results/level2/qwen25_dyop_generations.json"),
    )
    return parser.parse_args()


def generate_all(
    model,
    tokenizer,
    device: torch.device,
    prompts_by_family: dict[str, list[dict[str, str]]],
    *,
    max_new_tokens: int,
) -> dict[str, dict[str, str]]:
    return {
        family: generate_texts(
            model,
            tokenizer,
            device,
            prompts,
            max_new_tokens=max_new_tokens,
        )
        for family, prompts in prompts_by_family.items()
    }


def main() -> None:
    args = parse_args()
    if args.linear_backend == "native-cpu" or args.embedding_backend == "native-cpu":
        build_native_cpu()

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    token_ids = load_wikitext_tokens(
        tokenizer,
        args.data_dir / "wikitext2_test.txt",
        max_tokens=args.max_tokens,
    )
    questions = json.loads((args.data_dir / "arc_easy.json").read_text())
    prompts_by_family = {
        "arc": build_arc_prompts(tokenizer, questions, count=args.arc_count),
        "wikitext": build_wikitext_prompts(
            tokenizer,
            token_ids,
            count=args.wikitext_count,
            prefix_tokens=args.wikitext_prefix_tokens,
        ),
    }

    args.generations_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_source_generation:
        model.to(device)
        merge_generations(
            args.generations_file,
            variant=args.source_variant,
            prompts_by_family=prompts_by_family,
            generations_by_family=generate_all(
                model,
                tokenizer,
                device,
                prompts_by_family,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        print(f"Generated '{args.source_variant}'")

    encoded = (
        load_encoded_model(args.load_dyadic)
        if args.load_dyadic is not None
        else encode_model(
            model,
            max_bits=max(args.bits),
            optimize_prefix_bits=tuple(sorted(set(args.bits))),
            group_size=args.group_size or None,
        )
    )
    if args.save_dyadic is not None:
        save_encoded_model(encoded, args.save_dyadic)

    overrides = (
        {"model.embed_tokens": args.embedding_bits, "lm_head": args.embedding_bits}
        if args.embedding_bits
        else {}
    )
    model.to("cpu")
    for bits in args.bits:
        candidate, replacement = build_level2_model(
            model,
            encoded,
            bits=bits,
            overrides=overrides,
            dtype=torch.float32,
            linear_backend=args.linear_backend,
            embedding_backend=args.embedding_backend,
            copy_model=True,
        )
        candidate.eval().to(device)
        variant = f"{args.dyop_prefix}_{bits}"
        merge_generations(
            args.generations_file,
            variant=variant,
            prompts_by_family=prompts_by_family,
            generations_by_family=generate_all(
                candidate,
                tokenizer,
                device,
                prompts_by_family,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        print(
            f"Generated '{variant}' with {len(replacement.replaced_modules)} "
            "Level 2 modules"
        )
        del candidate

    print(f"Wrote {args.generations_file}")


if __name__ == "__main__":
    main()
