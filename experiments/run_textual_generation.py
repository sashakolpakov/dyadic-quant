"""Capture free-running generated text for one model variant.

This is the generation half of the textual-output comparison. It loads one
model (the BF16 source, a dequantized GGUF control, and/or per-channel dyadic
prefixes derived from the source), generates greedy continuations for two
deterministic prompt families (ARC-Easy instructions and WikiText
continuations), and merges them into a shared generations JSON keyed by variant.

The semantic comparison (embedding cosine + LLM judge) is performed afterward by
``compare_generations.py`` against the recorded BF16 source variant.

MPS is mandatory; CPU fallback is intentionally disabled, matching the other
large-model scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dyadic_quant.dyadic_torch import encode_model, materialize_prefix
from dyadic_quant.textgen import (
    build_arc_prompts,
    build_wikitext_prompts,
    generate_texts,
    merge_generations,
)
from experiments.run_qwen_dyadic import load_wikitext_tokens, require_mps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--gguf-file",
        type=Path,
        help="Optional local GGUF blob to dequantize instead of safetensors.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        required=True,
        help="Generation key for the loaded model (e.g. bf16_source, q4_k_m).",
    )
    parser.add_argument(
        "--dyadic-prefix",
        help="If set (and not --reference-only), also generate dyadic prefixes "
        "named '<prefix>_<bits>'.",
    )
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Generate only from the loaded model; skip dyadic encoding.",
    )
    parser.add_argument(
        "--keep-embedding-fp16",
        action="store_true",
        help="Exclude the tied embedding/lm_head matrix from dyadic encoding.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Weights per power-of-two exponent (0 = one exponent per channel).",
    )
    parser.add_argument(
        "--embedding-bits",
        type=int,
        default=0,
        help="If set, materialize the embedding/lm_head at this deeper prefix.",
    )
    parser.add_argument(
        "--skip-reference-generation",
        action="store_true",
        help="Do not regenerate the loaded model's own text (only dyadic prefixes).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--arc-count", type=int, default=20)
    parser.add_argument("--wikitext-count", type=int, default=10)
    parser.add_argument("--wikitext-prefix-tokens", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--generations-file",
        type=Path,
        default=Path("results/qwen25_generations.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_mps()
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding is correct for generation

    model_kwargs = {
        "local_files_only": True,
        "dtype": model_dtype,
        "attn_implementation": "eager",
    }
    if args.gguf_file is not None:
        model_kwargs["gguf_file"] = str(args.gguf_file.resolve())
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, **model_kwargs
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

    def generate_all() -> dict[str, dict[str, str]]:
        return {
            family: generate_texts(
                model,
                tokenizer,
                device,
                prompts,
                max_new_tokens=args.max_new_tokens,
            )
            for family, prompts in prompts_by_family.items()
        }

    exclude_names = (
        {"model.embed_tokens", "lm_head"} if args.keep_embedding_fp16 else set()
    )
    overrides = (
        {"model.embed_tokens": args.embedding_bits, "lm_head": args.embedding_bits}
        if args.embedding_bits
        else {}
    )
    encoded = None
    if not args.reference_only and args.dyadic_prefix is not None:
        encoded = encode_model(
            model,
            max_bits=max(args.bits),
            optimize_prefix_bits=tuple(sorted(set(args.bits))),
            exclude_names=exclude_names,
            group_size=args.group_size or None,
        )

    model.to(device)
    if not args.skip_reference_generation:
        reference_generations = generate_all()
        merge_generations(
            args.generations_file,
            variant=args.variant,
            prompts_by_family=prompts_by_family,
            generations_by_family=reference_generations,
        )
        print(f"Generated '{args.variant}' for {list(prompts_by_family)}")

    if encoded is not None:
        for bits in args.bits:
            materialize_prefix(model, encoded, bits=bits, overrides=overrides)
            variant = f"{args.dyadic_prefix}_{bits}"
            merge_generations(
                args.generations_file,
                variant=variant,
                prompts_by_family=prompts_by_family,
                generations_by_family=generate_all(),
            )
            print(f"Generated '{variant}'")

    model.to("cpu")
    torch.mps.empty_cache()
    print(f"Wrote {args.generations_file}")


if __name__ == "__main__":
    main()
