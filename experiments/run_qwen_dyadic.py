from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dyadic_quant.dyadic_torch import (
    encode_model,
    materialize_prefix,
    save_encoded_model,
    storage_bytes,
)


def require_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is disabled")
    return torch.device("mps")


def synchronize() -> None:
    torch.mps.synchronize()


def warmup_model(model, token_ids: torch.Tensor, device: torch.device) -> None:
    sample = token_ids[:64].unsqueeze(0).to(device)
    with torch.inference_mode():
        model(input_ids=sample, use_cache=False)
        model(input_ids=sample, use_cache=False)
    synchronize()


def load_wikitext_tokens(
    tokenizer, path: Path, *, max_tokens: int
) -> torch.Tensor:
    text = path.read_text()
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    ).input_ids[0]


def evaluate_language_model(
    model,
    token_ids: torch.Tensor,
    device: torch.device,
    *,
    sequence_length: int,
    reference_predictions: torch.Tensor | None,
) -> tuple[dict[str, float | int], torch.Tensor]:
    total_loss = 0.0
    total_tokens = 0
    predictions: list[torch.Tensor] = []
    synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(token_ids) - 1, sequence_length):
            chunk = token_ids[offset : offset + sequence_length + 1]
            if len(chunk) < 2:
                continue
            inputs = chunk[:-1].unsqueeze(0).to(device)
            targets = chunk[1:].unsqueeze(0).to(device)
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += targets.numel()
            predictions.append(logits.argmax(dim=-1).cpu())
    synchronize()
    elapsed = perf_counter() - start
    predicted = torch.cat(predictions, dim=1).squeeze(0)
    mean_loss = total_loss / total_tokens
    agreement = (
        float((predicted == reference_predictions).float().mean().item())
        if reference_predictions is not None
        else 1.0
    )
    return (
        {
            "cross_entropy": mean_loss,
            "perplexity": math.exp(min(mean_loss, 80.0)),
            "next_token_agreement": agreement,
            "evaluated_tokens": total_tokens,
            "wikitext_elapsed_s": elapsed,
            "wikitext_tokens_per_s": total_tokens / elapsed,
        },
        predicted,
    )


def format_arc_prompt(question: dict[str, object]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = "\n".join(
        f"{letters[index]}. {choice}"
        for index, choice in enumerate(question["choices"])
    )
    return (
        "Choose the correct answer. Respond with the answer choice.\n"
        f"Question: {question['question']}\n{choices}\nAnswer:"
    )


def score_arc(
    model,
    tokenizer,
    questions: list[dict[str, object]],
    device: torch.device,
) -> tuple[float, float]:
    correct = 0
    synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for question in questions:
            prompt = format_arc_prompt(question)
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            ).input_ids
            sequences = []
            continuation_starts = []
            for choice in question["choices"]:
                continuation = " " + str(choice)
                continuation_ids = tokenizer(
                    continuation, add_special_tokens=False
                ).input_ids
                sequences.append(prompt_ids + continuation_ids)
                continuation_starts.append(len(prompt_ids))
            encoded = tokenizer.pad(
                {"input_ids": sequences},
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            targets = input_ids[:, 1:]
            token_scores = log_probs.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1).cpu()
            scores = []
            for index, start_index in enumerate(continuation_starts):
                end_index = int(attention_mask[index].sum().item()) - 1
                continuation_score = token_scores[
                    index, start_index - 1 : end_index
                ].mean()
                scores.append(float(continuation_score.item()))
            if max(range(len(scores)), key=scores.__getitem__) == int(
                question["answer_index"]
            ):
                correct += 1
    synchronize()
    return correct / len(questions), perf_counter() - start


def generation_speed(model, tokenizer, device: torch.device) -> dict[str, float]:
    prompt = (
        "Explain in one concise paragraph why low-bit quantization can speed "
        "up neural network inference."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
        synchronize()
        start = perf_counter()
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        synchronize()
    elapsed = perf_counter() - start
    generated = output.shape[1] - inputs.input_ids.shape[1]
    return {
        "generated_tokens": generated,
        "generation_elapsed_s": elapsed,
        "generation_tokens_per_s": generated / elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--gguf-file",
        type=Path,
        help="Optional local Ollama/GGUF blob to dequantize instead of safetensors.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--arc-limit", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Common model evaluation dtype; BF16 matches the official source.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Evaluate the loaded source/GGUF model without creating dyadic prefixes.",
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
        help="Weights per power-of-two exponent (0 = one exponent per output "
        "channel). Smaller groups give finer scales at a small storage cost.",
    )
    parser.add_argument(
        "--embedding-bits",
        type=int,
        default=0,
        help="If set, materialize the tied embedding/lm_head at this deeper "
        "prefix width (mixed precision) instead of --bits.",
    )
    parser.add_argument(
        "--variant-name",
        default="qwen05b",
        help="Safe name used for result and metadata filenames.",
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        help="Optional saved original-source next-token predictions for agreement.",
    )
    parser.add_argument(
        "--save-reference-predictions",
        type=Path,
        help="Save this run's source/reference next-token predictions.",
    )
    parser.add_argument(
        "--save-dyadic",
        type=Path,
        help="Serialize the packed maximum-depth nested dyadic code.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_mps()
    model_dtype = (
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
    questions = json.loads((args.data_dir / "arc_easy.json").read_text())[
        : args.arc_limit
    ]
    external_reference_predictions = (
        torch.load(args.reference_predictions, map_location="cpu", weights_only=True)
        if args.reference_predictions is not None
        else None
    )

    exclude_names = (
        {"model.embed_tokens", "lm_head"} if args.keep_embedding_fp16 else set()
    )
    group_size = args.group_size or None
    # Mixed precision: keep the tied embedding/output matrix at a deeper prefix.
    overrides = (
        {"model.embed_tokens": args.embedding_bits, "lm_head": args.embedding_bits}
        if args.embedding_bits
        else {}
    )
    encoded = None
    if not args.reference_only:
        encoded = encode_model(
            model,
            max_bits=max(args.bits),
            optimize_prefix_bits=tuple(sorted(set(args.bits))),
            exclude_names=exclude_names,
            group_size=group_size,
        )
        if args.save_dyadic is not None:
            save_encoded_model(encoded, args.save_dyadic)
    rows = []

    model.to(device)
    warmup_model(model, token_ids, device)
    reference_metrics, reference_predictions = evaluate_language_model(
        model,
        token_ids,
        device,
        sequence_length=args.sequence_length,
        reference_predictions=external_reference_predictions,
    )
    if args.save_reference_predictions is not None:
        args.save_reference_predictions.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference_predictions, args.save_reference_predictions)
    arc_accuracy, arc_elapsed = score_arc(model, tokenizer, questions, device)
    reference_generation = generation_speed(model, tokenizer, device)
    reference_bytes = sum(
        tensor.numel() * 2
        for tensor in list(model.parameters()) + list(model.buffers())
    )
    rows.append(
        {
            "method": (
                "dequantized_gguf_reference"
                if args.gguf_file is not None
                else f"{args.dtype}_source_reference"
            ),
            "bits_per_weight": 16,
            "conversion_ms": 0.0,
            "materialization_ms": 0.0,
            "total_model_bytes": reference_bytes,
            "incremental_plane_bytes": 0,
            "arc_easy_accuracy": arc_accuracy,
            "arc_elapsed_s": arc_elapsed,
            **reference_metrics,
            **reference_generation,
        }
    )
    model.to("cpu")
    torch.mps.empty_cache()

    for bits in [] if args.reference_only else args.bits:
        assert encoded is not None
        materialization_ms = materialize_prefix(
            model, encoded, bits=bits, overrides=overrides
        )
        sizes = storage_bytes(model, encoded, bits=bits, overrides=overrides)
        model.to(device)
        warmup_model(model, token_ids, device)
        metrics, _ = evaluate_language_model(
            model,
            token_ids,
            device,
            sequence_length=args.sequence_length,
            reference_predictions=reference_predictions,
        )
        arc_accuracy, arc_elapsed = score_arc(model, tokenizer, questions, device)
        generation = generation_speed(model, tokenizer, device)
        rows.append(
            {
                "method": (
                    "per_channel_dyadic"
                    if group_size is None
                    else "block_dyadic"
                ),
                "bits_per_weight": bits,
                "group_size": args.group_size,
                "embedding_bits": args.embedding_bits or bits,
                "conversion_ms": encoded.conversion_ms,
                "materialization_ms": materialization_ms,
                "arc_easy_accuracy": arc_accuracy,
                "arc_elapsed_s": arc_elapsed,
                "effective_bits_per_weight": sizes["total_model_bytes"]
                * 8
                / sum(p.numel() for p in model.parameters()),
                **sizes,
                **metrics,
                **generation,
            }
        )
        print(
            f"{bits}-bit: ppl={metrics['perplexity']:.3f}, "
            f"agreement={metrics['next_token_agreement']:.3f}, "
            f"ARC={arc_accuracy:.3f}"
        )
        model.to("cpu")
        torch.mps.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    result_path = args.output_dir / f"{args.variant_name}_results.csv"
    frame.to_csv(result_path, index=False)
    metadata = {
        "arguments": vars(args) | {
            "model_dir": str(args.model_dir),
            "gguf_file": str(args.gguf_file) if args.gguf_file else None,
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "reference_predictions": (
                str(args.reference_predictions)
                if args.reference_predictions is not None
                else None
            ),
            "save_reference_predictions": (
                str(args.save_reference_predictions)
                if args.save_reference_predictions is not None
                else None
            ),
            "save_dyadic": (
                str(args.save_dyadic) if args.save_dyadic is not None else None
            ),
        },
        "device": str(device),
        "torch": torch.__version__,
        "evaluation_dtype": args.dtype,
        "platform": platform.platform(),
        "quantized_weight_count": (
            encoded.quantized_weight_count if encoded is not None else 0
        ),
        "exponent_count": encoded.exponent_count if encoded is not None else 0,
        "excluded_modules": sorted(exclude_names),
        "reference_only": args.reference_only,
        "reference_note": (
            f"The GGUF source is dequantized to {args.dtype} for Transformers/MPS "
            "evaluation before dyadic encoding."
            if args.gguf_file is not None
            else f"The source checkpoint is evaluated directly in {args.dtype}."
        ),
    }
    (args.output_dir / f"{args.variant_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(frame.to_string(index=False))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
