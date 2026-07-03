from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level1 import load_encoded_model, materialize_prefix
from dyadic_quant.level2 import build_level2_model, build_native_cpu
from experiments.level2.common import require_speed_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen logits from Level 1 materialized dyadic weights "
            "against Level 2 native dyop execution for the same packed artifact."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dyadic", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--prompt", default="Answer briefly: what is photosynthesis?")
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument("--generate-steps", type=int, default=0)
    parser.add_argument("--module-limit", type=int, default=0)
    parser.add_argument(
        "--qwen-mlp-backend",
        choices=["torch", "native-cpu-plan"],
        default="torch",
        help="Fuse Qwen MLP projections into a reusable native packed plan.",
    )
    parser.add_argument(
        "--qwen-norm-backend",
        choices=["torch", "native-cpu"],
        default="torch",
        help="Replace Qwen RMSNorm modules with native CPU execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/level2/qwen_logit_equivalence.json"),
    )
    parser.add_argument(
        "--speed-gates",
        type=Path,
        default=Path("results/level2/subkernel_speed_gates_arm64_neon_latest.csv"),
        help="CSV proving required Qwen native dyop kernels beat materialized gates.",
    )
    parser.add_argument(
        "--skip-speed-gate-check",
        action="store_true",
        help="Compare logits even when native speed gates are incomplete or failing.",
    )
    return parser.parse_args()


def tensor_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
    }


def flatten_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def main() -> None:
    args = parse_args()
    if not args.skip_speed_gate_check:
        require_speed_gates(args.speed_gates, "qwen")
    build_native_cpu()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    encoded = load_encoded_model(args.dyadic)

    source = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    level1 = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    materialize_ms = materialize_prefix(level1, encoded, bits=args.bits)
    start = perf_counter()
    level2, replacement = build_level2_model(
        source,
        encoded,
        bits=args.bits,
        dtype=torch.float32,
        linear_backend="native-cpu",
        embedding_backend="native-cpu",
        qwen_mlp_backend=args.qwen_mlp_backend,
        qwen_norm_backend=args.qwen_norm_backend,
    )
    level2_build_ms = (perf_counter() - start) * 1000
    level2.eval()

    names = list(replacement.replaced_modules)
    if args.module_limit > 0:
        names = names[: args.module_limit]
    level1_outputs: dict[str, torch.Tensor] = {}
    module_rows: list[dict[str, object]] = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            level1_outputs[name] = flatten_output(output).detach().float().cpu()

        return hook

    def compare(name: str):
        def hook(_module, _inputs, output):
            actual = flatten_output(output).detach().float().cpu()
            expected = level1_outputs[name]
            stats = tensor_stats(actual, expected)
            module_rows.append(
                {
                    "name": name,
                    "shape": list(actual.shape),
                    **stats,
                    "allclose_rtol1e-5_atol1e-5": bool(
                        torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
                    ),
                }
            )

        return hook

    handles = []
    for name in names:
        handles.append(level1.get_submodule(name).register_forward_hook(capture(name)))
    with torch.inference_mode():
        l1_logits = level1(**inputs, use_cache=False).logits.detach().float().cpu()
    for handle in handles:
        handle.remove()

    handles = []
    for name in names:
        handles.append(level2.get_submodule(name).register_forward_hook(compare(name)))
    with torch.inference_mode():
        l2_logits = level2(**inputs, use_cache=False).logits.detach().float().cpu()
    for handle in handles:
        handle.remove()

    logit_stats = tensor_stats(l2_logits, l1_logits)
    generation_rows: list[dict[str, object]] = []
    if args.generate_steps > 0:
        l1_ids = inputs.input_ids.clone()
        l2_ids = inputs.input_ids.clone()
        with torch.inference_mode():
            for step in range(args.generate_steps):
                l1_step_logits = level1(input_ids=l1_ids, use_cache=False).logits[:, -1]
                l2_step_logits = level2(input_ids=l2_ids, use_cache=False).logits[:, -1]
                l1_next = int(l1_step_logits.argmax(dim=-1).item())
                l2_next = int(l2_step_logits.argmax(dim=-1).item())
                stats = tensor_stats(l2_step_logits.cpu(), l1_step_logits.cpu())
                generation_rows.append(
                    {
                        "step": step,
                        "level1_token": l1_next,
                        "level2_token": l2_next,
                        "tokens_equal": l1_next == l2_next,
                        "level1_text": tokenizer.decode([l1_next]),
                        "level2_text": tokenizer.decode([l2_next]),
                        **stats,
                    }
                )
                l1_ids = torch.cat(
                    [l1_ids, torch.tensor([[l1_next]], dtype=l1_ids.dtype)], dim=1
                )
                l2_ids = torch.cat(
                    [l2_ids, torch.tensor([[l2_next]], dtype=l2_ids.dtype)], dim=1
                )
    first_bad = next(
        (
            row
            for row in module_rows
            if not row["allclose_rtol1e-5_atol1e-5"]
        ),
        None,
    )
    result = {
        "bits": args.bits,
        "prompt": args.prompt,
        "input_tokens": int(inputs.input_ids.numel()),
        "dyadic": str(args.dyadic),
        "materialize_ms": materialize_ms,
        "level2_build_ms": level2_build_ms,
        "replaced_modules": len(replacement.replaced_modules),
        "shared_weight_modules": list(replacement.shared_weight_modules),
        "fused_modules": list(replacement.fused_modules),
        "logits": {
            **logit_stats,
            "allclose_rtol1e-5_atol1e-5": bool(
                torch.allclose(l2_logits, l1_logits, rtol=1e-5, atol=1e-5)
            ),
            "argmax_equal_rate": float(
                (l2_logits.argmax(dim=-1) == l1_logits.argmax(dim=-1))
                .float()
                .mean()
                .item()
            ),
        },
        "generation": {
            "steps": args.generate_steps,
            "tokens_all_equal": all(row["tokens_equal"] for row in generation_rows),
            "first_divergence": next(
                (row for row in generation_rows if not row["tokens_equal"]),
                None,
            ),
            "rows": generation_rows,
        },
        "first_bad_module": first_bad,
        "module_rows": module_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("bits", "logits", "generation", "first_bad_module")},
            indent=2,
        )
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
