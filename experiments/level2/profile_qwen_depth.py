from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level1 import encode_model, load_encoded_model
from dyadic_quant.level2 import build_level2_model, build_native_cpu
from dyadic_quant.level2.modules import DyadicEmbedding, DyadicLinear


@dataclass
class Timing:
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)


def shape_of(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    if isinstance(value, (tuple, list)) and value:
        return shape_of(value[0])
    return ""


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def register_timers(model: nn.Module) -> tuple[list[torch.utils.hooks.RemovableHandle], dict[tuple[str, str, str], Timing]]:
    timings: dict[tuple[str, str, str], Timing] = defaultdict(Timing)
    starts: dict[int, list[tuple[float, str]]] = defaultdict(list)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def should_time(name: str, module: nn.Module) -> bool:
        if isinstance(module, (DyadicLinear, DyadicEmbedding)):
            return True
        if name.startswith("model.layers.") and name.count(".") == 2:
            return True
        return False

    for name, module in model.named_modules():
        if not should_time(name, module):
            continue

        def pre_hook(mod: nn.Module, inputs: tuple[object, ...], *, module_name: str = name) -> None:
            starts[id(mod)].append((perf_counter(), shape_of(inputs)))

        def post_hook(
            mod: nn.Module,
            inputs: tuple[object, ...],
            output: object,
            *,
            module_name: str = name,
        ) -> None:
            start, input_shape = starts[id(mod)].pop()
            elapsed_ms = (perf_counter() - start) * 1000.0
            key = (module_name, type(mod).__name__, input_shape)
            timings[key].add(elapsed_ms)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    return handles, timings


def forward_once(model: nn.Module, input_ids: torch.Tensor, device: torch.device) -> float:
    synchronize_device(device)
    start = perf_counter()
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    synchronize_device(device)
    return (perf_counter() - start) * 1000.0


def write_rows(
    output: Path,
    *,
    backend: str,
    batch_size: int,
    sequence_length: int,
    repeats: int,
    forward_ms: list[float],
    timings: dict[tuple[str, str, str], Timing],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "batch_size",
        "sequence_length",
        "repeats",
        "scope",
        "module_name",
        "module_type",
        "input_shape",
        "calls",
        "total_ms",
        "avg_us",
        "min_us",
        "max_us",
    ]
    write_header = not output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "backend": backend,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "repeats": repeats,
                "scope": "full_forward",
                "module_name": "",
                "module_type": "",
                "input_shape": f"{batch_size}x{sequence_length}",
                "calls": len(forward_ms),
                "total_ms": sum(forward_ms),
                "avg_us": 1000.0 * sum(forward_ms) / max(1, len(forward_ms)),
                "min_us": 1000.0 * min(forward_ms),
                "max_us": 1000.0 * max(forward_ms),
            }
        )
        for (module_name, module_type, input_shape), timing in sorted(
            timings.items(),
            key=lambda item: item[1].total_ms,
            reverse=True,
        ):
            writer.writerow(
                {
                    "backend": backend,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "repeats": repeats,
                    "scope": "module",
                    "module_name": module_name,
                    "module_type": module_type,
                    "input_shape": input_shape,
                    "calls": timing.calls,
                    "total_ms": timing.total_ms,
                    "avg_us": 1000.0 * timing.total_ms / max(1, timing.calls),
                    "min_us": 1000.0 * timing.min_ms,
                    "max_us": 1000.0 * timing.max_ms,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[8, 64, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--load-dyadic", type=Path)
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Also time the original CPU Transformers model before Level 2 native.",
    )
    parser.add_argument(
        "--source-device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="cpu",
        help="Device for the original Transformers source timing.",
    )
    parser.add_argument(
        "--skip-module-timing",
        action="store_true",
        help="Only write full-forward rows for Level 2 native timing.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/level2/qwen_depth_profile.csv"))
    return parser.parse_args()


def resolve_source_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --source-device cuda but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested --source-device mps but MPS is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        os.environ["DYOP_CPU_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)
    build_native_cpu()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    if args.include_source:
        source_device = resolve_source_device(args.source_device)
        model.to(source_device)
        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                input_ids = torch.full(
                    (batch_size, sequence_length),
                    fill_value=tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=source_device,
                )
                forward_once(model, input_ids, source_device)
                forward_ms = [
                    forward_once(model, input_ids, source_device)
                    for _ in range(args.repeats)
                ]
                write_rows(
                    args.output,
                    backend=f"transformers-source-{source_device.type}",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    repeats=args.repeats,
                    forward_ms=forward_ms,
                    timings={},
                )
                print(
                    f"source device={source_device.type} batch={batch_size} "
                    f"seq={sequence_length}: avg_forward_ms="
                    f"{sum(forward_ms) / max(1, len(forward_ms)):.3f}"
                )
        model.to("cpu")

    encoded = (
        load_encoded_model(args.load_dyadic)
        if args.load_dyadic is not None
        else encode_model(
            model,
            max_bits=args.bits,
            optimize_prefix_bits=(args.bits,),
        )
    )
    native_model, replacement = build_level2_model(
        model,
        encoded,
        bits=args.bits,
        dtype=torch.float32,
        linear_backend="native-cpu",
        embedding_backend="native-cpu",
    )
    native_model.eval()
    print(f"Profiling {len(replacement.replaced_modules)} Level 2 native modules")

    for sequence_length in args.sequence_lengths:
        for batch_size in args.batch_sizes:
            input_ids = torch.full(
                (batch_size, sequence_length),
                fill_value=tokenizer.eos_token_id,
                dtype=torch.long,
            )
            forward_once(native_model, input_ids, torch.device("cpu"))
            handles, timings = (
                ([], {}) if args.skip_module_timing else register_timers(native_model)
            )
            forward_ms: list[float] = []
            try:
                for _ in range(args.repeats):
                    forward_ms.append(
                        forward_once(native_model, input_ids, torch.device("cpu"))
                    )
            finally:
                for handle in handles:
                    handle.remove()
            write_rows(
                args.output,
                backend="level2-native",
                batch_size=batch_size,
                sequence_length=sequence_length,
                repeats=args.repeats,
                forward_ms=forward_ms,
                timings=timings,
            )
            print(
                f"native batch={batch_size} seq={sequence_length}: "
                f"avg_forward_ms={sum(forward_ms) / max(1, len(forward_ms)):.3f}"
            )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
