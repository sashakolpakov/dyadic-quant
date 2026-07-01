from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level1 import (
    encode_model,
    load_encoded_model,
    materialize_prefix,
    save_encoded_model,
    storage_bytes,
)
from dyadic_quant.level2 import build_level2_model

from experiments.level2.common import TinyDyadicNet, timed_forward, tiny_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Level 2 native dyop execution to Level 1 materialization."
    )
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--reload-packed",
        action="store_true",
        help="Save the Level 1 packed dyadic artifact and build Level 2 from it.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/level2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    source = TinyDyadicNet().eval()
    tokens, image = tiny_inputs(args.seed + 1)
    group_size = args.group_size or None
    encoded = encode_model(
        source,
        max_bits=args.max_bits,
        optimize_prefix_bits=(args.bits, args.max_bits),
        group_size=group_size,
    )

    level1 = copy.deepcopy(source).eval()
    materialize_ms = materialize_prefix(level1, encoded, bits=args.bits)
    level1_output, level1_ms = timed_forward(level1, tokens, image, args.repeats)

    packed_artifact = None
    level2_encoded = encoded
    if args.reload_packed:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        packed_artifact = args.output_dir / "native_dyop_smoke.dyadic.pt"
        save_encoded_model(encoded, packed_artifact)
        level2_encoded = load_encoded_model(packed_artifact)

    level2, replacement = build_level2_model(source, level2_encoded, bits=args.bits)
    level2.eval()
    level2_output, level2_ms = timed_forward(level2, tokens, image, args.repeats)

    max_abs_error = float(torch.max(torch.abs(level2_output - level1_output)).item())
    payload = {
        "level": 2,
        "bits": args.bits,
        "max_bits": args.max_bits,
        "group_size": args.group_size,
        "seed": args.seed,
        "repeats": args.repeats,
        "reload_packed": args.reload_packed,
        "packed_artifact": str(packed_artifact) if packed_artifact else None,
        "materialization_ms": materialize_ms,
        "level1_forward_ms": level1_ms,
        "level2_forward_ms": level2_ms,
        "max_abs_error_vs_level1_materialized": max_abs_error,
        "replaced_modules": replacement.replaced_modules,
        "shared_weight_modules": replacement.shared_weight_modules,
        "storage": storage_bytes(source, encoded, bits=args.bits),
        "note": (
            "Level 2 scalar kernels are correctness kernels. They avoid decoded "
            "weight execution but are not expected to be faster than PyTorch."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "native_dyop_smoke.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
