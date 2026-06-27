from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torch.nn.utils.fusion import fuse_conv_bn_eval

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from padic_quant.dyadic_torch import (
    encode_model,
    materialize_prefix,
    storage_bytes,
)


IMAGENETTE_TARGETS = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is intentionally disabled")
    return torch.device("mps")


def fuse_resnet_batch_norm(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    model.conv1 = fuse_conv_bn_eval(model.conv1, model.bn1)
    model.bn1 = torch.nn.Identity()
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(model, layer_name):
            block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = torch.nn.Identity()
            block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = torch.nn.Identity()
            if block.downsample is not None:
                block.downsample[0] = fuse_conv_bn_eval(
                    block.downsample[0], block.downsample[1]
                )
                block.downsample[1] = torch.nn.Identity()
    return model


def synchronize() -> None:
    torch.mps.synchronize()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    reference_logits: list[torch.Tensor] | None,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    correct = 0
    total = 0
    agreement = 0
    absolute_error = 0.0
    output_values = 0
    logits_out: list[torch.Tensor] = []
    synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device=device, dtype=torch.float16)
            target_indices = torch.tensor(
                [IMAGENETTE_TARGETS[loader.dataset.classes[index]] for index in targets],
                device=device,
            )
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == target_indices).sum().item())
            total += images.shape[0]
            cpu_logits = logits.float().cpu()
            logits_out.append(cpu_logits)
            if reference_logits is not None:
                reference = reference_logits[batch_index]
                agreement += int(
                    (cpu_logits.argmax(dim=1) == reference.argmax(dim=1)).sum().item()
                )
                absolute_error += float(torch.abs(cpu_logits - reference).sum().item())
                output_values += cpu_logits.numel()
    synchronize()
    elapsed = perf_counter() - start
    return (
        {
            "top1_accuracy": correct / total,
            "reference_agreement": agreement / total if reference_logits else 1.0,
            "logit_mae": absolute_error / output_values if output_values else 0.0,
            "images": total,
            "elapsed_s": elapsed,
            "images_per_s": total / elapsed,
        },
        logits_out,
    )


def latency(
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> float:
    sample = torch.randn(
        batch_size, 3, 224, 224, dtype=torch.float16, device=device
    )
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()
        start = perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize()
    return (perf_counter() - start) * 1000 / repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument(
        "--quantize-endpoints",
        action="store_true",
        help="Also quantize conv1 and fc; by default they remain FP16.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_mps()
    weights = ResNet18_Weights.DEFAULT
    base_model = resnet18(weights=None)
    base_model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    base_model = fuse_resnet_batch_norm(base_model)
    encoded = encode_model(
        base_model,
        max_bits=max(args.bits),
        optimize_prefix_bits=tuple(sorted(set(args.bits))),
        exclude_names=set() if args.quantize_endpoints else {"conv1", "fc"},
    )

    dataset = ImageFolder(args.data_root / "val", transform=weights.transforms())
    if set(dataset.classes) != set(IMAGENETTE_TARGETS):
        raise RuntimeError(f"unexpected Imagenette classes: {dataset.classes}")
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
        # Preserve ImageFolder metadata used by evaluate.
        dataset.classes = dataset.dataset.classes
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    rows: list[dict[str, object]] = []
    reference_model = copy.deepcopy(base_model).to(device=device, dtype=torch.float16)
    reference_metrics, reference_logits = evaluate(
        reference_model, loader, device, reference_logits=None
    )
    reference_latency_1 = latency(
        reference_model,
        device,
        batch_size=1,
        warmup=5,
        repeats=args.latency_repeats,
    )
    reference_latency_batch = latency(
        reference_model,
        device,
        batch_size=args.batch_size,
        warmup=3,
        repeats=max(5, args.latency_repeats // 3),
    )
    fp16_bytes = sum(
        tensor.numel() * 2
        for tensor in list(reference_model.parameters()) + list(reference_model.buffers())
    )
    rows.append(
        {
            "method": "fp16_reference",
            "bits_per_weight": 16,
            "conversion_ms": 0.0,
            "materialization_ms": 0.0,
            "total_model_bytes": fp16_bytes,
            "incremental_plane_bytes": 0,
            "latency_batch1_ms": reference_latency_1,
            "latency_batch_ms": reference_latency_batch,
            **reference_metrics,
        }
    )
    del reference_model
    torch.mps.empty_cache()

    for bits in args.bits:
        candidate = copy.deepcopy(base_model)
        materialization_ms = materialize_prefix(candidate, encoded, bits=bits)
        sizes = storage_bytes(candidate, encoded, bits=bits)
        candidate = candidate.to(device=device, dtype=torch.float16).eval()
        metrics, _ = evaluate(
            candidate, loader, device, reference_logits=reference_logits
        )
        rows.append(
            {
                "method": "per_channel_dyadic",
                "bits_per_weight": bits,
                "conversion_ms": encoded.conversion_ms,
                "materialization_ms": materialization_ms,
                "latency_batch1_ms": latency(
                    candidate,
                    device,
                    batch_size=1,
                    warmup=5,
                    repeats=args.latency_repeats,
                ),
                "latency_batch_ms": latency(
                    candidate,
                    device,
                    batch_size=args.batch_size,
                    warmup=3,
                    repeats=max(5, args.latency_repeats // 3),
                ),
                **sizes,
                **metrics,
            }
        )
        print(
            f"{bits}-bit: accuracy={metrics['top1_accuracy']:.4f}, "
            f"agreement={metrics['reference_agreement']:.4f}, "
            f"throughput={metrics['images_per_s']:.1f} img/s"
        )
        del candidate
        torch.mps.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    result_path = args.output_dir / "resnet18_dyadic_results.csv"
    frame.to_csv(result_path, index=False)
    metadata = {
        "arguments": vars(args) | {
            "data_root": str(args.data_root),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
        },
        "model": "torchvision ResNet18_Weights.DEFAULT",
        "torch": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
        "quantized_weight_count": encoded.quantized_weight_count,
        "exponent_count": encoded.exponent_count,
        "exponent_selection": "normalized_regret",
        "optimized_prefix_bits": sorted(set(args.bits)),
        "reference_dtype": "float16",
        "reference_graph": "Conv-BatchNorm fused before FP16 cast",
        "encoding_source_dtype": "float32",
        "storage_note": "Estimated packed format; no dyadic file is serialized.",
        "throughput_note": (
            "Whole-dataset images_per_s includes loader and first-run MPS "
            "startup and is not a cross-row speed comparison. Use warmed "
            "latency_batch*_ms for the FP16-materialized path."
        ),
        "dataset_size": len(dataset),
    }
    (args.output_dir / "resnet18_dyadic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(frame.to_string(index=False))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
