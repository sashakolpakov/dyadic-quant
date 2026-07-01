from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torch.nn.utils.fusion import fuse_conv_bn_eval

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dyadic_quant.level1 import (
    encode_model,
    load_encoded_model,
    materialize_prefix,
    storage_bytes,
)
from dyadic_quant.level2 import (
    build_level2_model,
    build_native_cpu,
    native_add_relu_cpu,
    warm_native_cpu_workers,
)
from experiments.level2.common import require_speed_gates


DEFAULT_OUTPUT_DIR = Path("results")
LEVEL2_OUTPUT_DIR = Path("results/level2")


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
        raise RuntimeError("MPS is unavailable; CPU execution is disabled")
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


class NativeResidualBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return native_add_relu_cpu(out.contiguous(), identity.contiguous())


def replace_resnet_basic_blocks_with_native_residuals(
    model: torch.nn.Module,
) -> tuple[str, ...]:
    replaced: list[str] = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for index, block in enumerate(layer):
            layer[index] = NativeResidualBasicBlock(block)
            replaced.append(f"{layer_name}.{index}")
    return tuple(replaced)


def level2_uses_native_cpu(args: argparse.Namespace) -> bool:
    return args.execution_backend == "level2-native" and (
        args.level2_linear_backend == "native-cpu"
        or args.level2_conv_backend == "native-cpu"
        or args.level2_spatial_backend == "native-cpu"
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    if level2_uses_native_cpu(args):
        return torch.device("cpu")
    return require_mps()


def resolve_eval_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float32 if level2_uses_native_cpu(args) else torch.float16


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    reference_logits: list[torch.Tensor] | None,
    dtype: torch.dtype,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    correct = 0
    total = 0
    agreement = 0
    absolute_error = 0.0
    cosine_sum = 0.0
    output_values = 0
    logits_out: list[torch.Tensor] = []
    synchronize(device)
    start = perf_counter()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device=device, dtype=dtype)
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
                cosine_sum += float(
                    F.cosine_similarity(cpu_logits, reference, dim=1).sum().item()
                )
                output_values += cpu_logits.numel()
    synchronize(device)
    elapsed = perf_counter() - start
    return (
        {
            "top1_accuracy": correct / total,
            "reference_agreement": agreement / total if reference_logits else 1.0,
            "logit_mae": absolute_error / output_values if output_values else 0.0,
            "logit_cosine": cosine_sum / total if reference_logits else 1.0,
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
    dtype: torch.dtype,
) -> float:
    if repeats <= 0:
        return 0.0
    sample = torch.randn(
        batch_size, 3, 224, 224, dtype=dtype, device=device
    )
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(device)
        start = perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize(device)
    return (perf_counter() - start) * 1000 / repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--per-class-limit",
        type=int,
        help=(
            "Evaluate a deterministic balanced subset with this many images per "
            "Imagenette class. Mutually exclusive with --limit."
        ),
    )
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument(
        "--quantize-endpoints",
        action="store_true",
        help="Also quantize conv1 and fc; by default they remain FP16.",
    )
    parser.add_argument(
        "--load-dyadic",
        type=Path,
        help="Load an existing packed dyadic artifact instead of encoding weights.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["materialized", "level2-native"],
        default="materialized",
        help=(
            "materialized decodes prefixes into FP16 weights; level2-native "
            "replaces encoded modules with native dyop execution modules."
        ),
    )
    parser.add_argument(
        "--level2-linear-backend",
        choices=["scalar", "native-cpu"],
        default="scalar",
        help="Level 2 backend for Linear/fc modules.",
    )
    parser.add_argument(
        "--level2-conv-backend",
        choices=["scalar", "native-cpu"],
        default="scalar",
        help="Level 2 backend for Conv2d modules.",
    )
    parser.add_argument(
        "--level2-spatial-backend",
        choices=["torch", "native-cpu"],
        help=(
            "Level 2 backend for spatial modules such as MaxPool2d and "
            "AdaptiveAvgPool2d. Defaults to native-cpu for level2-native runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Result directory. Defaults to results/ for materialized runs and "
            "results/level2/ for level2-native runs."
        ),
    )
    parser.add_argument(
        "--level2-speed-gates",
        type=Path,
        default=Path("results/level2/subkernel_speed_gates_arm64_neon_latest.csv"),
        help="CSV proving required ResNet native dyop kernels beat materialized gates.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.execution_backend == "level2-native":
        return LEVEL2_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def balanced_subset(dataset: ImageFolder, per_class_limit: int) -> torch.utils.data.Subset:
    selected: list[int] = []
    counts: Counter[int] = Counter()
    for index, (_, class_index) in enumerate(dataset.samples):
        if counts[class_index] >= per_class_limit:
            continue
        selected.append(index)
        counts[class_index] += 1
        if len(counts) == len(dataset.classes) and all(
            counts[class_index] >= per_class_limit
            for class_index in range(len(dataset.classes))
        ):
            break
    if len(counts) != len(dataset.classes):
        raise RuntimeError(
            f"balanced subset did not cover all classes: {dict(counts)}"
        )
    subset = torch.utils.data.Subset(dataset, selected)
    subset.classes = dataset.classes
    subset.class_sample_counts = {
        dataset.classes[class_index]: counts[class_index]
        for class_index in range(len(dataset.classes))
    }
    return subset


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.per_class_limit is not None:
        raise RuntimeError("--limit and --per-class-limit are mutually exclusive")
    if args.level2_spatial_backend is None:
        args.level2_spatial_backend = (
            "native-cpu" if args.execution_backend == "level2-native" else "torch"
        )
    args.output_dir = resolve_output_dir(args)
    if args.execution_backend == "level2-native":
        require_speed_gates(args.level2_speed_gates, "resnet")
    if level2_uses_native_cpu(args):
        build_native_cpu()
        warm_native_cpu_workers()
    device = resolve_device(args)
    eval_dtype = resolve_eval_dtype(args)
    weights = ResNet18_Weights.DEFAULT
    base_model = resnet18(weights=None)
    base_model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    base_model = fuse_resnet_batch_norm(base_model)
    encoded = (
        load_encoded_model(args.load_dyadic)
        if args.load_dyadic is not None
        else encode_model(
            base_model,
            max_bits=max(args.bits),
            optimize_prefix_bits=tuple(sorted(set(args.bits))),
            exclude_names=set() if args.quantize_endpoints else {"conv1", "fc"},
        )
    )

    dataset = ImageFolder(args.data_root / "val", transform=weights.transforms())
    if set(dataset.classes) != set(IMAGENETTE_TARGETS):
        raise RuntimeError(f"unexpected Imagenette classes: {dataset.classes}")
    class_sample_counts: dict[str, int]
    if args.per_class_limit is not None:
        dataset = balanced_subset(dataset, args.per_class_limit)
        class_sample_counts = dataset.class_sample_counts
    elif args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
        # Preserve ImageFolder metadata used by evaluate.
        dataset.classes = dataset.dataset.classes
        subset_targets = [
            dataset.dataset.samples[index][1]
            for index in range(min(args.limit, len(dataset.dataset)))
        ]
        class_counts = Counter(subset_targets)
        class_sample_counts = {
            dataset.classes[index]: class_counts.get(index, 0)
            for index in range(len(dataset.classes))
        }
    else:
        class_sample_counts = {
            class_name: count
            for class_name, count in zip(
                dataset.classes,
                torch.bincount(
                    torch.tensor([target for _, target in dataset.samples]),
                    minlength=len(dataset.classes),
                ).tolist(),
                strict=True,
            )
        }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    rows: list[dict[str, object]] = []
    reference_model = copy.deepcopy(base_model).to(device=device, dtype=eval_dtype)
    reference_metrics, reference_logits = evaluate(
        reference_model, loader, device, reference_logits=None, dtype=eval_dtype
    )
    reference_latency_1 = latency(
        reference_model,
        device,
        batch_size=1,
        warmup=5,
        repeats=args.latency_repeats,
        dtype=eval_dtype,
    )
    reference_latency_batch = latency(
        reference_model,
        device,
        batch_size=args.batch_size,
        warmup=3,
        repeats=0 if args.latency_repeats <= 0 else max(5, args.latency_repeats // 3),
        dtype=eval_dtype,
    )
    dtype_bytes = 4 if eval_dtype == torch.float32 else 2
    fp16_bytes = sum(
        tensor.numel() * dtype_bytes
        for tensor in list(reference_model.parameters()) + list(reference_model.buffers())
    )
    rows.append(
        {
            "method": (
                "fp32_reference" if eval_dtype == torch.float32 else "fp16_reference"
            ),
            "execution_backend": (
                "torch_fp32" if eval_dtype == torch.float32 else "torch_fp16"
            ),
            "level2_linear_backend": "",
            "level2_conv_backend": "",
            "level2_spatial_backend": "",
            "bits_per_weight": 16,
            "conversion_ms": 0.0,
            "materialization_ms": 0.0,
            "level2_build_ms": 0.0,
            "level2_native_residual_blocks": "",
            "total_model_bytes": fp16_bytes,
            "incremental_plane_bytes": 0,
            "latency_batch1_ms": reference_latency_1,
            "latency_batch_ms": reference_latency_batch,
            **reference_metrics,
        }
    )
    del reference_model
    empty_cache(device)

    for bits in args.bits:
        level2_build_ms = 0.0
        level2_replaced_modules: tuple[str, ...] = ()
        level2_shared_weight_modules: tuple[str, ...] = ()
        level2_native_residual_blocks: tuple[str, ...] = ()
        if args.execution_backend == "materialized":
            candidate = copy.deepcopy(base_model)
            materialization_ms = materialize_prefix(candidate, encoded, bits=bits)
        else:
            start = perf_counter()
            candidate, replacement = build_level2_model(
                base_model,
                encoded,
                bits=bits,
                dtype=eval_dtype,
                linear_backend=args.level2_linear_backend,
                conv_backend=args.level2_conv_backend,
                spatial_backend=args.level2_spatial_backend,
            )
            level2_build_ms = (perf_counter() - start) * 1000
            materialization_ms = 0.0
            level2_replaced_modules = replacement.replaced_modules
            level2_shared_weight_modules = replacement.shared_weight_modules
            if args.level2_spatial_backend == "native-cpu":
                level2_native_residual_blocks = (
                    replace_resnet_basic_blocks_with_native_residuals(candidate)
                )
        sizes = storage_bytes(base_model, encoded, bits=bits)
        candidate = candidate.to(device=device, dtype=eval_dtype).eval()
        metrics, _ = evaluate(
            candidate,
            loader,
            device,
            reference_logits=reference_logits,
            dtype=eval_dtype,
        )
        rows.append(
            {
                "method": (
                    "per_channel_dyadic"
                    if args.execution_backend == "materialized"
                    else "per_channel_dyadic_level2_native"
                ),
                "execution_backend": args.execution_backend,
                "level2_linear_backend": (
                    args.level2_linear_backend
                    if args.execution_backend == "level2-native"
                    else ""
                ),
                "level2_conv_backend": (
                    args.level2_conv_backend
                    if args.execution_backend == "level2-native"
                    else ""
                ),
                "level2_spatial_backend": (
                    args.level2_spatial_backend
                    if args.execution_backend == "level2-native"
                    else ""
                ),
                "bits_per_weight": bits,
                "conversion_ms": encoded.conversion_ms,
                "materialization_ms": materialization_ms,
                "level2_build_ms": level2_build_ms,
                "level2_replaced_modules": ",".join(level2_replaced_modules),
                "level2_shared_weight_modules": ",".join(
                    level2_shared_weight_modules
                ),
                "level2_native_residual_blocks": ",".join(
                    level2_native_residual_blocks
                ),
                "latency_batch1_ms": latency(
                    candidate,
                    device,
                    batch_size=1,
                    warmup=5,
                    repeats=args.latency_repeats,
                    dtype=eval_dtype,
                ),
                "latency_batch_ms": latency(
                    candidate,
                    device,
                    batch_size=args.batch_size,
                    warmup=3,
                    repeats=(
                        0
                        if args.latency_repeats <= 0
                        else max(5, args.latency_repeats // 3)
                    ),
                    dtype=eval_dtype,
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
        empty_cache(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    result_path = args.output_dir / "resnet18_dyadic_results.csv"
    frame.to_csv(result_path, index=False)
    metadata = {
        "arguments": vars(args) | {
            "data_root": str(args.data_root),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "load_dyadic": str(args.load_dyadic) if args.load_dyadic else None,
        },
        "model": "torchvision ResNet18_Weights.DEFAULT",
        "torch": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
        "quantized_weight_count": encoded.quantized_weight_count,
        "exponent_count": encoded.exponent_count,
        "exponent_selection": "normalized_regret",
        "optimized_prefix_bits": sorted(set(args.bits)),
        "reference_dtype": str(eval_dtype).replace("torch.", ""),
        "reference_graph": "Conv-BatchNorm fused before FP16 cast",
        "encoding_source_dtype": "float32",
        "execution_backend": args.execution_backend,
        "level2_linear_backend": args.level2_linear_backend,
        "level2_conv_backend": args.level2_conv_backend,
        "level2_spatial_backend": args.level2_spatial_backend,
        "native_residual_note": (
            "When level2_spatial_backend is native-cpu, torchvision BasicBlock "
            "instances are wrapped so the residual add plus final ReLU executes "
            "through native_add_relu_cpu instead of Torch tensor addition."
        ),
        "storage_note": (
            "Estimated packed format. For level2-native rows, storage is "
            "computed from the original Level 1 encoded source model because "
            "decoded weight parameters are intentionally absent from Level 2 "
            "dyop modules."
        ),
        "throughput_note": (
            "Whole-dataset images_per_s includes loader and first-run MPS "
            "startup and is not a cross-row speed comparison. Use warmed "
            "latency_batch*_ms for the FP16-materialized path."
        ),
        "dataset_size": len(dataset),
        "class_sample_counts": class_sample_counts,
    }
    (args.output_dir / "resnet18_dyadic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(frame.to_string(index=False))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
