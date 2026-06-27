from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from padic_quant.model import train_mlp
from padic_quant.progressive import build_progressive_model, evaluate_representation


def load_dataset(name: str, seed: int):
    if name == "moons":
        inputs, targets = make_moons(n_samples=2000, noise=0.20, random_state=seed)
        widths = [2, 16, 16, 2]
        epochs = 100
    elif name == "digits":
        data = load_digits()
        inputs, targets = data.data, data.target
        widths = [64, 32, 10]
        epochs = 55
    else:
        raise ValueError(name)

    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        targets,
        test_size=0.25,
        random_state=seed,
        stratify=targets,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float64)
    x_test = scaler.transform(x_test).astype(np.float64)
    return x_train, x_test, y_train, y_test, widths, epochs


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for dataset_name in args.datasets:
            x_train, x_test, y_train, y_test, widths, default_epochs = load_dataset(
                dataset_name, seed
            )
            trained = train_mlp(
                x_train,
                y_train,
                x_test,
                y_test,
                widths,
                seed=seed,
                epochs=args.epochs or default_epochs,
            )
            progressive = build_progressive_model(
                trained.weights, trained.biases, max_bits=args.max_bits
            )
            print(
                f"seed={seed} {dataset_name}: float={trained.float_accuracy:.4f}, "
                f"conversion={progressive.conversion_ms:.3f} ms"
            )
            for bits in range(1, args.max_bits + 1):
                for method in ("independent", "monna", "residual_binary"):
                    metrics = evaluate_representation(
                        progressive,
                        x_test,
                        y_test,
                        trained.float_logits,
                        method=method,
                        bits=bits,
                        timing_repeats=args.timing_repeats,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset_name,
                            "method": method,
                            "bits_per_weight": bits,
                            "float_accuracy": trained.float_accuracy,
                            "accuracy_delta": metrics["accuracy"]
                            - trained.float_accuracy,
                            "one_time_conversion_ms": (
                                progressive.monna_conversion_ms
                                if method == "monna"
                                else progressive.residual_conversion_ms
                                if method == "residual_binary"
                                else metrics["reconstruction_ms"]
                            ),
                            "incremental_bits_per_weight": (
                                bits if method == "independent" else 1
                            ),
                            "incremental_payload_bytes": (
                                (progressive.weight_count * bits + 7) // 8
                                if method == "independent"
                                else (progressive.weight_count + 7) // 8
                            ),
                            "weight_count": progressive.weight_count,
                            "bias_count": progressive.bias_count,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["moons", "digits"],
        default=["moons", "digits"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = run(args)
    results_path = args.output_dir / "progressive_plane_results.csv"
    frame.to_csv(results_path, index=False)

    summary = (
        frame.groupby(["dataset", "method", "bits_per_weight"], dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_delta_mean=("accuracy_delta", "mean"),
            weight_rmse_mean=("weight_rmse", "mean"),
            total_bytes=("total_bytes", "first"),
            conversion_ms_mean=("one_time_conversion_ms", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            primitive_weight_ops=("primitive_weight_ops_per_sample", "first"),
            incremental_plane_ops=("incremental_plane_ops_per_sample", "first"),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "progressive_plane_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    metadata = {
        "command_arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "timing_note": (
            "Runtime uses reconstructed dense NumPy matrices. It is not a "
            "bit-packed kernel benchmark."
        ),
    }
    (args.output_dir / "progressive_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nWrote {results_path} and {summary_path}")


if __name__ == "__main__":
    main()
