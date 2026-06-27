from __future__ import annotations

import argparse
import json
import math
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

from dyadic_quant.arithmetic import (
    entropy_bits,
    largest_prime_power_below,
    packed_bytes,
)
from dyadic_quant.inference import build_quantized_mlp, evaluate_quantized
from dyadic_quant.model import train_mlp


RNS_BY_BITS = {
    8: (11, 23),       # product 253
    12: (61, 67),      # product 4087
    16: (251, 257),    # product 64507
    18: (503, 509),    # product 256027
    24: (4093, 4099),  # product 16,777,207
}


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


def result_row(
    *,
    seed: int,
    dataset: str,
    operand_bits: int,
    arithmetic: str,
    family: str,
    modulus: int | None,
    prime: int | None,
    places: int | None,
    lane_moduli: tuple[int, int] | None,
    evaluation,
    float_accuracy: float,
    parameter_count: int,
    weight_count: int,
    bias_count: int,
):
    entropy = entropy_bits(modulus) if modulus is not None else math.nan
    accumulator_bits = math.ceil(entropy) if modulus is not None else 32
    lane_storage_bits = (
        sum(math.ceil(math.log2(value)) for value in lane_moduli)
        if lane_moduli
        else accumulator_bits
    )
    weight_bytes = packed_bytes(weight_count, 2**operand_bits)
    bias_bytes = math.ceil(bias_count * lane_storage_bits / 8)
    return {
        "seed": seed,
        "dataset": dataset,
        "operand_bits": operand_bits,
        "arithmetic": arithmetic,
        "family": family,
        "prime": prime,
        "places": places,
        "modulus": modulus,
        "rns_moduli": "x".join(map(str, lane_moduli)) if lane_moduli else "",
        "modulus_entropy_bits": entropy,
        "physical_lane_bits": lane_storage_bits,
        "accuracy": evaluation.accuracy,
        "float_accuracy": float_accuracy,
        "accuracy_delta": evaluation.accuracy - float_accuracy,
        "float_agreement": evaluation.agreement,
        "logit_mae": evaluation.logit_mae,
        "wrap_rate": evaluation.wrap_rate,
        "wrapped_values": evaluation.wrapped_values,
        "accumulator_values": evaluation.accumulator_values,
        "max_abs_accumulator": evaluation.max_abs_accumulator,
        "runtime_ms": evaluation.runtime_ms,
        "logical_macs": evaluation.multiply_accumulates,
        "residue_macs": evaluation.multiply_accumulates
        * (len(lane_moduli) if lane_moduli else 1),
        "input_clip_rate": evaluation.input_clip_rate,
        "weight_clip_rate": evaluation.weight_clip_rate,
        "parameter_count": parameter_count,
        "weight_count": weight_count,
        "bias_count": bias_count,
        "weight_storage_packed_bytes": weight_bytes,
        "bias_storage_packed_bytes": bias_bytes,
        "parameter_storage_packed_bytes": weight_bytes + bias_bytes,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
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
            print(
                f"seed={seed} {dataset_name}: "
                f"float accuracy={trained.float_accuracy:.4f}, "
                f"parameters={trained.parameter_count}"
            )

            for operand_bits in args.operand_bits:
                quantized = build_quantized_mlp(
                    trained.weights,
                    trained.biases,
                    trained.calibration_inputs,
                    operand_bits,
                )
                wide = evaluate_quantized(
                    quantized,
                    x_test,
                    y_test,
                    trained.float_logits,
                    arithmetic="wide",
                    timing_repeats=args.timing_repeats,
                )
                rows.append(
                    result_row(
                        seed=seed,
                        dataset=dataset_name,
                        operand_bits=operand_bits,
                        arithmetic="wide",
                        family="integer-wide-accumulator",
                        modulus=None,
                        prime=None,
                        places=None,
                        lane_moduli=None,
                        evaluation=wide,
                        float_accuracy=trained.float_accuracy,
                        parameter_count=trained.parameter_count,
                        weight_count=trained.weight_count,
                        bias_count=trained.bias_count,
                    )
                )

                for accumulator_bits in args.accumulator_bits:
                    for prime in args.primes:
                        modulus, places = largest_prime_power_below(
                            accumulator_bits, prime
                        )
                        modular = evaluate_quantized(
                            quantized,
                            x_test,
                            y_test,
                            trained.float_logits,
                            arithmetic="modular",
                            modulus=modulus,
                            timing_repeats=args.timing_repeats,
                        )
                        rows.append(
                            result_row(
                                seed=seed,
                                dataset=dataset_name,
                                operand_bits=operand_bits,
                                arithmetic="modular",
                                family=f"Z/{prime}^{places}Z",
                                modulus=modulus,
                                prime=prime,
                                places=places,
                                lane_moduli=None,
                                evaluation=modular,
                                float_accuracy=trained.float_accuracy,
                                parameter_count=trained.parameter_count,
                                weight_count=trained.weight_count,
                                bias_count=trained.bias_count,
                            )
                        )

                    rns_moduli = RNS_BY_BITS[accumulator_bits]
                    rns_modulus = math.prod(rns_moduli)
                    rns = evaluate_quantized(
                        quantized,
                        x_test,
                        y_test,
                        trained.float_logits,
                        arithmetic="rns",
                        rns_moduli=rns_moduli,
                        timing_repeats=args.timing_repeats,
                    )
                    rows.append(
                        result_row(
                            seed=seed,
                            dataset=dataset_name,
                            operand_bits=operand_bits,
                            arithmetic="rns",
                            family="RNS",
                            modulus=rns_modulus,
                            prime=None,
                            places=None,
                            lane_moduli=rns_moduli,
                            evaluation=rns,
                            float_accuracy=trained.float_accuracy,
                            parameter_count=trained.parameter_count,
                            weight_count=trained.weight_count,
                            bias_count=trained.bias_count,
                        )
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
    parser.add_argument("--operand-bits", nargs="+", type=int, default=[4, 8])
    parser.add_argument(
        "--accumulator-bits",
        nargs="+",
        type=int,
        choices=sorted(RNS_BY_BITS),
        default=[8, 12, 16, 18, 24],
    )
    parser.add_argument("--primes", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = run(args)
    csv_path = args.output_dir / "small_network_results.csv"
    frame.to_csv(csv_path, index=False)

    group_columns = [
        "dataset",
        "operand_bits",
        "family",
        "modulus_entropy_bits",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_delta_mean=("accuracy_delta", "mean"),
            wrap_rate_mean=("wrap_rate", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "small_network_summary.csv"
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
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nWrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
