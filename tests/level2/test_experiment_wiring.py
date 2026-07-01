from argparse import Namespace
from pathlib import Path

import torch

from experiments.level1.run_qwen_dyadic import (
    level2_uses_native_cpu as qwen_uses_native_cpu,
    resolve_device as resolve_qwen_device,
    resolve_model_dtype as resolve_qwen_model_dtype,
    resolve_output_dir as resolve_qwen_output_dir,
)
from experiments.level2.run_native_dyop_prefix_sweep import (
    CSV_FIELDS,
    prefix_row,
    validate_prefix_bits,
)
from experiments.level2.common import require_speed_gates
from experiments.level1.run_resnet18_dyadic import (
    level2_uses_native_cpu as resnet_uses_native_cpu,
    resolve_device as resolve_resnet_device,
    resolve_eval_dtype as resolve_resnet_eval_dtype,
    resolve_output_dir as resolve_resnet_output_dir,
)


def test_level2_native_runs_default_to_level2_results_dir():
    args = Namespace(execution_backend="level2-native", output_dir=None)

    assert resolve_resnet_output_dir(args) == Path("results/level2")
    assert resolve_qwen_output_dir(args) == Path("results/level2")


def test_materialized_runs_default_to_level1_results_dir():
    args = Namespace(execution_backend="materialized", output_dir=None)

    assert resolve_resnet_output_dir(args) == Path("results")
    assert resolve_qwen_output_dir(args) == Path("results")


def test_explicit_output_dir_overrides_backend_defaults():
    args = Namespace(
        execution_backend="level2-native",
        output_dir=Path("/tmp/custom-level2-results"),
    )

    assert resolve_resnet_output_dir(args) == Path("/tmp/custom-level2-results")
    assert resolve_qwen_output_dir(args) == Path("/tmp/custom-level2-results")


def test_resnet_native_cpu_backend_routes_to_cpu_float32():
    args = Namespace(
        execution_backend="level2-native",
        level2_linear_backend="native-cpu",
        level2_conv_backend="native-cpu",
    )

    assert resnet_uses_native_cpu(args)
    assert resolve_resnet_device(args) == torch.device("cpu")
    assert resolve_resnet_eval_dtype(args) == torch.float32


def test_qwen_native_cpu_backend_routes_to_cpu_float32():
    args = Namespace(
        execution_backend="level2-native",
        level2_linear_backend="native-cpu",
        level2_embedding_backend="native-cpu",
        dtype="bfloat16",
    )

    assert qwen_uses_native_cpu(args)
    assert resolve_qwen_device(args) == torch.device("cpu")
    assert resolve_qwen_model_dtype(args) == torch.float32


def test_prefix_sweep_validates_and_sorts_prefix_bits():
    assert validate_prefix_bits([8, 4, 6, 4], 8) == (4, 6, 8)


def test_prefix_sweep_rejects_out_of_range_prefix_bits():
    try:
        validate_prefix_bits([4, 9], 8)
    except ValueError as error:
        assert "prefix widths" in str(error)
    else:
        raise AssertionError("expected invalid prefix width to raise ValueError")


def test_prefix_sweep_row_matches_csv_schema():
    row = prefix_row(
        bits=6,
        linear_backend="native-cpu",
        embedding_backend="native-cpu",
        conv_backend="native-cpu",
        conversion_ms=1.0,
        materialization_ms=2.0,
        level2_load_ms=3.0,
        level2_build_ms=4.0,
        level1_forward_ms=5.0,
        level2_forward_ms=6.0,
        max_abs_error=0.0,
        allclose=True,
        sizes={
            "total_model_bytes": 10,
            "incremental_plane_bytes": 2,
            "weight_payload_bytes": 7,
            "exponent_bytes": 1,
            "other_model_bytes": 2,
        },
        replaced_modules=("embed", "proj"),
        shared_weight_modules=("proj",),
    )

    assert list(row.keys()) == CSV_FIELDS
    assert row["method"] == "level2_native_dyop"
    assert row["linear_backend"] == "native-cpu"
    assert row["embedding_backend"] == "native-cpu"
    assert row["conv_backend"] == "native-cpu"
    assert row["level2_replaced_modules"] == "embed,proj"
    assert row["level2_shared_weight_modules"] == "proj"


def test_qwen_speed_gate_requirement_fails_on_missing_or_failed_rows(tmp_path):
    gates = tmp_path / "gates.csv"
    gates.write_text(
        "subkernel,passes_speed_gate\n"
        "linear_gemm_qwen_seq,false\n"
        "linear_output_projection,true\n"
    )

    try:
        require_speed_gates(gates, "qwen")
    except RuntimeError as error:
        message = str(error)
        assert "failed gates: linear_gemm_qwen_seq" in message
        assert "missing gates: embedding_qwen_vocab_width" in message
    else:
        raise AssertionError("expected incomplete Qwen gates to fail")
