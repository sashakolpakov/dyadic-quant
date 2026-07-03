import csv

from experiments.level2.sweep_qwen_depth_backends import collect_rows, selected_backends


def test_selected_backends_defaults_to_all_labels():
    labels = {name for name, _, _ in selected_backends(None)}

    assert labels == {
        "native_linear",
        "native_linear_mlp_plan",
        "native_linear_rmsnorm",
        "native_linear_mlp_plan_rmsnorm",
    }


def test_collect_rows_keeps_full_forward_only(tmp_path):
    raw_csv = tmp_path / "profile.csv"
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
    with raw_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "backend": "level2-native",
                "batch_size": "2",
                "sequence_length": "8",
                "repeats": "3",
                "scope": "full_forward",
                "module_name": "",
                "module_type": "",
                "input_shape": "2x8",
                "calls": "3",
                "total_ms": "30.0",
                "avg_us": "10000.0",
                "min_us": "9000.0",
                "max_us": "11000.0",
            }
        )
        writer.writerow(
            {
                "backend": "level2-native",
                "batch_size": "2",
                "sequence_length": "8",
                "repeats": "3",
                "scope": "module",
                "module_name": "model.layers.0",
                "module_type": "Qwen2DecoderLayer",
                "input_shape": "2x8x896",
                "calls": "3",
                "total_ms": "12.0",
                "avg_us": "4000.0",
                "min_us": "3000.0",
                "max_us": "5000.0",
            }
        )

    rows = collect_rows(raw_csv, "native_linear_mlp_plan", "native-cpu-plan", "torch")

    assert rows == [
        {
            "backend_label": "native_linear_mlp_plan",
            "qwen_mlp_backend": "native-cpu-plan",
            "qwen_norm_backend": "torch",
            "batch_size": "2",
            "sequence_length": "8",
            "repeats": "3",
            "calls": "3",
            "total_ms": "30.000000",
            "avg_forward_ms": "10.000000",
            "min_forward_ms": "9.000000",
            "max_forward_ms": "11.000000",
            "raw_csv": str(raw_csv),
        }
    ]
