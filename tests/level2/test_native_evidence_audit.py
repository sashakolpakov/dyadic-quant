import pandas as pd

from experiments.level2.audit_native_evidence import (
    summarize_qwen,
    summarize_resnet,
    summarize_resnet_per_class,
    summarize_textual,
)


def test_textual_summary_requires_each_requested_native_bit():
    summary = pd.DataFrame(
        [
            {
                "family": "arc",
                "variant": "dyop_native_6",
                "prompts": 1,
                "mean_cosine": 0.9,
                "judge_equivalent_rate": 1.0,
                "judged_prompts": 1,
            },
            {
                "family": "wikitext",
                "variant": "dyop_native_6",
                "prompts": 1,
                "mean_cosine": 0.8,
                "judge_equivalent_rate": 0.0,
                "judged_prompts": 1,
            },
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "family": "arc",
                "prompt_id": "a",
                "variant": "dyop_native_6",
                "judge_equivalent": True,
            },
            {
                "family": "wikitext",
                "prompt_id": "w",
                "variant": "dyop_native_6",
                "judge_equivalent": False,
            },
        ]
    )

    _, issues, evidence = summarize_textual(summary, comparison, [4, 5, 6])

    assert "missing textual metrics for dyop_native_4" in issues
    assert "missing textual metrics for dyop_native_5" in issues
    assert "missing textual metrics for dyop_native_6" not in issues
    assert evidence["missing_judge_rows"] == 0


def test_textual_summary_flags_missing_family_and_judge_rows():
    summary = pd.DataFrame(
        [
            {
                "family": "arc",
                "variant": "dyop_native_8",
                "prompts": 1,
                "mean_cosine": 0.9,
                "judge_equivalent_rate": None,
                "judged_prompts": 0,
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "family": "arc",
                "prompt_id": "a",
                "variant": "dyop_native_8",
                "judge_equivalent": None,
            }
        ]
    )

    _, issues, evidence = summarize_textual(summary, comparison, [8])

    assert "missing textual family: wikitext" in issues
    assert "textual comparison has 1 missing judge rows" in issues
    assert evidence["missing_judge_rows"] == 1


def test_resnet_per_class_requires_all_imagenette_classes_per_bit():
    rows = [
        {
            "execution_backend": "level2-native",
            "bits_per_weight": 6,
            "class_name": f"class_{index}",
            "images": 1,
        }
        for index in range(9)
    ]

    _, issues = summarize_resnet_per_class(pd.DataFrame(rows), [6])

    assert "resnet 6 bit per-class evidence covers 9 classes" in issues


def test_resnet_per_class_accepts_all_classes_for_each_bit():
    rows = []
    for bit in (6, 8):
        for index in range(10):
            rows.append(
                {
                    "execution_backend": "level2-native",
                    "bits_per_weight": bit,
                    "class_name": f"class_{index}",
                    "images": 2,
                }
            )

    _, issues = summarize_resnet_per_class(pd.DataFrame(rows), [6, 8])

    assert issues == []


def test_qwen_quality_thresholds_flag_weak_native_row():
    frame = pd.DataFrame(
        [
            {
                "execution_backend": "transformers_source",
                "bits_per_weight": 16,
                "total_model_bytes": 1000,
                "perplexity": 10.0,
                "arc_easy_accuracy": 0.5,
            },
            {
                "execution_backend": "level2-native",
                "level2_linear_backend": "native-cpu",
                "level2_embedding_backend": "native-cpu",
                "bits_per_weight": 6,
                "total_model_bytes": 250,
                "effective_bits_per_weight": 6.0,
                "perplexity": 30.0,
                "next_token_agreement": 0.6,
                "arc_easy_accuracy": 0.4,
                "evaluated_tokens": 100,
                "wikitext_tokens_per_s": 10.0,
            },
        ]
    )

    summary, issues = summarize_qwen(
        frame,
        [6],
        min_agreement=0.8,
        max_perplexity_ratio=2.0,
    )

    assert "qwen 6 bit agreement 0.6000 below 0.8000" in issues
    assert "qwen 6 bit perplexity ratio 3.0000 above 2.0000" in issues
    assert summary.iloc[0]["compression_vs_source"] == 4.0
    assert abs(summary.iloc[0]["arc_easy_delta_vs_reference"] + 0.1) < 1e-12


def test_resnet_quality_thresholds_flag_weak_native_row():
    frame = pd.DataFrame(
        [
            {
                "execution_backend": "torch_fp32",
                "bits_per_weight": 16,
                "total_model_bytes": 1000,
                "top1_accuracy": 0.7,
            },
            {
                "execution_backend": "level2-native",
                "level2_linear_backend": "native-cpu",
                "level2_conv_backend": "native-cpu",
                "level2_spatial_backend": "native-cpu",
                "bits_per_weight": 6,
                "images": 500,
                "total_model_bytes": 250,
                "top1_accuracy": 0.5,
                "reference_agreement": 0.75,
                "logit_cosine": 0.8,
                "logit_mae": 0.2,
                "latency_batch1_ms": 1.0,
                "images_per_s": 10.0,
            },
        ]
    )

    summary, issues = summarize_resnet(
        frame,
        [6],
        min_images=500,
        min_logit_cosine=0.95,
        min_reference_agreement=0.9,
        max_top1_drop=0.1,
    )

    assert "resnet 6 bit agreement 0.7500 below 0.9000" in issues
    assert "resnet 6 bit logit cosine 0.8000 below 0.9500" in issues
    assert "resnet 6 bit top1 drop 0.2000 above 0.1000" in issues
    assert summary.iloc[0]["compression_vs_source"] == 4.0
