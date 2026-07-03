import copy

import torch
from torch import nn

from dyadic_quant.level1 import (
    encode_model,
    load_encoded_model,
    materialize_prefix,
    save_encoded_model,
)
from dyadic_quant.level2 import (
    DyadicConv2d,
    DyadicEmbedding,
    DyadicLinear,
    DyadicQwenMLPNative,
    NativeRMSNorm,
    build_level2_model,
)


def test_level2_sequential_mlp_matches_level1_materialized_forward():
    torch.manual_seed(10)
    source = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
    ).eval()
    inputs = torch.randn(2, 4)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(4, 6, 8))

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=6)
    level2, report = build_level2_model(source, encoded, bits=6)
    level2.eval()

    assert report.replaced_modules == ("0", "2")
    assert isinstance(level2[0], DyadicLinear)
    assert isinstance(level2[2], DyadicLinear)
    torch.testing.assert_close(level2(inputs), level1(inputs), rtol=1e-6, atol=1e-6)


def test_level2_conv_model_matches_level1_materialized_forward():
    torch.manual_seed(11)
    source = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(3, 4, kernel_size=(2, 3), stride=(1, 2)),
    ).eval()
    inputs = torch.randn(1, 2, 5, 6)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(5, 8))

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=5)
    level2, report = build_level2_model(source, encoded, bits=5)
    level2.eval()

    assert report.replaced_modules == ("0", "2")
    assert isinstance(level2[0], DyadicConv2d)
    assert isinstance(level2[2], DyadicConv2d)
    torch.testing.assert_close(level2(inputs), level1(inputs), rtol=1e-6, atol=1e-6)


def test_level2_tied_embedding_projection_reuses_one_encoded_tensor():
    class TiedTinyLm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(13, 4)
            self.proj = nn.Linear(4, 13, bias=False)
            self.proj.weight = self.embed.weight

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            hidden = self.embed(tokens)
            return self.proj(hidden)

    torch.manual_seed(12)
    source = TiedTinyLm().eval()
    tokens = torch.tensor([[1, 4, 7], [0, 12, 3]])
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=6)
    level2, report = build_level2_model(source, encoded, bits=6)
    level2.eval()

    assert len(encoded.modules) == 1
    assert report.replaced_modules == ("embed", "proj")
    assert report.shared_weight_modules == ("proj",)
    assert isinstance(level2.embed, DyadicEmbedding)
    assert isinstance(level2.proj, DyadicLinear)
    assert level2.embed.encoded is level2.proj.encoded
    torch.testing.assert_close(level2(tokens), level1(tokens), rtol=1e-6, atol=1e-6)


def test_level2_model_respects_prefix_overrides():
    torch.manual_seed(13)
    source = nn.Sequential(
        nn.Embedding(9, 4),
        nn.Linear(4, 6),
    ).eval()
    tokens = torch.tensor([[1, 2, 3]])
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(4, 8))
    overrides = {"0": 8, "1": 4}

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=4, overrides=overrides)
    level2, report = build_level2_model(source, encoded, bits=4, overrides=overrides)
    level2.eval()

    assert report.replaced_modules == ("0", "1")
    torch.testing.assert_close(level2(tokens), level1(tokens), rtol=1e-6, atol=1e-6)


def test_level2_model_executes_loaded_packed_artifact(tmp_path):
    torch.manual_seed(14)
    source = nn.Sequential(
        nn.Embedding(8, 3),
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    ).eval()
    tokens = torch.tensor([[1, 2], [3, 4]])
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(5, 8))
    artifact = tmp_path / "tiny.dyadic.pt"
    save_encoded_model(encoded, artifact)
    loaded = load_encoded_model(artifact)

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=5)
    level2, report = build_level2_model(source, loaded, bits=5)
    level2.eval()

    assert report.replaced_modules == ("0", "1", "3")
    torch.testing.assert_close(level2(tokens), level1(tokens), rtol=1e-6, atol=1e-6)


def test_level2_qwen_mlp_native_plan_matches_materialized_forward():
    class TinyQwenMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(4, 7)
            self.up_proj = nn.Linear(4, 7)
            self.down_proj = nn.Linear(7, 4)
            self.act_fn = torch.nn.functional.silu

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.down_proj(
                self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs)
            )

    torch.manual_seed(15)
    source = TinyQwenMLP().eval()
    inputs = torch.randn(2, 3, 4)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=6)
    level2, report = build_level2_model(
        source,
        encoded,
        bits=6,
        linear_backend="native-cpu",
        qwen_mlp_backend="native-cpu-plan",
    )
    level2.eval()

    assert report.replaced_modules == ("gate_proj", "up_proj", "down_proj")
    assert report.fused_modules == ("",)
    assert isinstance(level2, DyadicQwenMLPNative)
    torch.testing.assert_close(level2(inputs), level1(inputs), rtol=1e-5, atol=1e-5)


def test_level2_qwen_rms_norm_native_matches_source_forward():
    class TinyRMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5))
            self.variance_epsilon = 1e-6

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            variance = inputs.pow(2).mean(dim=-1, keepdim=True)
            return inputs * torch.rsqrt(variance + self.variance_epsilon) * self.weight

    class TinyNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = TinyRMSNorm()
            self.proj = nn.Linear(5, 3)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.proj(self.input_layernorm(inputs))

    torch.manual_seed(16)
    source = TinyNormModel().eval()
    inputs = torch.randn(2, 4, 5)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    level1 = copy.deepcopy(source).eval()
    materialize_prefix(level1, encoded, bits=6)
    level2, report = build_level2_model(
        source,
        encoded,
        bits=6,
        linear_backend="native-cpu",
        qwen_norm_backend="native-cpu",
    )
    level2.eval()

    assert report.replaced_modules == ("input_layernorm", "proj")
    assert isinstance(level2.input_layernorm, NativeRMSNorm)
    torch.testing.assert_close(level2(inputs), level1(inputs), rtol=1e-5, atol=1e-5)
