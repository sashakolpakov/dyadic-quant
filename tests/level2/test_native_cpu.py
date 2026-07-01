import copy

import torch
import torch.nn.functional as F
from torch import nn

from dyadic_quant.level1 import encode_model, encode_tensor_per_output_channel
from dyadic_quant.level2 import (
    DyadicConv2d,
    DyadicEmbedding,
    DyadicLinear,
    NativeAdaptiveAvgPool2d,
    NativeMaxPool2d,
    NativeReLU,
    build_level2_model,
    build_native_cpu,
    dyadic_conv2d,
    dyadic_conv2d_native_cpu,
    dyadic_embedding,
    dyadic_embedding_native_cpu,
    dyadic_linear,
    dyadic_linear_native_cpu,
    dyadic_output_projection,
    native_add_cpu,
    native_add_relu_cpu,
    native_adaptive_avg_pool2d_cpu,
    native_max_pool2d_cpu,
    native_relu_cpu,
    warm_native_cpu_workers,
)


def test_native_cpu_linear_matches_scalar_level2_and_level1_baseline():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(21)
    weight = torch.randn(4, 7)
    bias = torch.randn(4)
    inputs = torch.randn(3, 7)
    encoded = encode_tensor_per_output_channel(
        weight,
        max_bits=8,
        optimize_prefix_bits=(4, 6, 8),
        group_size=3,
    )

    actual = dyadic_linear_native_cpu(inputs, encoded, bias=bias, bits=6)
    scalar = dyadic_linear(inputs, encoded, bias=bias, bits=6)
    level1 = F.linear(inputs, encoded.decode(6), bias)

    torch.testing.assert_close(actual, scalar, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, level1, rtol=1e-6, atol=1e-6)


def test_native_cpu_output_projection_matches_level1_baseline():
    build_native_cpu()
    torch.manual_seed(22)
    hidden = torch.randn(2, 5)
    lm_head = torch.randn(11, 5)
    encoded = encode_tensor_per_output_channel(lm_head, max_bits=8, group_size=2)

    actual = dyadic_linear_native_cpu(hidden, encoded, bits=5)
    expected = dyadic_output_projection(hidden, encoded, bits=5)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_cpu_embedding_matches_scalar_level2_and_level1_baseline():
    build_native_cpu()
    torch.manual_seed(25)
    weight = torch.randn(10, 6)
    indices = torch.tensor([[0, 3, 4], [9, 2, 0]], dtype=torch.long)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=4)

    actual = dyadic_embedding_native_cpu(indices, encoded, bits=6)
    scalar = dyadic_embedding(indices, encoded, bits=6)
    level1 = F.embedding(indices, encoded.decode(6))

    torch.testing.assert_close(actual, scalar, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, level1, rtol=1e-6, atol=1e-6)

    padded_actual = dyadic_embedding_native_cpu(indices, encoded, bits=6, padding_idx=0)
    padded_scalar = dyadic_embedding(indices, encoded, bits=6, padding_idx=0)
    torch.testing.assert_close(padded_actual, padded_scalar, rtol=1e-6, atol=1e-6)


def test_level2_model_uses_native_cpu_linear_backend():
    build_native_cpu()
    torch.manual_seed(23)
    source = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    ).eval()
    inputs = torch.randn(3, 4)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    scalar_model, _ = build_level2_model(source, encoded, bits=6)
    native_model, report = build_level2_model(
        source,
        encoded,
        bits=6,
        linear_backend="native-cpu",
    )
    scalar_model.eval()
    native_model.eval()

    assert report.replaced_modules == ("0", "2")
    assert isinstance(native_model[0], DyadicLinear)
    assert native_model[0].linear_backend == "native-cpu"
    torch.testing.assert_close(
        native_model(inputs),
        scalar_model(inputs),
        rtol=1e-6,
        atol=1e-6,
    )


def test_level2_model_uses_native_cpu_embedding_and_linear_backends():
    build_native_cpu()
    torch.manual_seed(26)
    source = nn.Sequential(
        nn.Embedding(12, 4),
        nn.Linear(4, 3),
    ).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    scalar_model, _ = build_level2_model(source, encoded, bits=6)
    native_model, report = build_level2_model(
        source,
        encoded,
        bits=6,
        linear_backend="native-cpu",
        embedding_backend="native-cpu",
    )
    scalar_model.eval()
    native_model.eval()

    assert report.replaced_modules == ("0", "1")
    assert isinstance(native_model[0], DyadicEmbedding)
    assert native_model[0].embedding_backend == "native-cpu"
    assert isinstance(native_model[1], DyadicLinear)
    assert native_model[1].linear_backend == "native-cpu"
    torch.testing.assert_close(
        native_model(tokens),
        scalar_model(tokens),
        rtol=1e-6,
        atol=1e-6,
    )


def test_native_cpu_conv2d_matches_scalar_level2_and_level1_baseline():
    build_native_cpu()
    torch.manual_seed(27)
    inputs = torch.randn(2, 3, 6, 6)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=5)

    actual = dyadic_conv2d_native_cpu(
        inputs,
        encoded,
        bias=bias,
        bits=6,
        stride=2,
        padding=1,
    )
    scalar = dyadic_conv2d(
        inputs,
        encoded,
        bias=bias,
        bits=6,
        stride=2,
        padding=1,
    )
    level1 = F.conv2d(
        inputs,
        encoded.decode(6),
        bias=bias,
        stride=2,
        padding=1,
    )

    torch.testing.assert_close(actual, scalar, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, level1, rtol=1e-6, atol=1e-6)


def test_native_cpu_grouped_conv2d_fails_closed():
    build_native_cpu()
    torch.manual_seed(28)
    inputs = torch.randn(1, 4, 5, 5)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=6)

    try:
        dyadic_conv2d_native_cpu(
            inputs,
            encoded,
            bias=bias,
            bits=5,
            padding=1,
            groups=2,
        )
    except NotImplementedError as error:
        assert "groups=1" in str(error)
    else:
        raise AssertionError("native grouped Conv2d must fail closed")


def test_native_cpu_resnet_stride2_conv2d_matches_level1_baseline():
    build_native_cpu()
    torch.manual_seed(30)
    inputs = torch.randn(2, 64, 16, 16)
    weight = torch.randn(128, 64, 3, 3)
    bias = torch.randn(128)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8)

    actual = dyadic_conv2d_native_cpu(
        inputs,
        encoded,
        bias=bias,
        bits=6,
        stride=2,
        padding=1,
    )
    level1 = F.conv2d(
        inputs,
        encoded.decode(6),
        bias=bias,
        stride=2,
        padding=1,
    )

    torch.testing.assert_close(actual, level1, rtol=1e-4, atol=1e-4)


def test_native_cpu_maxpool2d_matches_torch_baseline():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(31)
    inputs = torch.randn(2, 3, 8, 7)

    actual = native_max_pool2d_cpu(
        inputs,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
    )
    expected = F.max_pool2d(
        inputs,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_cpu_adaptive_avg_pool2d_matches_torch_baseline():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(32)
    inputs = torch.randn(2, 3, 8, 7)

    actual = native_adaptive_avg_pool2d_cpu(inputs, (1, 1))
    expected = F.adaptive_avg_pool2d(inputs, (1, 1))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_cpu_relu_matches_torch_baseline():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(34)
    inputs = torch.randn(2, 3, 8, 7)

    actual = native_relu_cpu(inputs)
    expected = F.relu(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_cpu_add_and_add_relu_match_torch_baseline():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(35)
    left = torch.randn(2, 3, 8, 7)
    right = torch.randn(2, 3, 8, 7)

    torch.testing.assert_close(
        native_add_cpu(left, right),
        left + right,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        native_add_relu_cpu(left, right),
        F.relu(left + right),
        rtol=1e-6,
        atol=1e-6,
    )


def test_native_cpu_residual_add_relu_composes_with_conv_path():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(36)
    conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False).eval()
    inputs = torch.randn(2, 3, 8, 8)
    residual = torch.randn(2, 3, 8, 8)
    encoded = encode_tensor_per_output_channel(conv.weight.detach(), max_bits=8)

    conv_out = dyadic_conv2d_native_cpu(inputs, encoded, bits=6, padding=1)
    actual = native_add_relu_cpu(conv_out.contiguous(), residual.contiguous())
    expected = F.relu(F.conv2d(inputs, encoded.decode(6), padding=1) + residual)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_level2_model_uses_all_native_cpu_weight_backends():
    build_native_cpu()
    torch.manual_seed(29)

    class MixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, 4)
            self.proj = nn.Linear(4, 5)
            self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
            self.head = nn.Linear(5 + 3 * 4 * 4, 2)

        def forward(self, tokens: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            text = self.proj(self.embed(tokens)).mean(dim=1)
            vision = torch.relu(self.conv(image)).flatten(1)
            return self.head(torch.cat([text, vision], dim=1))

    source = MixedModel().eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    image = torch.randn(2, 2, 4, 4)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))
    scalar_model, _ = build_level2_model(source, encoded, bits=6)
    native_model, report = build_level2_model(
        source,
        encoded,
        bits=6,
        linear_backend="native-cpu",
        embedding_backend="native-cpu",
        conv_backend="native-cpu",
    )
    scalar_model.eval()
    native_model.eval()

    assert report.replaced_modules == ("embed", "proj", "conv", "head")
    assert isinstance(native_model.embed, DyadicEmbedding)
    assert native_model.embed.embedding_backend == "native-cpu"
    assert isinstance(native_model.proj, DyadicLinear)
    assert native_model.proj.linear_backend == "native-cpu"
    assert isinstance(native_model.conv, DyadicConv2d)
    assert native_model.conv.conv_backend == "native-cpu"
    torch.testing.assert_close(
        native_model(tokens, image),
        scalar_model(tokens, image),
        rtol=1e-6,
        atol=1e-6,
    )


def test_level2_model_uses_native_cpu_spatial_backends():
    build_native_cpu()
    warm_native_cpu_workers()
    torch.manual_seed(33)
    source = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
    ).eval()
    image = torch.randn(2, 2, 8, 8)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(6, 8))

    native_model, report = build_level2_model(
        source,
        encoded,
        bits=6,
        conv_backend="native-cpu",
        spatial_backend="native-cpu",
    )
    level1 = copy.deepcopy(source).eval()
    for module in level1.modules():
        if isinstance(module, nn.Conv2d):
            encoded_module = next(item for item in encoded.modules if item.name == "0")
            module.weight.data.copy_(encoded_module.tensor.decode(6))

    assert report.replaced_modules == ("0", "1", "2", "3")
    assert isinstance(native_model[0], DyadicConv2d)
    assert isinstance(native_model[1], NativeReLU)
    assert isinstance(native_model[2], NativeMaxPool2d)
    assert isinstance(native_model[3], NativeAdaptiveAvgPool2d)
    torch.testing.assert_close(native_model(image), level1(image), rtol=1e-6, atol=1e-6)



def test_native_cpu_module_matches_copied_source_level1_baseline():
    build_native_cpu()
    torch.manual_seed(24)
    source = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    inputs = torch.randn(2, 3)
    encoded = encode_model(source, max_bits=8, optimize_prefix_bits=(5, 8))

    level1 = copy.deepcopy(source).eval()
    for item in encoded.modules:
        module = dict(level1.named_modules())[item.name]
        module.weight.data.copy_(item.tensor.decode(5))

    native_model, _ = build_level2_model(
        source,
        encoded,
        bits=5,
        linear_backend="native-cpu",
    )
    native_model.eval()

    torch.testing.assert_close(native_model(inputs), level1(inputs), rtol=1e-6, atol=1e-6)
