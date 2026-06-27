import torch

from dyadic_quant.dyadic_torch import (
    encode_tensor_per_output_channel,
    storage_bytes,
)


def test_per_channel_scale_is_power_of_two_and_prefixes_are_nested():
    weight = torch.tensor(
        [
            [[[-0.9, 0.2], [0.4, 0.7]]],
            [[[-0.04, 0.01], [0.02, 0.03]]],
        ]
    )
    encoded = encode_tensor_per_output_channel(weight, max_bits=8)
    # One exponent per output channel by default (group spans the full row).
    assert encoded.exponents.shape == (2, 1)
    assert encoded.group_size == 4
    steps = torch.pow(2.0, encoded.exponents.float().squeeze(-1))
    assert torch.all(steps == torch.tensor([2.0**-7, 2.0**-11]))

    previous_error = None
    for bits in range(2, 9):
        reconstructed = encoded.decode(bits)
        error = torch.max(torch.abs(reconstructed - weight)).item()
        if previous_error is not None:
            assert error <= previous_error + 1e-7
        previous_error = error

    code_4 = encoded.magnitude_code >> 4
    code_5 = encoded.magnitude_code >> 3
    torch.testing.assert_close(code_4, code_5 >> 1)


def test_block_wise_exponents_isolate_outliers_per_group():
    # One row with a large outlier in the first half and tiny weights in the
    # second half. A per-channel scale must cover the outlier and therefore
    # quantizes the small weights coarsely; block-wise scaling gives the small
    # group its own fine exponent.
    weight = torch.tensor([[8.0, 7.0, 0.02, 0.03, 0.02, 0.05, 0.04, 0.03]])

    per_channel = encode_tensor_per_output_channel(weight, max_bits=8)
    assert per_channel.exponents.shape == (1, 1)

    block = encode_tensor_per_output_channel(weight, max_bits=8, group_size=4)
    assert block.exponents.shape == (1, 2)
    assert block.group_size == 4
    # The low-magnitude group earns a much smaller step than the outlier group.
    assert block.exponents[0, 1] < block.exponents[0, 0]

    small = weight[:, 4:]
    per_channel_err = (per_channel.decode(8)[:, 4:] - small).abs().mean()
    block_err = (block.decode(8)[:, 4:] - small).abs().mean()
    assert block_err < per_channel_err

    # Prefixes remain nested regardless of grouping.
    torch.testing.assert_close(
        block.magnitude_code >> 4, (block.magnitude_code >> 3) >> 1
    )


def test_uneven_group_size_pads_and_reconstructs():
    weight = torch.randn(3, 10)
    encoded = encode_tensor_per_output_channel(weight, max_bits=8, group_size=4)
    # 10 elements / group 4 -> 3 blocks (last block padded).
    assert encoded.exponents.shape == (3, 3)
    assert encoded.decode(8).shape == weight.shape


def test_storage_adds_exactly_one_plane_per_refinement():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    from dyadic_quant.dyadic_torch import encode_model

    encoded = encode_model(model, max_bits=8)
    size_4 = storage_bytes(model, encoded, bits=4)
    size_5 = storage_bytes(model, encoded, bits=5)
    assert (
        size_5["weight_payload_bytes"] - size_4["weight_payload_bytes"]
        == size_4["incremental_plane_bytes"]
    )


def test_tied_embedding_and_output_weight_is_encoded_once():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 16, bias=False)
            self.proj.weight = self.embed.weight

    from dyadic_quant.dyadic_torch import encode_model

    model = TiedModel()
    encoded = encode_model(model, max_bits=8)
    assert encoded.quantized_weight_count == model.embed.weight.numel()
    assert len(encoded.modules) == 1
    sizes = storage_bytes(model, encoded, bits=6)
    assert sizes["weight_payload_bytes"] == 16 * 8 * 6 // 8


def test_packed_artifact_contains_sign_and_maximum_depth_code(tmp_path):
    from dyadic_quant.dyadic_torch import encode_model, save_encoded_model

    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    encoded = encode_model(
        model,
        max_bits=8,
        optimize_prefix_bits=(4, 5, 6, 8),
    )
    output = tmp_path / "model.dyadic.pt"
    save_encoded_model(encoded, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    packed = payload["modules"][0]["packed_sign_magnitude"]
    sign = torch.bitwise_right_shift(packed, 7)
    magnitude = torch.bitwise_and(packed, 0x7F)
    assert payload["format"] == "progressive_dyadic_sign_magnitude_v1"
    assert payload["max_bits"] == 8
    assert torch.equal(
        sign,
        (encoded.modules[0].tensor.signs < 0).to(torch.uint8),
    )
    assert torch.equal(
        magnitude,
        encoded.modules[0].tensor.magnitude_code.to(torch.uint8),
    )
