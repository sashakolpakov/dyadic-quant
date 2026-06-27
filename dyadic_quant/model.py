from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallMLP(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class TrainedModel:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    calibration_inputs: list[np.ndarray]
    float_logits: np.ndarray
    float_accuracy: float
    parameter_count: int
    weight_count: int
    bias_count: int


def _numpy_linear_parameters(model: SmallMLP) -> tuple[list[np.ndarray], list[np.ndarray]]:
    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    weights = [
        layer.weight.detach().cpu().numpy().astype(np.float64) for layer in linear_layers
    ]
    biases = [
        layer.bias.detach().cpu().numpy().astype(np.float64) for layer in linear_layers
    ]
    return weights, biases


def _collect_inputs_and_logits(
    model: SmallMLP, inputs: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    current = torch.from_numpy(inputs.astype(np.float32))
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                collected.append(current.cpu().numpy().astype(np.float64))
            current = layer(current)
    return collected, current.cpu().numpy().astype(np.float64)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    widths: list[int],
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 3e-3,
    batch_size: int = 128,
) -> TrainedModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    model = SmallMLP(widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(epochs):
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    calibration_inputs, _ = _collect_inputs_and_logits(model, x_train)
    _, test_logits = _collect_inputs_and_logits(model, x_test)
    predictions = np.argmax(test_logits, axis=1)
    weights, biases = _numpy_linear_parameters(model)
    return TrainedModel(
        weights=weights,
        biases=biases,
        calibration_inputs=calibration_inputs,
        float_logits=test_logits,
        float_accuracy=float(np.mean(predictions == y_test)),
        parameter_count=sum(weight.size + bias.size for weight, bias in zip(weights, biases)),
        weight_count=sum(weight.size for weight in weights),
        bias_count=sum(bias.size for bias in biases),
    )
