from __future__ import annotations

from time import perf_counter

import torch
from torch import nn


class TinyDyadicNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(17, 6)
        self.proj = nn.Linear(6, 8)
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.head = nn.Linear(8 + 3 * 4 * 4, 5)

    def forward(self, tokens: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        token_features = self.proj(self.embed(tokens)).mean(dim=1)
        image_features = torch.relu(self.conv(image)).flatten(1)
        return self.head(torch.cat([token_features, image_features], dim=1))


def tiny_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 17, (2, 5), generator=generator)
    image = torch.randn(2, 2, 4, 4, generator=generator)
    return tokens, image


def timed_forward(
    model: nn.Module,
    tokens: torch.Tensor,
    image: torch.Tensor,
    repeats: int,
) -> tuple[torch.Tensor, float]:
    output = model(tokens, image)
    start = perf_counter()
    for _ in range(repeats):
        output = model(tokens, image)
    elapsed_ms = (perf_counter() - start) * 1000 / repeats
    return output, elapsed_ms
