"""Small, measurable experiments for modular neural-network inference."""

from .arithmetic import centered_mod, crt2_centered
from .inference import QuantizedMLP, evaluate_quantized

__all__ = [
    "QuantizedMLP",
    "centered_mod",
    "crt2_centered",
    "evaluate_quantized",
]

