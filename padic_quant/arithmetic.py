from __future__ import annotations

import math

import numpy as np


def centered_mod(x: np.ndarray, modulus: int) -> np.ndarray:
    """Return the canonical signed representative modulo ``modulus``."""
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    residues = np.mod(x, modulus)
    cutoff = (modulus - 1) // 2
    return np.where(residues > cutoff, residues - modulus, residues)


def crt2_centered(
    residue_a: np.ndarray,
    residue_b: np.ndarray,
    modulus_a: int,
    modulus_b: int,
) -> np.ndarray:
    """Reconstruct centered integers from two coprime residue arrays."""
    if math.gcd(modulus_a, modulus_b) != 1:
        raise ValueError("CRT moduli must be coprime")
    inverse = pow(modulus_a, -1, modulus_b)
    coefficient = np.mod((residue_b - residue_a) * inverse, modulus_b)
    reconstructed = residue_a + modulus_a * coefficient
    return centered_mod(reconstructed, modulus_a * modulus_b)


def largest_prime_power_below(bit_budget: int, prime: int) -> tuple[int, int]:
    """Return ``(p**n, n)`` with maximum n such that p**n <= 2**bits."""
    if bit_budget < 1:
        raise ValueError("bit_budget must be positive")
    if prime < 2:
        raise ValueError("prime must be at least 2")
    exponent = int(math.floor(bit_budget / math.log2(prime)))
    exponent = max(exponent, 1)
    return prime**exponent, exponent


def entropy_bits(modulus: int) -> float:
    return math.log2(modulus)


def packed_bytes(count: int, levels: int) -> int:
    """Information-theoretic packed size for ``count`` symbols."""
    return math.ceil(count * math.log2(levels) / 8)

