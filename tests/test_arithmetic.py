import numpy as np

from dyadic_quant.arithmetic import centered_mod, crt2_centered


def test_centered_mod_even_and_odd_moduli():
    values = np.arange(-20, 21)
    for modulus in (8, 9, 25):
        centered = centered_mod(values, modulus)
        assert np.all(np.mod(centered, modulus) == np.mod(values, modulus))
        assert centered.min() >= -(modulus // 2)
        assert centered.max() <= (modulus - 1) // 2


def test_crt_reconstructs_centered_values():
    values = np.arange(-126, 127)
    reconstructed = crt2_centered(values % 11, values % 23, 11, 23)
    np.testing.assert_array_equal(reconstructed, values)

