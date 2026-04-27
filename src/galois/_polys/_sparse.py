"""
A module containing polynomial arithmetic for polynomials with sparse coefficients.
"""

from __future__ import annotations

import numpy as np

from .._domains import Array


def add(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) + b(x)
    """
    pass


def negative(
    a_degrees: np.ndarray,
    a_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    pass


def subtract(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) - b(x)
    """
    pass


def multiply(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    pass
