"""
A module containing polynomial arithmetic for polynomials over GF(2).
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """
    c(x) = a(x) + b(x)
    """
    pass


def negative(a: int) -> int:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    pass


def subtract(a: int, b: int) -> int:
    """
    c(x) = a(x) - b(x)
    """
    pass


def multiply(a: int, b: int) -> int:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    pass


def divmod(a: int, b: int) -> tuple[int, int]:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    pass


def floordiv(a: int, b: int) -> int:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    pass


def mod(a: int, b: int) -> int:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    pass


def pow(a: int, b: int, c: int | None = None) -> int:
    """
    d(x) = a(x)^b % c(x)
    """
    pass
