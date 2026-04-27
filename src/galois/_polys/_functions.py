"""
A module with functions for polynomials over Galois fields.
"""

from __future__ import annotations

from ._poly import Poly


def gcd(a: Poly, b: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.gcd()`.
    """
    pass


def egcd(a: Poly, b: Poly) -> tuple[Poly, Poly, Poly]:
    """
    This function is wrapped and documented in `_polymorphic.egcd()`.
    """
    pass


def lcm(*args: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.lcm()`.
    """
    pass


def prod(*args: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.prod()`.
    """
    pass
