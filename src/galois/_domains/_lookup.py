"""
A module containing various ufunc dispatchers with lookup table arithmetic added. These "lookup" implementations use
exponential, logarithm (base primitive element), and Zech logarithm (base primitive element) lookup tables to reduce
the complex finite field arithmetic to a few table lookups and an integer addition/subtraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _ufunc

if TYPE_CHECKING:
    from ._array import Array


class add_ufunc(_ufunc.add_ufunc):
    """
    Addition ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global EXP, LOG, ZECH_LOG, ZECH_E
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_LOG = self.field._ZECH_LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a + b = α^m + α^n
              = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
              = α^m * α^ZECH_LOG(n - m)
              = α^(m + ZECH_LOG(n - m))
        """
        pass


class negative_ufunc(_ufunc.negative_ufunc):
    """
    Additive inverse ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global EXP, LOG, ZECH_E
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        -a = -α^m
           = -1 * α^m
           = α^e * α^m
           = α^(e + m)
        """
        pass


class subtract_ufunc(_ufunc.subtract_ufunc):
    """
    Subtraction ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global ORDER, EXP, LOG, ZECH_LOG, ZECH_E
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_LOG = self.field._ZECH_LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a - b = α^m - α^n
              = α^m + (-α^n)
              = α^m + (-1 * α^n)
              = α^m + (α^e * α^n)
              = α^m + α^(e + n)
        """
        pass


class multiply_ufunc(_ufunc.multiply_ufunc):
    """
    Multiplication ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global EXP, LOG
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a * b = α^m * α^n
              = α^(m + n)
        """
        pass


class reciprocal_ufunc(_ufunc.reciprocal_ufunc):
    """
    Multiplicative inverse ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        1 / a = 1 / α^m
              = α^(-m)
              = 1 * α^(-m)
              = α^(ORDER - 1) * α^(-m)
              = α^(ORDER - 1 - m)
        """
        pass


class divide_ufunc(_ufunc.divide_ufunc):
    """
    Division ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a / b = α^m / α^n
              = α^(m - n)
              = 1 * α^(m - n)
              = α^(ORDER - 1) * α^(m - n)
              = α^(ORDER - 1 + m - n)
        """
        pass


class power_ufunc(_ufunc.power_ufunc):
    """
    Exponentiation ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b in Z

        a ** b = α^m ** b
               = α^(m * b)
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1) + b % (ORDER - 1)))
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1)) * α^(m * (b % (ORDER - 1)))
               = 1 * α^(m * (b % (ORDER - 1)))
               = α^(m * (b % (ORDER - 1)))
        """
        pass


class log_ufunc(_ufunc.log_ufunc):
    """
    Logarithm ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
        global LOG
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        b is a primitive element of GF(p^m)
        a = b^c

        log(a, b) = log(b^m, b)
                  = c
        """
        pass


class sqrt_ufunc(_ufunc.sqrt_ufunc):
    """
    Square root ufunc dispatcher with lookup table arithmetic added.
    """

    def implementation(self, a: Array) -> Array:
        """
        Computes the square root of an element in a Galois field or Galois ring.
        """
        pass


###############################################################################
# Array mixin class
###############################################################################


class UFuncMixin(_ufunc.UFuncMixin):
    """
    The UFuncMixin class with lookup table construction added.
    """

    @classmethod
    def _build_lookup_tables(cls):
        """
        Construct EXP, LOG, and ZECH_LOG lookup tables to be used in the "lookup" arithmetic functions
        """
        pass
