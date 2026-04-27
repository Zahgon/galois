"""
A module containing polynomial arithmetic for polynomials with dense coefficients.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import int64

from .._domains import Array
from .._domains._function import Function
from .._helper import verify_isinstance


class add_jit(Function):
    """
    Computes polynomial addition of two polynomials.

    Algorithm:
        c(x) = a(x) + b(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global ADD
        ADD = self.field._add.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


def negative(a: Array) -> Array:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    pass


def subtract(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) - b(x)
    """
    pass


class subtract_jit(Function):
    """
    Computes polynomial subtraction of two polynomials.

    Algorithm:
        c(x) = a(x) - b(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global SUBTRACT
        SUBTRACT = self.field._subtract.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


def multiply(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    pass


class divmod_jit(Function):
    """
    Computes polynomial division with remainder of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> tuple[Array, Array]:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)

        a_degree = a.size - 1
        b_degree = b.size - 1

        # TODO: Merge all of this into `implementation()`
        if b_degree == 0:
            q, r = a // b, self.field([0])
        elif a_degree == 0 and a[0] == 0:
            q, r = self.field([0]), self.field([0])
        elif a_degree < b_degree:
            q, r = self.field([0]), a
        else:
            assert 1 <= a.ndim <= 2 and b.ndim == 1
            dtype = a.dtype
            a_1d = a.ndim == 1
            a = np.atleast_2d(a)
            # TODO: Do not support 2D -- it is no longer needed

            q_degree = a.shape[-1] - b.shape[-1]
            r_degree = b.shape[-1] - 1

            if self.field.ufunc_mode != "python-calculate":
                qr = self.jit(a.astype(np.int64), b.astype(np.int64))
                qr = qr.astype(dtype)
            else:
                qr = self.python(a.view(np.ndarray), b.view(np.ndarray))
            qr = self.field._view(qr)

            q = qr[:, 0 : q_degree + 1]
            r = qr[:, q_degree + 1 : q_degree + 1 + r_degree + 1]

            if a_1d:
                q = q.reshape(q.size)
                r = r.reshape(r.size)

        return q, r

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:, :], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


class floordiv_jit(Function):
    """
    Computes polynomial division without remainder of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            q = self.jit(a.astype(np.int64), b.astype(np.int64))
            q = q.astype(dtype)
        else:
            q = self.python(a.view(np.ndarray), b.view(np.ndarray))
        q = self.field._view(q)

        return q

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


class mod_jit(Function):
    """
    Computes the modular division of two polynomials.

    Algorithm:
        a(x) = q(x)*b(x) + r(x)
    """

    def __call__(self, a: Array, b: Array) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        assert a.ndim == 1 and b.ndim == 1
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            r = self.jit(a.astype(np.int64), b.astype(np.int64))
            r = r.astype(dtype)
        else:
            r = self.python(a.view(np.ndarray), b.view(np.ndarray))
        r = self.field._view(r)

        return r

    def set_globals(self):
        global SUBTRACT, MULTIPLY, RECIPROCAL
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


class pow_jit(Function):
    """
    Performs modular exponentiation on the polynomial f(x).

    Algorithm:
        d(x) = a(x)^b % c(x)
    """

    def __call__(self, a: Array, b: int, c: Array | None = None) -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, int)
        verify_isinstance(c, self.field, optional=True)
        assert a.ndim == 1 and c.ndim == 1 if c is not None else True
        dtype = a.dtype

        # Convert the integer b into a vector of int64 [MSWord, ..., LSWord] so arbitrarily large exponents may be
        # passed into the JIT-compiled version. Each element of b_vec is a 63-bit word.
        b_vec = []  # Pop on LSWord -> MSWord
        while b >= 2**63:
            q, r = divmod(b, 2**63)
            b_vec.append(r)
            b = q
        b_vec.append(b)
        b_vec = np.array(b_vec[::-1], dtype=np.int64)  # Make vector MSWord -> LSWord

        if self.field.ufunc_mode != "python-calculate":
            c_ = np.array([], dtype=np.int64) if c is None else c.astype(np.int64)
            z = self.jit(a.astype(np.int64), b_vec, c_)
            z = z.astype(dtype)
        else:
            c_ = np.array([], dtype=dtype) if c is None else c.view(np.ndarray)
            z = self.python(a.view(np.ndarray), b_vec, c_)
        z = self.field._view(z)

        return z

    def set_globals(self):
        global POLY_MULTIPLY, POLY_MOD
        POLY_MULTIPLY = self.field._convolve.function
        POLY_MOD = mod_jit(self.field).function

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:], int64[:]))

    @staticmethod
    def implementation(a, b_vec, c):
        """
        b is a vector of int64 [MSWord, ..., LSWord] so that arbitrarily large exponents may be passed
        """
        pass


class evaluate_elementwise_jit(Function):
    """
    Evaluates the polynomial f(x) elementwise at xi.
    """

    def __call__(self, coeffs: Array, x: Array) -> Array:
        dtype = x.dtype
        shape = x.shape
        x = np.atleast_1d(x.flatten())

        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(coeffs.astype(np.int64), x.astype(np.int64))
            y = y.astype(dtype)
        else:
            y = self.python(coeffs.view(np.ndarray), x.view(np.ndarray))
        y = self.field._view(y)
        y = y.reshape(shape)

        return y

    def set_globals(self):
        global ADD, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))
    _PARALLEL = True

    @staticmethod
    def implementation(coeffs, values):
        """Stub for implementation."""
        pass


class roots_jit(Function):
    """
    Finds the roots of the polynomial f(x).
    """

    def __call__(self, nonzero_degrees: np.ndarray, nonzero_coeffs: Array) -> Array:
        verify_isinstance(nonzero_degrees, np.ndarray)
        verify_isinstance(nonzero_coeffs, self.field)
        dtype = nonzero_coeffs.dtype

        if self.field.ufunc_mode != "python-calculate":
            roots = self.jit(
                nonzero_degrees.astype(np.int64), nonzero_coeffs.astype(np.int64), int(self.field.primitive_element)
            )[0, :]
            roots = roots.astype(dtype)
        else:
            roots = self.python(
                nonzero_degrees.view(np.ndarray), nonzero_coeffs.view(np.ndarray), int(self.field.primitive_element)
            )[0, :]
        roots = self.field._view(roots)
        idxs = np.argsort(roots)

        return roots[idxs]

    def set_globals(self):
        global ORDER, ADD, MULTIPLY, POWER
        ORDER = self.field.order
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        POWER = self.field._power.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:], int64[:], int64))

    @staticmethod
    def implementation(nonzero_degrees, nonzero_coeffs, primitive_element):  # pragma: no cover
        """Stub for implementation."""
        pass
