"""
A module that contains a NumPy function dispatcher and an Array mixin class that override NumPy functions. The function
dispatcher classes have snake_case naming because they are act like functions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Type

import numba
import numpy as np
import numpy.typing as npt
from numba import int64

from .._helper import verify_isinstance
from .._prime import factors as _factors
from ._meta import ArrayMeta

if TYPE_CHECKING:
    from ._array import Array


class Function:
    """
    A function dispatcher for Array objects. The dispatcher will invoke a JIT-compiled or pure-Python function
    depending on the size of the Galois field or Galois ring.
    """

    _CACHE = {}  # A cache of compiled functions

    def __init__(self, field: Type[Array]):
        self.field = field

    def __call__(self):
        """
        Invokes the function, either JIT-compiled or pure-Python, performing necessary input/output conversion.
        """
        pass

    def set_globals(self):
        """
        Sets the global variables used in `implementation()` before JIT compiling it or before invoking it in
        pure Python.
        """
        pass

    _SIGNATURE: numba.types.FunctionType
    """The function's Numba signature."""

    _PARALLEL = False
    """Indicates if parallel processing should be performed."""

    implementation: Callable
    """The function's implementation in pure Python."""

    ###############################################################################
    # Various ufuncs based on implementation and compilation
    ###############################################################################

    @property
    def key_1(self):
        return (self.field.characteristic, self.field.degree, int(self.field.irreducible_poly))

    @property
    def key_2(self):
        if self.field.ufunc_mode == "jit-lookup":
            key = (str(self.__class__), self.field.ufunc_mode, int(self.field.primitive_element))
        else:
            key = (str(self.__class__), self.field.ufunc_mode)
        return key

    @property
    def function(self):
        """
        Returns a JIT-compiled or pure-Python function based on field size.
        """
        pass

    @property
    def jit(self) -> numba.types.FunctionType:
        """
        Returns a JIT-compiled function implemented over the given field.
        """
        pass

    @property
    def python(self) -> Callable:
        """
        Returns the pure-Python function implemented over the given field.
        """
        pass


###############################################################################
# Ndarray function wrappers
###############################################################################


class convolve_jit(Function):
    """
    Function dispatcher to convolve two 1-D arrays.
    """

    def __call__(self, a: Array, b: Array, mode="full") -> Array:
        """Stub for __call__."""
        pass

    def set_globals(self):
        """Stub for set_globals."""
        pass

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        """Stub for implementation."""
        pass


class fft_jit(Function):
    """
    Function dispatcher to compute the Discrete Fourier Transform of the input array.
    """

    _direction = "forward"

    def __call__(self, x: Array, n=None, axis=-1, norm=None) -> Array:
        """Stub for __call__."""
        pass

    @staticmethod
    @functools.lru_cache(None)
    def _prime_factors(length: int) -> npt.NDArray[np.int64]:
        """
        Returns the prime factors of `length` with multiplicity, e.g. 176 → (2,2,2,2,11).
        """
        pass

    def set_globals(self):
        """Stub for set_globals."""
        pass

    _SIGNATURE = numba.types.FunctionType(
        int64[:](
            int64[:],
            int64,
            numba.types.Array(int64, 1, "C", readonly=True),  # Tell Numba that this is a read-only array
        )
    )

    @staticmethod
    def implementation(array, omega, factors):
        """
        Compute the (mixed-radix) FFT of `array` using an iterative Cooley-Tukey style algorithm.

        This routine is written to operate over an abstract algebraic domain (finite fields, rings, etc.)
        by using the primitive operations `ADD/SUBTRACT/MULTIPLY/POWER` instead of Python arithmetic.

        Arguments:
            array:
                1-D input of length N.
            omega:
                A primitive N-th root of unity in the domain (so omega**N = 1, and no smaller positive
                power equals 1).
            factors:
                A factorization of N into small radices (e.g., for N = 2**k, factors = [2, 2, ..., 2]).
                The algorithm consumes these radices from the end (reverse order), which matches the
                indexing/layout used in the referenced implementation.

        Returns:
            The FFT of the input, same shape/dtype as `array` (1-D).

        Notes:
            The algorithm proceeds in stages. At each stage with radix `r`:

                - We assume we already have many independent FFTs of length `m` (initially m = 1).
                - We "stitch" groups of `r` such FFTs together to form FFTs of length `m*r`.

            Let:
                N = len(array)
                r = radix for this stage
                m = current block length (FFT size already computed per block)
                q = N / (m*r) = number of blocks after grouping

            We view the data as 3-D to make the grouping explicit:

                in_view  has shape (r, q, m)
                out_view has shape (q, r, m)

            For each fixed (qi, b) pair we combine `r` values:

                x_k = in_view[k, qi, b],   k = 0..r-1

            into `r` outputs stored as:

                out_view[qi, f, b],        f = 0..r-1

            The combination uses twiddle factors derived from omega. In this implementation,
            the twiddle “step” for the stage is:

                twiddle_step = omega^( N / (m*r) ) = omega^q

            and a running twiddle value `twiddle` is updated by multiplying by `twiddle_step` in the same
            order as the original reference implementation.

        References:
            - https://dsp-book.narod.ru/FFTBB/0270_PDF_C15.pdf
        """
        pass


class ifft_jit(fft_jit):
    """
    Function dispatcher to compute the Inverse Discrete Fourier Transform of the input array.
    """

    _direction = "backward"


###############################################################################
# Array mixin class
###############################################################################


class FunctionMixin(np.ndarray, metaclass=ArrayMeta):
    """
    An Array mixin class that overrides the invocation of NumPy functions on Array objects.
    """

    _UNSUPPORTED_FUNCTIONS = [
        # Unary
        np.packbits,
        np.unpackbits,
        np.unwrap,
        np.around,
        np.round,
        np.fix,
        np.gradient,
        np.i0,
        np.sinc,
        np.angle,
        np.real,
        np.imag,
        np.conj,
        np.conjugate,
        # Binary
        np.lib.scimath.logn,
        np.cross,
    ]

    if np.lib.NumpyVersion(np.__version__) < "2.4.0":
        _UNSUPPORTED_FUNCTIONS.append(np.trapz)
    else:
        _UNSUPPORTED_FUNCTIONS.append(np.trapezoid)

    _FUNCTIONS_REQUIRING_VIEW = [
        np.concatenate,
        np.broadcast_to,
        np.trace,
    ]

    _OVERRIDDEN_FUNCTIONS = {
        np.convolve: "_convolve",
        np.fft.fft: "_fft",
        np.fft.ifft: "_ifft",
    }

    _convolve: Function
    _fft: Function
    _ifft: Function

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._convolve = convolve_jit(cls)
        cls._fft = fft_jit(cls)
        cls._ifft = ifft_jit(cls)

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        field = type(self)

        if func in field._OVERRIDDEN_FUNCTIONS:
            output = getattr(field, field._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

        elif func in field._UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(
                f"The NumPy function {func.__name__!r} is not supported on FieldArray. "
                "If you believe this function should be supported, "
                "please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\n"
                "If you'd like to perform this operation on the data, you should first call "
                "`array = array.view(np.ndarray)` and then call the function."
            )

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._verify_array_like_types_and_values(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)

            if func in field._FUNCTIONS_REQUIRING_VIEW:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

        return output

    def dot(self, b, out=None):
        # The `np.dot(a, b)` ufunc is also available as `a.dot(b)`. Need to override this method for
        # consistent results.
        return np.dot(self, b, out=out)
