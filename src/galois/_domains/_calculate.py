"""
A module containing various ufunc dispatchers with explicit calculation arithmetic added. Various algorithms for
each type of arithmetic are implemented here.
"""

from typing import Type

import numba
import numpy as np

from .._prime import factors
from . import _lookup
from ._array import Array

###############################################################################
# Helper JIT functions
###############################################################################

DTYPE = np.int64


@numba.jit(["int64[:](int64, int64, int64)"], nopython=True, cache=True)
def int_to_vector(a: int, characteristic: int, degree: int) -> np.ndarray:
    """
    Converts the integer representation to vector/polynomial representation.
    """
    a_vec = np.zeros(degree, dtype=DTYPE)
    for i in range(degree - 1, -1, -1):
        q, r = divmod(a, characteristic)
        a_vec[i] = r
        a = q

    return a_vec


@numba.jit(["int64(int64[:], int64, int64)"], nopython=True, cache=True)
def vector_to_int(a_vec: np.ndarray, characteristic: int, degree: int) -> int:
    """
    Converts the vector/polynomial representation to the integer representation.
    """
    a = 0
    factor = 1
    for i in range(degree - 1, -1, -1):
        a += a_vec[i] * factor
        factor *= characteristic

    return a


@numba.jit(["int64[:](int64, int64)"], nopython=True, cache=True)
def egcd(a: int, b: int) -> np.ndarray:  # pragma: no cover
    """
    Computes the Extended Euclidean Algorithm. Returns (d, s, t).

    Algorithm:
        s*x + t*y = gcd(x, y) = d
    """
    r2, r1 = a, b
    s2, s1 = 1, 0
    t2, t1 = 0, 1

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q * r1
        s2, s1 = s1, s2 - q * s1
        t2, t1 = t1, t2 - q * t1

    # Ensure the GCD is positive
    if r2 < 0:
        r2 *= -1
        s2 *= -1
        t2 *= -1

    return np.array([r2, s2, t2], dtype=DTYPE)


EGCD = egcd


@numba.jit(["int64(int64[:], int64[:])"], nopython=True, cache=True)
def crt(remainders: np.ndarray, moduli: np.ndarray) -> int:  # pragma: no cover
    """
    Computes the simultaneous solution to the system of congruences xi == ai (mod mi).
    """
    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1, m1 = remainders[0], moduli[0]
    for a2, m2 in zip(remainders[1:], moduli[1:]):
        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = gcd(m1, m2).
        d, b1, b2 = EGCD(m1, m2)

        if d == 1:
            # The moduli (m1, m2) are coprime
            x = (a1 * b2 * m2) + (a2 * b1 * m1)  # Compute x through explicit construction
            m1 = m1 * m2  # The new modulus
        else:
            # The moduli (m1, m2) are not coprime, however if a1 == b2 (mod d)
            # then a unique solution still exists.
            if not (a1 % d) == (a2 % d):
                raise ArithmeticError
            x = ((a1 * b2 * m2) + (a2 * b1 * m1)) // d  # Compute x through explicit construction
            m1 = (m1 * m2) // d  # The new modulus

        a1 = x % m1  # The new equivalent remainder

    # At the end of the process x == a1 (mod m1) where a1 and m1 are the new/modified residual
    # and remainder.

    return a1


def set_helper_globals(field: Type[Array]):
    global DTYPE, INT_TO_VECTOR, VECTOR_TO_INT, EGCD, CRT
    if field.ufunc_mode != "python-calculate":
        DTYPE = np.int64
        INT_TO_VECTOR = int_to_vector
        VECTOR_TO_INT = vector_to_int
        EGCD = egcd
        CRT = crt
    else:
        DTYPE = np.object_
        INT_TO_VECTOR = int_to_vector.py_func
        VECTOR_TO_INT = vector_to_int.py_func
        EGCD = egcd.py_func
        CRT = crt.py_func


###############################################################################
# Specific explicit calculation algorithms
###############################################################################


class add_modular(_lookup.add_ufunc):
    """
    A ufunc dispatcher that provides addition modulo the characteristic.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class add_vector(_lookup.add_ufunc):
    """
    A ufunc dispatcher that provides addition for extensions.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class negative_modular(_lookup.negative_ufunc):
    """
    A ufunc dispatcher that provides additive inverse modulo the characteristic.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int) -> int:
        """Stub for calculate."""
        pass


class negative_vector(_lookup.negative_ufunc):
    """
    A ufunc dispatcher that provides additive inverse for extensions.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int) -> int:
        """Stub for calculate."""
        pass


class subtract_modular(_lookup.subtract_ufunc):
    """
    A ufunc dispatcher that provides subtraction modulo the characteristic.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class subtract_vector(_lookup.subtract_ufunc):
    """
    A ufunc dispatcher that provides subtraction for extensions.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if self.field.ufunc_mode == "jit-lookup" or method != "__call__":
            # Use the lookup ufunc on each array entry
            return super().__call__(ufunc, method, inputs, kwargs, meta)

        # Convert entire array to polynomial/vector representation, perform array operation in GF(p), and convert
        # back to GF(p^m).
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._convert_inputs_to_vector(inputs, kwargs)
        output = getattr(ufunc, method)(*inputs, **kwargs)
        output = self._convert_output_from_vector(output, meta["dtype"])
        return output

    def set_calculate_globals(self):
        global CHARACTERISTIC, DEGREE
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class multiply_binary(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication modulo 2.

    Algorithm:
        a in GF(2^m), can be represented as a degree m-1 polynomial a(x) in GF(2)[x]
        b in GF(2^m), can be represented as a degree m-1 polynomial b(x) in GF(2)[x]
        p(x) in GF(2)[x] with degree m is the irreducible polynomial of GF(2^m)

        a * b = c
              = (a(x) * b(x)) % p(x) in GF(2)
              = c(x)
              = c
    """

    def set_calculate_globals(self):
        global ORDER, IRREDUCIBLE_POLY
        ORDER = self.field.order
        IRREDUCIBLE_POLY = self.field._irreducible_poly_int

    @staticmethod
    def calculate(a: int, b: int) -> int:
        # Re-order operands such that a > b so the while loop has less loops
        """Stub for calculate."""
        pass


class multiply_modular(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication modulo the characteristic.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class multiply_vector(_lookup.multiply_ufunc):
    """
    A ufunc dispatcher that provides multiplication for extensions.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        CHARACTERISTIC = self.field.characteristic
        DEGREE = self.field.degree
        IRREDUCIBLE_POLY = self.field._irreducible_poly_int
        set_helper_globals(self.field)

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class reciprocal_modular_egcd(_lookup.reciprocal_ufunc):
    """
    A ufunc dispatcher that provides the multiplicative inverse modulo the characteristic.
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC
        CHARACTERISTIC = self.field.characteristic

    @staticmethod
    def calculate(a: int) -> int:
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
        pass


# NOTE: Commented out because it's not currently being used. This prevents it from being
#       flagged as "not covered".
# class reciprocal_fermat(_lookup.reciprocal_ufunc):
#     """
#     A ufunc dispatcher that provides the multiplicative inverse using Fermat's Little Theorem.

#     Algorithm:
#         a in GF(p^m)
#         a^(p^m - 1) = 1

#         a * a^-1 = 1
#         a * a^-1 = a^(p^m - 1)
#             a^-1 = a^(p^m - 2)
#     """
#     def set_calculate_globals(self):
#         global ORDER, POSITIVE_POWER
#         ORDER = self.field.order
#         POSITIVE_POWER = self.field._positive_power.ufunc

#     @staticmethod
#     def calculate(a: int) -> int:
#         if a == 0:
#             raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

#         return POSITIVE_POWER(a, ORDER - 2)


class reciprocal_itoh_tsujii(_lookup.reciprocal_ufunc):
    """
    A ufunc dispatcher that provides the multiplicative inverse using the Itoh-Tsujii inversion algorithm.

    Algorithm:
        a in GF(p^m)

        1. Compute r = (p^m - 1) / (p - 1)
        2. Compute a^(r - 1) in GF(p^m)
        3. Compute a^r = a^(r - 1) * a = a.field_norm(), a^r is in GF(p)
        4. Compute (a^r)^-1 in GF(p)
        5. Compute a^-1 = (a^r)^-1 * a^(r - 1)
    """

    def set_calculate_globals(self):
        global CHARACTERISTIC, ORDER, MULTIPLY, POSITIVE_POWER, SUBFIELD_RECIPROCAL
        CHARACTERISTIC = self.field.characteristic
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        POSITIVE_POWER = self.field._positive_power.ufunc
        SUBFIELD_RECIPROCAL = getattr(self.field.prime_subfield._reciprocal, self.field.ufunc_mode.replace("-", "_"))

    @staticmethod
    def calculate(a: int) -> int:
        """Stub for calculate."""
        pass


class divide(_lookup.divide_ufunc):
    """
    A ufunc dispatcher that provides division.
    """

    def set_calculate_globals(self):
        global MULTIPLY, RECIPROCAL
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class positive_power_square_and_multiply(_lookup.power_ufunc):
    """
    A ufunc dispatcher that provides exponentiation (positive exponents only) using the Square and Multiply algorithm.

    Algorithm:
        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
           c = c_m * c_s
    """

    def set_calculate_globals(self):
        global MULTIPLY
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class power_square_and_multiply(_lookup.power_ufunc):
    """
    A ufunc dispatcher that provides exponentiation using the Square and Multiply algorithm.

    - This algorithm is applicable to fields since the exponent may be negative.

    Algorithm:
        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
             = result_m * result_s
    """

    def set_calculate_globals(self):
        global RECIPROCAL, POSITIVE_POWER
        RECIPROCAL = self.field._reciprocal.ufunc
        POSITIVE_POWER = self.field._positive_power.ufunc

    @staticmethod
    def calculate(a: int, b: int) -> int:
        """Stub for calculate."""
        pass


class log_brute_force(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using a brute-force search.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)

        i = log(beta, alpha)
        beta = alpha^i
        """
        pass


class log_pollard_rho(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using the Pollard ρ algorithm.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        set_helper_globals(self.field)

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)
        Compute x = log_alpha(beta)

        Algorithm 3.60 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf
        """
        pass


class log_pohlig_hellman(_lookup.log_ufunc):
    """
    A ufunc dispatcher that provides logarithm calculation using the Pohlig-Hellman algorithm.
    """

    def set_calculate_globals(self):
        global ORDER, MULTIPLY, RECIPROCAL, POWER, BRUTE_FORCE_LOG, FACTORS, MULTIPLICITIES
        ORDER = self.field.order
        MULTIPLY = self.field._multiply.ufunc
        RECIPROCAL = self.field._reciprocal.ufunc
        POWER = self.field._power.ufunc
        if self.field.ufunc_mode in ["jit-lookup", "jit-calculate"]:
            # We can never use the lookup table version of log because it has a fixed base
            BRUTE_FORCE_LOG = log_brute_force(self.field).jit_calculate
        else:
            BRUTE_FORCE_LOG = log_brute_force(self.field).python_calculate
        FACTORS, MULTIPLICITIES = factors(self.field.order - 1)
        set_helper_globals(self.field)
        FACTORS = np.array(FACTORS, dtype=DTYPE)
        MULTIPLICITIES = np.array(MULTIPLICITIES, dtype=DTYPE)

    @staticmethod
    def calculate(beta: int, alpha: int) -> int:  # pragma: no cover
        """
        beta is an element of GF(p^m)
        alpha is a primitive element of GF(p^m)
        The n = p1^e1 * ... * pr^er prime factorization is required
        Compute x = log_alpha(beta)

        Algorithm 3.63 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf
        """
        pass


class sqrt_binary(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides the square root in binary extension fields.
    """

    def implementation(self, a: Array) -> Array:
        """
        Fact 3.42 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        pass


class sqrt(_lookup.sqrt_ufunc):
    """
    A ufunc dispatcher that provides square root using NumPy array arithmetic.
    """

    def implementation(self, a: Array) -> Array:
        """
        Algorithm 3.34 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        Algorithm 3.36 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        pass
