"""
A module containing math and arithmetic routines on integers. Some of these functions are polymorphic and wrapped in
`_polymorphic.py`.
"""

from __future__ import annotations

import math
import sys

from ._helper import export, verify_isinstance

###############################################################################
# Divisibility
###############################################################################


def gcd(a: int, b: int) -> int:
    """
    This function is wrapped and documented in `_polymorphic.gcd()`.
    """
    pass


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """
    This function is wrapped and documented in `_polymorphic.egcd()`.
    """
    pass


def lcm(*args: int) -> int:
    """
    This function is wrapped and documented in `_polymorphic.lcm()`.
    """
    pass


def prod(*args: int) -> int:
    """
    This function is wrapped and documented in `_polymorphic.prod()`.
    """
    pass


###############################################################################
# Integer (floor) arithmetic
###############################################################################


@export
def isqrt(n: int) -> int:
    r"""
    Computes $x = \lfloor\sqrt{n}\rfloor$ such that $x^2 \le n < (x + 1)^2$.

    .. info::

        This function is included for Python versions before 3.8. For Python 3.8 and later, this function
        calls :func:`math.isqrt` from the standard library.

    Arguments:
        n: A non-negative integer.

    Returns:
        The integer square root of $n$.

    See Also:
        iroot, ilog

    Examples:
        .. ipython:: python

            n = 1000
            x = galois.isqrt(n); x
            print(f"{x**2} <= {n} < {(x + 1)**2}")

    Group:
        number-theory-integer
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        return math.isqrt(n)

    verify_isinstance(n, int)
    if not n >= 0:
        raise ValueError(f"Argument 'n' must be non-negative, not {n}.")

    if n < 2:
        return n

    # Recursively compute the integer square root
    x = isqrt(n >> 2) << 1

    if (x + 1) ** 2 <= n:
        x += 1

    return x


@export
def iroot(n: int, k: int) -> int:
    r"""
    Computes $x = \lfloor n^{\frac{1}{k}} \rfloor$ such that $x^k \le n < (x + 1)^k$.

    Arguments:
        n: A non-negative integer.
        k: The positive root $k$.

    Returns:
        The integer $k$-th root of $n$.

    See Also:
        isqrt, ilog

    Examples:
        .. ipython :: python

            n = 1000
            x = galois.iroot(n, 5); x
            print(f"{x**5} <= {n} < {(x + 1)**5}")

    Group:
        number-theory-integer
    """
    pass


@export
def ilog(n: int, b: int) -> int:
    r"""
    Computes $x = \lfloor\textrm{log}_b(n)\rfloor$ such that $b^x \le n < b^{x + 1}$.

    Arguments:
        n: A positive integer.
        b: The logarithm base $b$, must be at least 2.

    Returns:
        The integer logarithm base $b$ of $n$.

    See Also:
        iroot, isqrt

    Examples:
        .. ipython :: python

            n = 1000
            x = galois.ilog(n, 5); x
            print(f"{5**x} <= {n} < {5**(x + 1)}")

    Group:
        number-theory-integer
    """
    pass
