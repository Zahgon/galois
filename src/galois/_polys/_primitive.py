"""
A module containing functions to generate and test primitive polynomials.
"""

from __future__ import annotations

import functools
from typing import Iterator

from typing_extensions import Literal

from .._domains import _factory
from .._helper import export, method_of, verify_isinstance
from .._prime import factors, is_prime, is_prime_power
from ._irreducible import is_irreducible
from ._poly import Poly
from ._search import (
    _deterministic_search,
    _deterministic_search_fixed_terms,
    _minimum_terms,
    _random_search,
    _random_search_fixed_terms,
)


@method_of(Poly)
@functools.lru_cache(maxsize=8192)
def is_primitive(f: Poly) -> bool:
    r"""
    Determines whether the polynomial $f(x)$ over $\mathrm{GF}(q)$ is primitive.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Returns:
        `True` if the polynomial is primitive.

    See Also:
        primitive_poly, primitive_polys, conway_poly, matlab_primitive_poly

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(q)$ is *primitive* if it is
        irreducible and $f(x) \mid (x^k - 1)$ for $k = q^m - 1$ and no $k$ less than
        $q^m - 1$.

    References:
        - Algorithm 4.77 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            f = galois.conway_poly(2, 8); f
            assert f.is_primitive()

            f = galois.conway_poly(3, 5); f
            assert f.is_primitive()

        The irreducible polynomial of $\mathrm{GF}(2^8)$ for AES is not primitive.

        .. ipython:: python

            f = galois.Poly.Degrees([8, 4, 3, 1, 0]); f
            assert f.is_irreducible()
            assert not f.is_primitive()
    """
    pass


@export
def primitive_poly(
    order: int,
    degree: int,
    terms: int | str | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> Poly:
    r"""
    Returns a monic primitive polynomial $f(x)$ over $\mathrm{GF}(q)$ with degree $m$.

    Arguments:
        order: The prime power order $q$ of the field $\mathrm{GF}(q)$ that the polynomial is over.
        degree: The degree $m$ of the desired primitive polynomial.
        terms: The desired number of non-zero terms $t$ in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        method: The search method for finding the primitive polynomial.

            - `"min"` (default): Returns the lexicographically first polynomial.
            - `"max"`: Returns the lexicographically last polynomial.
            - `"random"`: Returns a random polynomial.

    Returns:
        The degree-$m$ monic primitive polynomial over $\mathrm{GF}(q)$.

    Raises:
        RuntimeError: If no monic primitive polynomial of degree $m$ over $\mathrm{GF}(q)$ with
            $t$ terms exists. If `terms` is `None` or `"min"`, this should never be raised.

    See Also:
        Poly.is_primitive, matlab_primitive_poly, conway_poly

    Notes:
        If $f(x)$ is a primitive polynomial over $\mathrm{GF}(q)$ and
        $a \in \mathrm{GF}(q) \backslash \{0\}$, then $a \cdot f(x)$ is also primitive.

        In addition to other applications, $f(x)$ produces the field extension $\mathrm{GF}(q^m)$
        of $\mathrm{GF}(q)$. Since $f(x)$ is primitive, $x$ is a primitive element $\alpha$
        of $\mathrm{GF}(q^m)$ such that
        $\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}$.

    Examples:
        Find the lexicographically first, lexicographically last, and a random monic primitive polynomial.

        .. ipython:: python

            galois.primitive_poly(7, 3)
            galois.primitive_poly(7, 3, method="max")
            galois.primitive_poly(7, 3, method="random")

        Find the lexicographically first monic primitive polynomial with four terms.

        .. ipython:: python

            galois.primitive_poly(7, 3, terms=4)

        Find the lexicographically first monic irreducible polynomial with the minimum number of non-zero terms.

        .. ipython:: python

            galois.primitive_poly(7, 3, terms="min")

        Notice :func:`~galois.primitive_poly` returns the lexicographically first primitive polynomial but
        :func:`~galois.conway_poly` returns the lexicographically first primitive polynomial that is *consistent*
        with smaller Conway polynomials. This is sometimes the same polynomial.

        .. ipython:: python

            galois.primitive_poly(2, 4)
            galois.conway_poly(2, 4)

        However, it is not always.

        .. ipython:: python

            galois.primitive_poly(7, 10)
            galois.conway_poly(7, 10)

        Monic primitive polynomials scaled by non-zero field elements (now non-monic) are also primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.primitive_poly(7, 5, method="random"); f
            assert f.is_primitive()
            g = f * GF(3); g
            assert g.is_primitive()

    Group:
        polys-primitive
    """
    pass


@export
def primitive_polys(
    order: int,
    degree: int,
    terms: int | str | None = None,
    reverse: bool = False,
) -> Iterator[Poly]:
    r"""
    Iterates through all monic primitive polynomials $f(x)$ over $\mathrm{GF}(q)$ with degree $m$.

    Arguments:
        order: The prime power order $q$ of the field $\mathrm{GF}(q)$ that the polynomial is over.
        degree: The degree $m$ of the desired primitive polynomial.
        terms: The desired number of non-zero terms $t$ in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        reverse: Indicates to return the primitive polynomials from lexicographically last to first.
            The default is `False`.

    Returns:
        An iterator over all degree-$m$ monic primitive polynomials over $\mathrm{GF}(q)$.

    See Also:
        Poly.is_primitive, irreducible_polys

    Notes:
        If $f(x)$ is a primitive polynomial over $\mathrm{GF}(q)$ and
        $a \in \mathrm{GF}(q) \backslash \{0\}$, then $a \cdot f(x)$ is also primitive.

        In addition to other applications, $f(x)$ produces the field extension $\mathrm{GF}(q^m)$
        of $\mathrm{GF}(q)$. Since $f(x)$ is primitive, $x$ is a primitive element $\alpha$
        of $\mathrm{GF}(q^m)$ such that
        $\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}$.

    Examples:
        Find all monic primitive polynomials over $\mathrm{GF}(3)$ with degree 4. You may also use `tuple()` on
        the returned generator.

        .. ipython:: python

            list(galois.primitive_polys(3, 4))

        Find all monic primitive polynomials with five terms.

        .. ipython:: python

            list(galois.primitive_polys(3, 4, terms=5))

        Find all monic primitive polynomials with the minimum number of non-zero terms.

        .. ipython:: python

            list(galois.primitive_polys(3, 4, terms="min"))

        Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the
        polynomials that would have been found after the `break` condition is never incurred.

        .. ipython:: python

            for poly in galois.primitive_polys(3, 4, reverse=True):
                if poly.coeffs[1] < 2:  # Early exit condition
                    break
                print(poly)

        Or, manually iterate over the generator.

        .. ipython:: python

            generator = galois.primitive_polys(3, 4, reverse=True); generator
            next(generator)
            next(generator)
            next(generator)

    Group:
        polys-primitive
    """
    pass


@export
def matlab_primitive_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns Matlab's default primitive polynomial $f(x)$ over $\mathrm{GF}(p)$ with degree $m$.

    Arguments:
        characteristic: The prime characteristic $p$ of the field $\mathrm{GF}(p)$ that the polynomial
            is over.
        degree: The degree $m$ of the desired primitive polynomial.

    Returns:
        Matlab's default degree-$m$ primitive polynomial over $\mathrm{GF}(p)$.

    See Also:
        Poly.is_primitive, primitive_poly, conway_poly

    Notes:
        This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the lexicographically first
        primitive polynomial with minimum terms, which is equivalent to `galois.primitive_poly(p, m, terms="min")`.
        There are three notable exceptions, however:

        1. In $\mathrm{GF}(2^7)$, Matlab uses $x^7 + x^3 + 1$,
           not $x^7 + x + 1$.
        2. In $\mathrm{GF}(2^{14})$, Matlab uses $x^{14} + x^{10} + x^6 + x + 1$,
           not $x^{14} + x^5 + x^3 + x + 1$.
        3. In $\mathrm{GF}(2^{16})$, Matlab uses $x^{16} + x^{12} + x^3 + x + 1$,
           not $x^{16} + x^5 + x^3 + x^2 + 1$.

    Warning:
        This has been tested for all the $\mathrm{GF}(2^m)$ fields for $2 \le m \le 16$ (Matlab doesn't
        support larger than 16). And it has been spot-checked for $\mathrm{GF}(p^m)$. There may exist other
        exceptions. Please submit a `GitHub issue <https://github.com/mhostetter/galois/issues>`_ if you discover one.

    References:
        - Lin, S. and Costello, D. Error Control Coding. Table 2.7.

    Examples:
        .. ipython:: python

            galois.primitive_poly(2, 6, terms="min")
            galois.matlab_primitive_poly(2, 6)

        Below is one of the exceptions.

        .. ipython:: python

            galois.primitive_poly(2, 7, terms="min")
            galois.matlab_primitive_poly(2, 7)

    Group:
        polys-primitive
    """
    pass
