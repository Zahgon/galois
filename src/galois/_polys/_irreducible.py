"""
A module containing functions to generate and test irreducible polynomials.
"""

from __future__ import annotations

import functools
from typing import Iterator

from typing_extensions import Literal

from .._databases import IrreduciblePolyDatabase
from .._domains import _factory
from .._helper import export, method_of, verify_isinstance
from .._prime import factors, is_prime_power
from ._functions import gcd
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
def is_irreducible(f: Poly) -> bool:
    r"""
    Determines whether the polynomial $f(x)$ over $\mathrm{GF}(p^m)$ is irreducible.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Returns:
        `True` if the polynomial is irreducible.

    See Also:
        irreducible_poly, irreducible_polys

    Notes:
        A polynomial $f(x) \in \mathrm{GF}(p^m)[x]$ is *reducible* over $\mathrm{GF}(p^m)$ if it can
        be represented as $f(x) = g(x) h(x)$ for some $g(x), h(x) \in \mathrm{GF}(p^m)[x]$ of strictly
        lower degree. If $f(x)$ is not reducible, it is said to be *irreducible*. Since Galois fields are not
        algebraically closed, such irreducible polynomials exist.

        This function implements Rabin's irreducibility test. It says a degree-$m$ polynomial $f(x)$
        over $\mathrm{GF}(q)$ for prime power $q$ is irreducible if and only if
        $f(x) \mid (x^{q^m} - x)$ and $\textrm{gcd}(f(x),\ x^{q^{m_i}} - x) = 1$ for
        $1 \le i \le k$, where $m_i = m/p_i$ for the $k$ prime divisors $p_i$ of $m$.

    References:
        - Rabin, M. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273-280.
          https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
        - Gao, S. and Panarino, D. Tests and constructions of irreducible polynomials over finite fields.
          https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
        - Section 4.5.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
        - https://en.wikipedia.org/wiki/Factorization_of_polynomials_over_finite_fields

    Examples:
        .. ipython:: python

            # Conway polynomials are always irreducible (and primitive)
            f = galois.conway_poly(2, 5); f

            # f(x) has no roots in GF(2), a necessary but not sufficient condition of being irreducible
            f.roots()

            assert f.is_irreducible()

        .. ipython:: python

            g = galois.irreducible_poly(2**4, 2, method="random"); g
            h = galois.irreducible_poly(2**4, 3, method="random"); h
            f = g * h; f

            assert not f.is_irreducible()
    """
    pass


@export
def irreducible_poly(
    order: int,
    degree: int,
    terms: int | str | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> Poly:
    r"""
    Returns a monic irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ with degree $m$.

    Arguments:
        order: The prime power order $q$ of the field $\mathrm{GF}(q)$ that the polynomial is over.
        degree: The degree $m$ of the desired irreducible polynomial.
        terms: The desired number of non-zero terms $t$ in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        method: The search method for finding the irreducible polynomial.

            - `"min"` (default): Returns the lexicographically first polynomial.
            - `"max"`: Returns the lexicographically last polynomial.
            - `"random"`: Returns a random polynomial.

    .. fast-performance::

        Depending on the type of polynomial requested, this function may use a database of precomputed polynomials.
        Under these conditions, this function returns very quickly.

        - For $q = 2$, $2 \le m \le 10000$, `terms="min"`, and `method="min"`, the
          `HP Table of Low-Weight Binary Irreducible Polynomials
          <https://www.hpl.hp.com/techreports/98/HPL-98-135.pdf>`_ is used.

    Returns:
        The degree-$m$ monic irreducible polynomial over $\mathrm{GF}(q)$.

    Raises:
        RuntimeError: If no monic irreducible polynomial of degree $m$ over $\mathrm{GF}(q)$ with
            $t$ terms exists. If `terms` is `None` or `"min"`, this should never be raised.

    See Also:
        Poly.is_irreducible, primitive_poly, conway_poly

    Notes:
        If $f(x)$ is an irreducible polynomial over $\mathrm{GF}(q)$ and
        $a \in \mathrm{GF}(q) \backslash \{0\}$, then $a \cdot f(x)$ is also irreducible.

        In addition to other applications, $f(x)$ produces the field extension $\mathrm{GF}(q^m)$ of
        $\mathrm{GF}(q)$.

    Examples:
        Find the lexicographically first, lexicographically last, and a random monic irreducible polynomial.

        .. ipython:: python

            galois.irreducible_poly(7, 3)
            galois.irreducible_poly(7, 3, method="max")
            galois.irreducible_poly(7, 3, method="random")

        Find the lexicographically first monic irreducible polynomial with four terms.

        .. ipython:: python

            galois.irreducible_poly(7, 3, terms=4)

        Find the lexicographically first monic irreducible polynomial with the minimum number of non-zero terms.

        .. ipython:: python

            galois.irreducible_poly(7, 3, terms="min")

        Monic irreducible polynomials scaled by non-zero field elements (now non-monic) are also irreducible.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.irreducible_poly(7, 5, method="random"); f
            assert f.is_irreducible()
            g = f * GF(3); g
            assert g.is_irreducible()

    Group:
        polys-irreducible
    """
    pass


@export
def irreducible_polys(
    order: int,
    degree: int,
    terms: int | str | None = None,
    reverse: bool = False,
) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials $f(x)$ over $\mathrm{GF}(q)$ with degree $m$.

    Arguments:
        order: The prime power order $q$ of the field $\mathrm{GF}(q)$ that the polynomial is over.
        degree: The degree $m$ of the desired irreducible polynomial.
        terms: The desired number of non-zero terms $t$ in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        reverse: Indicates to return the irreducible polynomials from lexicographically last to first.
            The default is `False`.

    Returns:
        An iterator over all degree-$m$ monic irreducible polynomials over $\mathrm{GF}(q)$.

    See Also:
        Poly.is_irreducible, primitive_polys

    Notes:
        If $f(x)$ is an irreducible polynomial over $\mathrm{GF}(q)$ and
        $a \in \mathrm{GF}(q) \backslash \{0\}$, then $a \cdot f(x)$ is also irreducible.

        In addition to other applications, $f(x)$ produces the field extension $\mathrm{GF}(q^m)$ of
        $\mathrm{GF}(q)$.

    Examples:
        Find all monic irreducible polynomials over $\mathrm{GF}(3)$ with degree 4. You may also use `tuple()` on
        the returned generator.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4))

        Find all monic irreducible polynomials with four terms.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4, terms=4))

        Find all monic irreducible polynomials with the minimum number of non-zero terms.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4, terms="min"))

        Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the
        polynomials that would have been found after the `break` condition is never incurred.

        .. ipython:: python

            for poly in galois.irreducible_polys(3, 4, reverse=True):
                if poly.coeffs[1] < 2:  # Early exit condition
                    break
                print(poly)

        Or, manually iterate over the generator.

        .. ipython:: python

            generator = galois.irreducible_polys(3, 4, reverse=True); generator
            next(generator)
            next(generator)
            next(generator)

    Group:
        polys-irreducible
    """
    pass
