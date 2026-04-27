"""
A module containing functions to find and test Conway polynomials.
"""

from __future__ import annotations

import functools
from typing import Iterator, Sequence

from .._databases import ConwayPolyDatabase
from .._domains import _factory
from .._helper import export, method_of, verify_isinstance
from .._prime import divisors, is_prime
from ._poly import Poly
from ._primitive import is_primitive


@method_of(Poly)
def is_conway(f: Poly, search: bool = False) -> bool:
    r"""
    Checks whether the degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the
    Conway polynomial $C_{p,m}(x)$.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Arguments:
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        `True` if the polynomial $f(x)$ is the Conway polynomial $C_{p,m}(x)$.

    Raises:
        LookupError: If `search=False` and the Conway polynomial $C_{p,m}$ is not found in Frank Luebeck's
            database.

    See Also:
        conway_poly, Poly.is_conway_consistent, Poly.is_primitive

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the *Conway polynomial*
        $C_{p,m}(x)$ if it is monic, primitive, compatible with Conway polynomials $C_{p,n}(x)$ for all
        $n \mid m$, and is lexicographically first according to a special ordering.

        A Conway polynomial $C_{p,m}(x)$ is *compatible* with Conway polynomials $C_{p,n}(x)$ for
        $n \mid m$ if $C_{p,n}(x^r)$ divides $C_{p,m}(x)$, where $r = \frac{p^m - 1}{p^n - 1}$.

        The Conway lexicographic ordering is defined as follows. Given two degree-$m$ polynomials
        $g(x) = \sum_{i=0}^m g_i x^i$ and $h(x) = \sum_{i=0}^m h_i x^i$, then $g < h$ if and only if
        there exists $i$ such that $g_j = h_j$ for all $j > i$ and
        $(-1)^{m-i} g_i < (-1)^{m-i} h_i$.

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            assert f.is_primitive()
            assert g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            assert f.is_conway_consistent()
            assert g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            assert not f.is_conway()
            assert g.is_conway()
            galois.conway_poly(7, 3)
    """
    pass


@method_of(Poly)
@functools.lru_cache()
def is_conway_consistent(f: Poly, search: bool = False) -> bool:
    r"""
    Determines whether the degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is consistent
    with smaller Conway polynomials $C_{p,n}(x)$ for all $n \mid m$.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Arguments:
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        `True` if the polynomial $f(x)$ is primitive and consistent with smaller Conway polynomials
        $C_{p,n}(x)$ for all $n \mid m$.

    Raises:
        LookupError: If `search=False` and a smaller Conway polynomial $C_{p,n}$ is not found in Frank Luebeck's
            database.

    See Also:
        conway_poly, Poly.is_conway, Poly.is_primitive

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is *compatible* with Conway polynomials
        $C_{p,n}(x)$ for $n \mid m$ if $C_{p,n}(x^r)$ divides $f(x)$, where
        $r = \frac{p^m - 1}{p^n - 1}$.

        A Conway-consistent polynomial has all the properties of a Conway polynomial except that it is not
        necessarily lexicographically first (according to a special ordering).

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            assert f.is_primitive()
            assert g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            assert f.is_conway_consistent()
            assert g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            assert not f.is_conway()
            assert g.is_conway()
            galois.conway_poly(7, 3)
    """
    pass


@export
def conway_poly(characteristic: int, degree: int, search: bool = False) -> Poly:
    r"""
    Returns the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$.

    Arguments:
        characteristic: The prime characteristic $p$ of the field $\mathrm{GF}(p)$ that the polynomial
            is over.
        degree: The degree $m$ of the Conway polynomial.
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        The degree-$m$ Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$.

    See Also:
        Poly.is_conway, Poly.is_conway_consistent, Poly.is_primitive, primitive_poly

    Raises:
        LookupError: If `search=False` and the Conway polynomial $C_{p,m}$ is not found in Frank Luebeck's
            database.

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the *Conway polynomial*
        $C_{p,m}(x)$ if it is monic, primitive, compatible with Conway polynomials $C_{p,n}(x)$ for all
        $n \mid m$, and is lexicographically first according to a special ordering.

        A Conway polynomial $C_{p,m}(x)$ is *compatible* with Conway polynomials $C_{p,n}(x)$ for
        $n \mid m$ if $C_{p,n}(x^r)$ divides $C_{p,m}(x)$, where $r = \frac{p^m - 1}{p^n - 1}$.

        The Conway lexicographic ordering is defined as follows. Given two degree-$m$ polynomials
        $g(x) = \sum_{i=0}^m g_i x^i$ and $h(x) = \sum_{i=0}^m h_i x^i$, then $g < h$ if and only if
        there exists $i$ such that $g_j = h_j$ for all $j > i$ and
        $(-1)^{m-i} g_i < (-1)^{m-i} h_i$.

        The Conway polynomial $C_{p,m}(x)$ provides a standard representation of $\mathrm{GF}(p^m)$ as a
        splitting field of $C_{p,m}(x)$. Conway polynomials provide compatibility between fields and their
        subfields and, hence, are the common way to represent extension fields.

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            assert f.is_primitive()
            assert g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            assert f.is_conway_consistent()
            assert g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            assert not f.is_conway()
            assert g.is_conway()
            galois.conway_poly(7, 3)

    Group:
        polys-primitive
    """
    pass


def _conway_poly_database(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$
    from Frank Luebeck's database.

    Raises:
        LookupError: If the Conway polynomial $C_{p,m}(x)$ is not found in Frank Luebeck's database.
    """
    pass


@functools.lru_cache()
def _conway_poly_search(characteristic: int, degree: int) -> Poly:
    r"""
    Manually searches for the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$.
    """
    pass


def _conway_lexicographic_order(
    characteristic: int,
    degree: int,
) -> Iterator[Poly]:
    r"""
    Yields all monic polynomials of degree $m$ over $\mathrm{GF}(p)$ in the lexicographic order
    defined for Conway polynomials.
    """
    pass
