"""
A module containing functions to generate and test primitive elements of finite fields.
"""

from __future__ import annotations

import random

from typing_extensions import Literal

from .._helper import export, verify_isinstance
from .._modular import totatives
from .._polys import Poly
from .._prime import factors
from ..typing import PolyLike


@export
def is_primitive_element(element: PolyLike, irreducible_poly: Poly) -> bool:
    r"""
    Determines whether an element of the extension field $\mathrm{GF}(q^m)$ is primitive.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let $\alpha$ denote
    the residue class of $x$ in the quotient ring

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    Every field element can be written uniquely as $g(\alpha)$ with $\deg g < m$. This function
    determines whether the element represented by the input polynomial $g(x)$ has multiplicative
    order $q^m - 1$, i.e., generates $\mathrm{GF}(q^m)^\times$.

    Arguments:
        element:
            A polynomial $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$. This may be any
            :obj:`~galois.typing.PolyLike` and will be converted to a :class:`~galois.Poly` over the same field
            as `irreducible_poly`. The corresponding field element is the residue class
            $g(\alpha)$ in $\mathrm{GF}(q)[x] / (f(x))$, where $\alpha$ is the image of $x$ modulo
            $f(x)$.
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$

    Returns:
        `True` if the residue class $g(\alpha)$ is a primitive element of $\mathrm{GF}(q^m)$,
        otherwise `False`.

    Notes:
        An element $g(\alpha)$ is **primitive** if it generates the entire multiplicative group, i.e.,
        its multiplicative order is exactly $N$.

        This function implements the standard group-theoretic test for primitivity, using the
        representative polynomial $g(x)$ modulo $f(x)$.

        1. Factor $N$ into primes

           $$
           N = \prod_i p_i^{e_i},
           $$

           and let $\{p_i\}$ be the set of distinct prime factors.

        2. For each prime factor $p_i$ of $N$, compute

           $$
           h_i = g(\alpha)^{N / p_i}.
           $$

           If $h_i = 1$ for any $i$, then the order of $g(\alpha)$ is a proper divisor of $N$, so
           $g(\alpha)$ is **not** primitive.

        3. If none of these powers equals $1$, and $g(\alpha)^N = 1$, then the multiplicative
           order of $g(\alpha)$ is exactly $N$ and the element is primitive.

        In this implementation, `element` is the polynomial $g(x)$, `irreducible_poly` is
        $f(x)$, and exponentiation is performed in the quotient ring
        $\mathrm{GF}(q)[x] / (f(x))$ via the built-in `pow()` with the modulus `irreducible_poly`.

        The expression `pow(element, k, irreducible_poly)` computes the residue class of $g(x)^k$ modulo $f(x)$,
        which corresponds to $g(\alpha)^k$ in $\mathrm{GF}(q^m)$.

    See Also:
        primitive_element, primitive_elements, FieldArray.primitive_element

    Examples:
        In the extension field $\mathrm{GF}(3^4)$, the polynomial $g(x) = x + 2$ represents a
        primitive element whose order is $3^4 - 1 = 80$.

        .. ipython:: python

            GF = galois.GF(3**4)
            f = GF.irreducible_poly; f
            assert galois.is_primitive_element("x + 2", f)
            GF("x + 2").multiplicative_order()

        However, the polynomial $x + 1$ does not represent a primitive element, as its
        multiplicative order is only 20.

        .. ipython:: python

            assert not galois.is_primitive_element("x + 1", f)
            GF("x + 1").multiplicative_order()

    Group:
        galois-fields-primitive-elements
    """
    pass


def _is_primitive_element(element: Poly, irreducible_poly: Poly) -> bool:
    r"""
    A private version of :func:`is_primitive_element` without type checking/conversion for internal use.

    The arguments are:
        - `element`: a polynomial $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$.
        - `irreducible_poly`: the irreducible polynomial $f(x)$ defining the field
          $\mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x))$.

    The return value is `True` if the residue class of $g(x)$ modulo $f(x)$ is primitive in
    $\mathrm{GF}(q^m)$.
    """
    pass


@export
def primitive_element(irreducible_poly: Poly, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Finds a primitive element of the extension field $\mathrm{GF}(q^m)$ defined by an
    irreducible polynomial.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    This function searches for a polynomial $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$ whose
    residue class modulo $f(x)$ is primitive, i.e., generates the multiplicative group
    $\mathrm{GF}(q^m)^\times$ of order $q^m - 1$.

    Arguments:
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$
        method:
            The search method for finding a primitive representative polynomial. Choices are:

            - `"min"`: Returns a primitive representative with the smallest integer encoding.
            - `"max"`: Returns a primitive representative with the largest integer encoding.
            - `"random"`: Returns a random primitive representative.

    Returns:
        A polynomial $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$ such that its residue
        class modulo $f(x)$ is a primitive element of $\mathrm{GF}(q^m)$.

    Notes:
        Integers in the range $[0, q^m - 1]$ correspond to polynomials over $\mathrm{GF}(q)$ with
        degree less than $m$ via the usual base-$q$ expansion. Each such polynomial $g(x)$
        represents a unique residue class $g(\alpha)$ in $\mathrm{GF}(q)[x] / (f(x))$.

        Constant polynomials (integers $0, 1, \dots, q - 1$) represent elements of the base
        field $\mathrm{GF}(q)$ and can never be primitive in $\mathrm{GF}(q^m)$ for $m > 1$,
        since their multiplicative orders divide $q - 1$.

        Therefore, this function searches over integers in the range $[q, q^m - 1]$, which
        correspond to non-constant polynomials $g(x)$ of degree at most $m - 1$.

    See Also:
        is_primitive_element, primitive_elements, FieldArray.primitive_element

    Examples:
        Construct the extension field $\mathrm{GF}(7^5)$.

        .. ipython:: python

            f = galois.irreducible_poly(7, 5, method="max"); f
            GF = galois.GF(7**5, irreducible_poly=f, repr="poly")
            print(GF.properties)

        Find the smallest primitive representative polynomial for the degree-5 extension of
        $\mathrm{GF}(7)$ with irreducible polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            assert g.multiplicative_order() == GF.order - 1

        Find the largest primitive representative polynomial for the degree-5 extension of
        $\mathrm{GF}(7)$ with irreducible polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f, method="max"); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            assert g.multiplicative_order() == GF.order - 1

        Find a random primitive representative polynomial for the degree-5 extension of
        $\mathrm{GF}(7)$ with irreducible polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f, method="random"); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            assert g.multiplicative_order() == GF.order - 1
            @suppress
            GF.repr()

    Group:
        galois-fields-primitive-elements
    """
    pass


@export
def primitive_elements(irreducible_poly: Poly) -> list[Poly]:
    r"""
    Enumerates all primitive elements of the extension field $\mathrm{GF}(q^m)$ defined by an
    irreducible polynomial.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    This function returns all polynomials $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$ whose
    residue classes modulo $f(x)$ are primitive generators of the multiplicative group
    $\mathrm{GF}(q^m)^\times$.

    Arguments:
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$

    Returns:
        List of all polynomials $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$ whose residue
        classes modulo $f(x)$ are primitive elements of $\mathrm{GF}(q^m)$.

    Notes:
        The multiplicative group $\mathrm{GF}(q^m)^\times$ is cyclic of order $N = q^m - 1$.
        If $g(\alpha)$ is any fixed primitive element, then **all** primitive elements of
        $\mathrm{GF}(q^m)$ are precisely the powers

        $$
        g(\alpha)^k \quad \text{with} \quad \gcd(k, N) = 1.
        $$

        The number of primitive elements is $\varphi(N)$, where $\varphi$ is the Euler totient
        function. See :obj:`~galois.euler_phi`.

        This function:

        1. Finds one primitive representative polynomial $g(x)$ using :func:`primitive_element`.
        2. Iterates over all integers $k$ in the totatives of $N$ (i.e., $1 \le k < N$ and
           $\gcd(k, N) = 1$).
        3. Computes $g(x)^k \bmod f(x)$ in $\mathrm{GF}(q)[x] / (f(x))$.
        4. Returns the corresponding list of primitive representative polynomials.

    See Also:
        is_primitive_element, primitive_element, FieldArray.primitive_elements

    Examples:
        Construct the extension field $\mathrm{GF}(3^4)$.

        .. ipython:: python

            f = galois.irreducible_poly(3, 4, method="max"); f
            GF = galois.GF(3**4, irreducible_poly=f, repr="poly")
            print(GF.properties)

        Find all primitive representative polynomials for the degree-4 extension of $\mathrm{GF}(3)$.

        .. ipython:: python

            g = galois.primitive_elements(f); g

        The number of primitive elements is given by $\varphi(q^m - 1)$.

        .. ipython:: python

            phi = galois.euler_phi(3**4 - 1); phi
            assert len(g) == phi

        Shows that each representative polynomial corresponds to an element of multiplicative
        order $q^m - 1$ in $\mathrm{GF}(3^4)$.

        .. ipython:: python

            # Convert the polynomials over GF(3) into elements of GF(3^4)
            g = GF([int(gi) for gi in g]); g
            assert np.all(g.multiplicative_order() == GF.order - 1)
            @suppress
            GF.repr()

    Group:
        galois-fields-primitive-elements
    """
    pass
