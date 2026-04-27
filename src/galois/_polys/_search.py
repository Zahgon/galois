"""
A module containing functions that search for monic irreducible or primitive polynomials.

Searches are performed in lexicographic, reverse lexicographic, or random order. Additionally, a fixed number
of non-zero terms may be specified. Memoization is heavily used to prevent repeated, expensive searches.
"""

from __future__ import annotations

import functools
import random
from typing import Iterator, Sequence, Type

import numpy as np

from .._domains import Array, _factory
from ._poly import Poly


@functools.lru_cache(maxsize=8192)
def _deterministic_search(
    field: Type[Array],
    start: int,
    stop: int,
    step: int,
    test: str,
) -> Poly | None:
    """
    Searches for a monic polynomial in the specified range, returning the first one that passes the specified test
    (either 'is_irreducible()' or 'is_primitive()'). This function returns `None` if no such polynomial exists.
    """
    pass


def _deterministic_search_fixed_terms(
    order: int,
    degree: int,
    terms: int,
    test: str,
    reverse: bool = False,
) -> Iterator[Poly]:
    """
    Iterates over all monic polynomials of the given degree and number of non-zero terms in lexicographic
    order, only yielding those that pass the specified test (either 'is_irreducible()' or 'is_primitive()').

    One of the non-zero degrees is the x^m term and the other is the x^0 term. The x^0 term is required so that
    the polynomial is irreducible.
    """
    pass


def _random_search(order: int, degree: int, test: str) -> Iterator[Poly]:
    """
    Searches for a random monic polynomial of specified degree, only yielding those that pass the specified test
    (either 'is_irreducible()' or 'is_primitive()').
    """
    pass


def _random_search_fixed_terms(
    order: int,
    degree: int,
    terms: int,
    test: str,
) -> Iterator[Poly]:
    """
    Searches for a random monic polynomial of specified degree and number of non-zero terms, only yielding those that
    pass the specified test (either 'is_irreducible()' or 'is_primitive()').
    """
    pass


@functools.lru_cache(maxsize=8192)
def _minimum_terms(order: int, degree: int, test: str) -> int:
    """
    Finds the minimum number of terms of an irreducible or primitive polynomial of specified degree over the
    finite field of specified order.
    """
    pass
