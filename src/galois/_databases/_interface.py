"""
A module that handles interfacing with the SQLite databases.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from threading import Lock


class DatabaseInterface:
    """
    An abstract class to interface with SQLite databases.
    """

    _lock = Lock()
    _singleton = None
    file: Path

    def __new__(cls):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
                cls.conn = sqlite3.connect(cls.file, check_same_thread=False)
                cls.cursor = cls.conn.cursor()
        return cls._singleton


class PrimeFactorsDatabase(DatabaseInterface):
    """
    A class to interface with the prime factors database.
    """

    file = Path(__file__).parent / "prime_factors.db"

    def fetch(self, n: int) -> tuple[list[int], list[int], int]:
        """
        Fetches the prime factors and multiplicities of the given integer.

        Arguments:
            n: An integer.

        Returns:
            A tuple containing the prime factors and multiplicities.
        """
        pass


class IrreduciblePolyDatabase(DatabaseInterface):
    """
    A class to interface with the irreducible polynomials database.
    """

    file = Path(__file__).parent / "irreducible_polys.db"

    def fetch(self, characteristic: int, degree: int) -> tuple[list[int], list[int]]:
        """
        Fetches the irreducible polynomial of degree `degree` over GF(`characteristic`).

        Arguments:
            characteristic: The prime characteristic of the field.
            degree: The degree of the polynomial.

        Returns:
            A tuple containing the non-zero degrees and coefficients of the irreducible polynomial.
        """
        pass


class ConwayPolyDatabase(DatabaseInterface):
    """
    A class to interface with the Conway polynomials database.
    """

    file = Path(__file__).parent / "conway_polys.db"

    def fetch(self, characteristic: int, degree: int) -> tuple[list[int], list[int]]:
        """
        Fetches the Conway polynomial of degree `degree` over GF(`characteristic`).

        Arguments:
            characteristic: The prime characteristic of the field.
            degree: The degree of the polynomial.

        Returns:
            A tuple containing the non-zero degrees and coefficients of the Conway polynomial.
        """
        pass
