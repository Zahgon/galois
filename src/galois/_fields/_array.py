"""
A module that defines the abstract base class FieldArray.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
from typing_extensions import Literal, Self

from .._domains import Array, _linalg
from .._helper import export, extend_docstring, verify_isinstance, verify_literal
from .._polys import Poly
from .._polys._conversions import integer_to_poly, poly_to_str, str_to_integer
from .._prime import divisors
from ..typing import ArrayLike, DTypeLike, ElementLike, IterableLike, ShapeLike
from ._meta import FieldArrayMeta

# from ._normal_element import is_normal_element

DOCSTRING_MAP = {
    "Array": "FieldArray",
}


@export
class FieldArray(Array, metaclass=FieldArrayMeta):
    r"""
    An abstract :obj:`~numpy.ndarray` subclass over $\mathrm{GF}(p^m)$.

    .. abstract::

        :obj:`~galois.FieldArray` is an abstract base class and cannot be instantiated directly. Instead,
        :obj:`~galois.FieldArray` subclasses are created using the class factory :func:`~galois.GF`.

    Examples:
        Create a :obj:`~galois.FieldArray` subclass over $\mathrm{GF}(3^5)$ using the class factory
        :func:`~galois.GF`.

        .. ipython-with-reprs:: int,poly,power

            GF = galois.GF(3**5)
            issubclass(GF, galois.FieldArray)
            print(GF.properties)

        Create a :obj:`~galois.FieldArray` instance using `GF`'s constructor.

        .. ipython-with-reprs:: int,poly,power

            x = GF([44, 236, 206, 138]); x
            isinstance(x, GF)

    Group:
        galois-fields
    """

    def __new__(
        cls,
        x: ElementLike | ArrayLike,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ) -> Self:
        if cls is FieldArray:
            raise NotImplementedError(
                "FieldArray is an abstract base class that cannot be directly instantiated. "
                "Instead, create a FieldArray subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` "
                "and instantiate an array using `x = GF(array_like)`."
            )
        return super().__new__(cls, x, dtype, copy, order, ndmin)

    def __init__(
        self,
        x: ElementLike | ArrayLike,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ):
        r"""
        Creates an array over $\mathrm{GF}(p^m)$.

        Arguments:
            x: A finite field scalar or array.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.FieldArray` subclass (the first element in
                :obj:`~galois.FieldArray.dtypes`).
            copy: The `copy` keyword argument from :func:`numpy.array`. The default is `True`.
            order: The `order` keyword argument from :func:`numpy.array`. The default is `"K"`.
            ndmin: The `ndmin` keyword argument from :func:`numpy.array`. The default is 0.

        Examples:
            Create a :obj:`~galois.FieldArray` subclass for $\mathrm{GF}(3^5)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**5)
                print(GF.properties)
                alpha = GF.primitive_element; alpha

            Create a finite field scalar from its integer representation, polynomial representation,
            or a power of the primitive element.

            .. ipython-with-reprs:: int,poly,power

                GF(17)
                GF("x^2 + 2x + 2")
                alpha ** 222

            Create a finite field array from its integer representation, polynomial representation,
            or powers of the primitive element.

            .. ipython-with-reprs:: int,poly,power

                GF([17, 4, 148, 205])
                GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])
                alpha ** np.array([[222, 69], [54, 24]])
        """
        # Adding __init__ and not doing anything is done to overwrite the superclass's __init__ docstring
        return

    ###############################################################################
    # Verification routines
    ###############################################################################

    @classmethod
    def _verify_array_like_types_and_values(cls, x: ElementLike | ArrayLike) -> ElementLike | ArrayLike:
        if isinstance(x, (int, np.integer)):
            cls._verify_scalar_value(x)
        elif isinstance(x, cls):
            # This was a previously created and vetted array -- there's no need to re-verify
            if x.ndim == 0:
                # Ensure that in "large" fields with dtype=object that FieldArray objects aren't assigned to the array.
                # The arithmetic functions are designed to operate on Python ints.
                x = int(x)
        elif isinstance(x, str):
            x = cls._convert_to_element(x)
            cls._verify_scalar_value(x)
        elif isinstance(x, (list, tuple)):
            x = cls._convert_iterable_to_elements(x)
            cls._verify_array_values(x)
        elif isinstance(x, np.ndarray):
            # If this a NumPy array, but not a FieldArray, verify the array
            if x.dtype == np.object_:
                x = cls._verify_element_types_and_convert(x, object_=True)
            elif not np.issubdtype(x.dtype, np.integer):
                raise TypeError(f"{cls.name} arrays must have integer dtypes, not {x.dtype}.")
            cls._verify_array_values(x)
        else:
            raise TypeError(
                f"{cls.name} arrays can be created with scalars of type int/str, lists/tuples, "
                f"or ndarrays, not {type(x)}."
            )

        return x

    @classmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        if array.size == 0:
            return array
        if object_:
            return np.vectorize(cls._convert_to_element, otypes=[object])(array)
        return np.vectorize(cls._convert_to_element)(array)

    @classmethod
    def _verify_scalar_value(cls, scalar: int):
        if not 0 <= scalar < cls.order:
            raise ValueError(f"{cls.name} scalars must be in `0 <= x < {cls.order}`, not {scalar}.")

    @classmethod
    def _verify_array_values(cls, array: np.ndarray):
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in `0 <= x < {cls.order}`, not {values}.")

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    def _convert_to_element(cls, element: ElementLike) -> int:
        if isinstance(element, (int, np.integer)):
            element = int(element)
        elif isinstance(element, str):
            element = str_to_integer(element, cls.prime_subfield)
        elif isinstance(element, FieldArray):
            element = int(element)
        else:
            raise TypeError(f"Valid element-like values are integers and string, not {type(element)}.")

        return element

    @classmethod
    def _convert_iterable_to_elements(cls, iterable: IterableLike) -> np.ndarray:
        if cls.dtypes == [np.object_]:
            array = np.array(iterable, dtype=object)
            array = cls._verify_element_types_and_convert(array, object_=True)
        else:
            # Try to convert to an integer array in one shot
            array = np.array(iterable)

            if not np.issubdtype(array.dtype, np.integer):
                # There are strings in the array, try again and manually convert each element
                array = np.array(iterable, dtype=object)
                array = cls._verify_element_types_and_convert(array)

        return array

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    @extend_docstring(
        Array.Zeros,
        DOCSTRING_MAP,
        """
        Examples:
            .. ipython:: python

                GF = galois.GF(31)
                GF.Zeros((2, 5))
        """,
    )
    def Zeros(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        return super().Zeros(shape, dtype=dtype)

    @classmethod
    @extend_docstring(
        Array.Ones,
        DOCSTRING_MAP,
        """
        Examples:
            .. ipython:: python

                GF = galois.GF(31)
                GF.Ones((2, 5))
        """,
    )
    def Ones(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        return super().Ones(shape, dtype=dtype)

    @classmethod
    @extend_docstring(
        Array.Range,
        DOCSTRING_MAP,
        """
        Examples:
            For prime fields, the increment is simply a finite field element, since all elements are integers.

            .. ipython:: python

                GF = galois.GF(31)
                GF.Range(10, 20)
                GF.Range(10, 20, 2)

            For extension fields, the increment is the integer increment between finite field elements in their
            :ref:`integer representation <int-repr>`.

            .. ipython-with-reprs:: int,poly

                GF = galois.GF(3**3)
                GF.Range(10, 20)
                GF.Range(10, 20, 2)
        """,
    )
    def Range(
        cls,
        start: ElementLike,
        stop: ElementLike,
        step: int = 1,
        dtype: DTypeLike | None = None,
    ) -> Self:
        return super().Range(start, stop, step=step, dtype=dtype)

    @classmethod
    @extend_docstring(
        Array.Random,
        DOCSTRING_MAP,
        """
        Examples:
            Generate a random matrix with an unpredictable seed.

            .. ipython:: python

                GF = galois.GF(31)
                GF.Random((2, 5))

            Generate a random array with a specified seed. This produces repeatable outputs.

            .. ipython:: python

                GF.Random(10, seed=123456789)
                GF.Random(10, seed=123456789)

            Generate a group of random arrays using a single global seed.

            .. ipython:: python

                rng = np.random.default_rng(123456789)
                GF.Random(10, seed=rng)
                GF.Random(10, seed=rng)
        """,
    )
    def Random(
        cls,
        shape: ShapeLike = (),
        low: ElementLike = 0,
        high: ElementLike | None = None,
        seed: int | np.random.Generator | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        return super().Random(shape=shape, low=low, high=high, seed=seed, dtype=dtype)

    @classmethod
    @extend_docstring(
        Array.Identity,
        DOCSTRING_MAP,
        """
        Examples:
            .. ipython:: python

                GF = galois.GF(31)
                GF.Identity(4)
        """,
    )
    def Identity(cls, size: int, dtype: DTypeLike | None = None) -> Self:
        return super().Identity(size, dtype=dtype)

    @classmethod
    def Vandermonde(cls, element: ElementLike, rows: int, cols: int, dtype: DTypeLike | None = None) -> Self:
        r"""
        Creates an $m \times n$ Vandermonde matrix of $a \in \mathrm{GF}(q)$.

        Arguments:
            element: An element $a$ of $\mathrm{GF}(q)$.
            rows: The number of rows $m$ in the Vandermonde matrix.
            cols: The number of columns $n$ in the Vandermonde matrix.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.FieldArray` subclass (the first element in
                :obj:`~galois.FieldArray.dtypes`).

        Returns:
            A $m \times n$ Vandermonde matrix.

        Examples:
            .. ipython-with-reprs:: int,poly,power

                @suppress
                np.set_printoptions(linewidth=200)
                GF = galois.GF(2**3)
                a = GF.primitive_element; a
                V = GF.Vandermonde(a, 7, 7); V
                @suppress
                np.set_printoptions(linewidth=75)
        """
        pass

    ###############################################################################
    # Conversions
    ###############################################################################

    @classmethod
    def Vector(cls, array: ArrayLike, dtype: DTypeLike | None = None) -> FieldArray:
        r"""
        Converts length-$m$ vectors over the prime subfield $\mathrm{GF}(p)$ to an array
        over $\mathrm{GF}(p^m)$.

        Arguments:
            array: An array over $\mathrm{GF}(p)$ with last dimension $m$. An array with shape
                `(n1, n2, m)` has output shape `(n1, n2)`. By convention, the vectors are ordered from degree
                $m-1$ to degree 0.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.FieldArray` subclass (the first element in
                :obj:`~galois.FieldArray.dtypes`).

        Returns:
            An array over $\mathrm{GF}(p^m)$.

        Notes:
            This method is the inverse of the :func:`vector` method.

        Examples:
            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**3)
                a = GF.Vector([[1, 0, 2], [0, 2, 1]]); a
                a.vector()

        Group:
            Conversions

        Order:
            21
        """
        pass

    def vector(self, dtype: DTypeLike | None = None) -> FieldArray:
        r"""
        Converts an array over $\mathrm{GF}(p^m)$ to length-$m$ vectors over the prime subfield
        $\mathrm{GF}(p)$.

        Arguments:
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.FieldArray` subclass (the first element in
                :obj:`~galois.FieldArray.dtypes`).

        Returns:
            An array over $\mathrm{GF}(p)$ with last dimension $m$.

        Notes:
            This method is the inverse of the :func:`Vector` constructor. For an array with shape `(n1, n2)`,
            the output shape is `(n1, n2, m)`. By convention, the vectors are ordered from degree $m-1$
            to degree 0.

        Examples:
            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**3)
                a = GF([11, 7]); a
                vec = a.vector(); vec
                GF.Vector(vec)

        Group:
            Conversions

        Order:
            21
        """
        pass

    # @classmethod
    # def polynomial_basis(
    #     cls,
    #     order: Literal["desc", "asc"] = "desc",
    # ) -> FieldArray:
    #     r"""
    #     Returns the polynomial (power) basis of the extension field $\mathrm{GF}(p^m)$.

    #     Let $f(x)$ be the irreducible polynomial of degree $m$ that defines the field

    #     $$
    #     \mathrm{GF}(p^m) \cong \mathrm{GF}(p)[x] / (f(x)).
    #     $$

    #     If $\alpha$ denotes the residue class of $x$ modulo $f(x)$, then the **polynomial basis**
    #     (also called the **power basis**) is

    #     $$
    #     \{1, \alpha, \alpha^2, \dots, \alpha^{m-1}\}.
    #     $$

    #     This method returns these $m$ basis elements as :class:`~galois.FieldArray` objects,
    #     computed as successive powers of $\alpha$ inside the field.

    #     Arguments:
    #         order:
    #             Basis ordering:

    #             - `"desc"` (default): Returns $(\alpha^{m-1}, \dots, \alpha, 1)$.
    #             - `"asc"`: Returns $(1, \alpha, \alpha^2, \dots, \alpha^{m-1})$.

    #     Returns:
    #         A 1-D :class:`~galois.FieldArray` of length `m` containing the polynomial basis elements.

    #     Notes:
    #         The element $\alpha$ is the canonical image of the polynomial indeterminate $x$
    #         in the quotient field. All elements of $\mathrm{GF}(p^m)$ can be expressed uniquely
    #         as a linear combination of this basis with coefficients in $\mathrm{GF}(p)$.

    #     Examples:
    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(2**5)
    #             GF.polynomial_basis()
    #             GF.polynomial_basis(order="asc")

    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(3**4)
    #             GF.polynomial_basis()
    #             GF.polynomial_basis(order="asc")

    #     Group:
    #         Conversions
    #     """
    #     verify_literal(order, ("desc", "asc"))

    #     field = cls
    #     p = field.characteristic
    #     m = field.degree

    #     # Build ascending degrees
    #     basis = field.Zeros(m)
    #     x = field(p)  # The polynomial x mod f(x)
    #     basis[0] = field(1)  # Start with x^0 mod f(x)
    #     for i in range(1, m):
    #         basis[i] = basis[i - 1] * x

    #     if order == "desc":
    #         basis = basis[::-1]

    #     return basis

    # @classmethod
    # def normal_basis(
    #     cls, normal_element: ElementLike | None = None, order: Literal["desc", "asc"] = "desc"
    # ) -> FieldArray:
    #     r"""
    #     Returns a normal basis of the extension field $\mathrm{GF}(p^m)$.

    #     A **normal basis** is generated by a normal element $\beta \in \mathrm{GF}(p^m)$, that is,
    #     an element whose Frobenius conjugates

    #     $$
    #     \{\beta, \beta^{p}, \beta^{p^{2}}, \dots, \beta^{p^{m-1}}\}
    #     $$

    #     form a basis of $\mathrm{GF}(p^m)$ as a vector space over the base field $\mathrm{GF}(p)$.

    #     This method returns these $m$ conjugates as :class:`~galois.FieldArray` elements.

    #     Arguments:
    #         normal_element:
    #             A normal element of the field to generate the basis. If `None`, the field's canonical
    #             normal element :attr:`~galois.FieldArray.normal_element` is used.
    #         order:
    #             Basis ordering:

    #             - `"desc"`: Returns $(\beta^{p^{m-1}}, \dots, \beta^{p}, \beta)$.
    #             - `"asc"`: Returns $(\beta, \beta^{p}, \dots, \beta^{p^{m-1}})$.

    #     Returns:
    #         A 1-D :class:`~galois.FieldArray` of length `m` containing the normal basis elements.

    #     Notes:
    #         A normal basis is often advantageous for hardware or bit-level arithmetic because
    #         the Frobenius map $a \mapsto a^p$ corresponds to a coordinate rotation.

    #     Examples:
    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(2**5)
    #             GF.normal_basis()
    #             GF.normal_basis(order="asc")

    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(3**4)
    #             GF.normal_basis()
    #             GF.normal_basis(order="asc")

    #     Group:
    #         Conversions
    #     """
    #     verify_literal(order, ("desc", "asc"))

    #     field = cls
    #     p = field.characteristic
    #     m = field.degree
    #     if normal_element is None:
    #         normal_element = field.normal_element
    #     else:
    #         normal_element = field(normal_element)
    #         verify_isinstance(normal_element, field)
    #         if not normal_element.ndim == 0:
    #             raise ValueError(f"Argument 'normal_element' must be a scalar, not {normal_element}.")
    #         if not is_normal_element(normal_element, field.irreducible_poly):
    #             raise ValueError(f"Argument normal_element must be normal in {field.name}, {normal_element} is not.")

    #     # Build ascending degrees
    #     basis = field.Zeros(m)
    #     basis[0] = normal_element  # Start with β
    #     for i in range(1, m):
    #         basis[i] = basis[i - 1] ** p

    #     if order == "desc":
    #         basis = basis[::-1]

    #     return basis

    # @classmethod
    # def change_of_basis_matrix(
    #     cls,
    #     from_basis: FieldArray,
    #     to_basis: FieldArray,
    # ) -> FieldArray:
    #     r"""
    #     Computes the linear transformation matrix that converts coordinates from one basis of
    #     $\mathrm{GF}(p^m)$ (viewed as a vector space over $\mathrm{GF}(p)$) to another.

    #     Let $\mathcal{B} = (b_0, \dots, b_{m-1})$ be the `from_basis` and
    #     $\mathcal{C} = (c_0, \dots, c_{m-1})$ be the `to_basis`.
    #     This method returns the unique matrix $T_{\mathcal{B}\to\mathcal{C}$ over $\mathrm{GF}(p)$ satisfying

    #     .. math::

    #         [x]_\mathcal{C} = T_{\mathcal{B}\to\mathcal{C}\,[x]_\mathcal{B}, \qquad \text{for all } x \in \mathrm{GF}(p^m),

    #     where $[x]_\mathcal{B}$ and $[x]_\mathcal{C}$ denote the coordinate vectors of $x$ with respect to
    #     $\mathcal{B}$ and $\mathcal{C}$, respectively.

    #     Arguments:
    #         from_basis:
    #             A length-`m` :class:`~galois.FieldArray` containing a basis
    #             $(b_0, \dots, b_{m-1})$ of $\mathrm{GF}(p^m)$ over $\mathrm{GF}(p)$.
    #         to_basis:
    #             A length-`m` :class:`~galois.FieldArray` containing a basis
    #             $(c_0, \dots, c_{m-1})$ of the same field.

    #     Returns:
    #         A square `m x m` matrix over `GF(p)` that converts coordinates in
    #         `from_basis` to coordinates in `to_basis`.

    #     Notes:
    #         Internally, each basis vector is expanded into its coordinate vector in the
    #         polynomial/power basis via :meth:`FieldArray.vector`.
    #         The change-of-basis matrix is then computed by solving $C\,T_{\mathcal{B}\to\mathcal{C} = B$
    #         over $\mathrm{GF}(p)$.

    #     Examples:
    #         .. ipython:: python

    #             GF = galois.GF(2**5)
    #             B = GF.polynomial_basis()
    #             C = GF.normal_basis()
    #             GF.change_of_basis_matrix(B, C)

    #         .. ipython:: python

    #             GF = galois.GF(3**4)
    #             B = GF.polynomial_basis()
    #             C = GF.normal_basis()
    #             GF.change_of_basis_matrix(B, C)

    #     Group:
    #         Conversions
    #     """
    #     verify_isinstance(from_basis, FieldArray)
    #     verify_isinstance(to_basis, FieldArray)
    #     if not from_basis.size == cls.degree:
    #         raise ValueError(
    #             f"Argument 'from_basis' must be a basis of size {cls.degree} for {cls.name}, not {from_basis.size}."
    #         )
    #     if not to_basis.size == cls.degree:
    #         raise ValueError(
    #             f"Argument 'to_basis' must be a basis of size {cls.degree} for {cls.name}, not {to_basis.size}."
    #         )

    #     B = from_basis.vector()  # Each row is a basis polynomial over GF(p)
    #     C = to_basis.vector()  # Each row is a basis polynomial over GF(p)

    #     T_B_to_C = np.linalg.solve(C, B)

    #     return T_B_to_C

    # def change_bases(
    #     self,
    #     from_basis: FieldArray,
    #     to_basis: FieldArray,
    # ) -> FieldArray:
    #     r"""
    #     Expresses the field elements in a new basis of $\mathrm{GF}(p^m)$.

    #     Let $\mathcal{B} = (b_0,\dots,b_{m-1})$ be the `from_basis` and
    #     $\mathcal{C} = (c_0,\dots,c_{m-1})$ be the `to_basis`.
    #     If $T_{\mathcal{B}\to\mathcal{C}}$ denotes the change-of-basis matrix satisfying

    #     .. math::

    #         [x]_{\mathcal{C}} = T_{\mathcal{B}\to\mathcal{C}} \,[x]_{\mathcal{B}}, \qquad
    #         x \in \mathrm{GF}(p^m),

    #     then this method applies that transformation to each element of `self`.

    #     Arguments:
    #         from_basis:
    #             A length-`m` basis $\mathcal{B}$ of the field over $\mathrm{GF}(p)$.
    #         to_basis:
    #             A length-`m` basis $\mathcal{C}$ of the field over $\mathrm{GF}(p)$.

    #     Returns:
    #         A :class:`~galois.FieldArray` with the same shape as `self`, but whose coordinates
    #         are expressed in the `to_basis` representation.

    #     Notes:
    #         Internally, this computes the change-of-basis matrix $T_{\mathcal{B}\to\mathcal{C}}$ using
    #         :meth:`~galois.FieldArray.change_of_basis_matrix`, converts `self` into coordinate vectors in
    #         $\mathcal{B}$, and multiplies them by $T_{\mathcal{B}\to\mathcal{C}}$.

    #     Examples:
    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(2**5)
    #             x = GF.Random(10); x
    #             B = GF.polynomial_basis()
    #             C = GF.normal_basis()
    #             y = x.change_bases(B, C); y
    #             assert np.array_equal(x, y.change_bases(C, B))

    #         .. ipython-with-reprs:: int,poly,power

    #             GF = galois.GF(3**4)
    #             x = GF.Random(10); x
    #             B = GF.polynomial_basis()
    #             C = GF.normal_basis()
    #             y = x.change_bases(B, C); y
    #             assert np.array_equal(x, y.change_bases(C, B))

    #     Group:
    #         Conversions
    #     """
    #     field = type(self)
    #     T_B_to_C = field.change_of_basis_matrix(from_basis, to_basis)

    #     V_B = self.vector()  # Each row is a vector over GF(p)
    #     shape = V_B.shape
    #     V_B = V_B.reshape(-1, shape[-1])  # Reshape to 2-D array
    #     V_C = V_B @ T_B_to_C
    #     V_C = V_C.reshape(shape)  # Reshape back to original shape with last dimension m

    #     return field.Vector(V_C)

    ###############################################################################
    # Class methods
    ###############################################################################

    @classmethod
    def compile(cls, mode: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]):
        """
        Recompile the just-in-time compiled ufuncs for a new calculation mode.

        This function updates :obj:`ufunc_mode`.

        Arguments:
            mode: The ufunc calculation mode.

                - `"auto"`: Selects `"jit-lookup"` for fields with order less than $2^{20}$, `"jit-calculate"`
                  for larger fields, and `"python-calculate"` for fields whose elements cannot be represented with
                  :obj:`numpy.int64`.
                - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for
                  efficient computation. In the few cases where explicit calculation is faster than table lookup,
                  explicit calculation is used.
                - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"`
                  mode is designed for large fields that cannot or should not store lookup tables in RAM. Generally,
                  the `"jit-calculate"` mode is slower than `"jit-lookup"`.
                - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is intended for fields
                  whose elements cannot be represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_`
                  with Python :obj:`int` (which has arbitrary precision). However, this mode can be used for any
                  field, enabling the code to run without Numba JIT compilation.

        Group:
            Arithmetic compilation

        Order:
            33
        """
        return super().compile(mode)

    @classmethod
    def repr(cls, element_repr: Literal["int", "poly", "power"] = "int") -> Generator[None, None, None]:
        r"""
        Sets the element representation for all arrays from this :obj:`~galois.FieldArray` subclass.

        Arguments:
            element_repr: The field element representation to be set.

                - `"int"` (default): The :ref:`integer representation <int-repr>`.
                - `"poly"`: The :ref:`polynomial representation <poly-repr>`.
                - `"power"`: The :ref:`power representation <power-repr>`.

                .. slow-performance::

                    To display elements in the power representation, :obj:`galois` must compute the discrete logarithm
                    of each element displayed. For large fields or fields using
                    :ref:`explicit calculation <explicit-calculation>`, this process can take a while. However, when
                    using :ref:`lookup tables <lookup-tables>` this representation is just as fast as the others.

        Returns:
            A context manager for use in a `with` statement. If permanently setting the element representation,
            disregard the return value.

        Notes:
            This function updates :obj:`~galois.FieldArray.element_repr`.

        Examples:
            The default element representation is the integer representation.

            .. ipython:: python

                GF = galois.GF(3**2)
                x = GF.elements; x

            Permanently set the element representation by calling :func:`repr`.

            .. md-tab-set::

                .. md-tab-item:: Polynomial

                    .. ipython:: python

                        GF.repr("poly");
                        x

                .. md-tab-item:: Power

                    .. ipython:: python

                        GF.repr("power");
                        x
                        @suppress
                        GF.repr()

            Temporarily modify the element representation by using :func:`repr` as a context manager.

            .. md-tab-set::

                .. md-tab-item:: Polynomial

                    .. ipython:: python

                        print(x)
                        with GF.repr("poly"):
                            print(x)
                        # Outside the context manager, the element representation reverts to its previous value
                        print(x)

                .. md-tab-item:: Power

                    .. ipython:: python

                        print(x)
                        with GF.repr("power"):
                            print(x)
                        # Outside the context manager, the element representation reverts to its previous value
                        print(x)
                        @suppress
                        GF.repr()

        Group:
            Element representation

        Order:
            31
        """
        pass

    @classmethod
    def repr_table(
        cls,
        element: ElementLike | None = None,
        sort: Literal["power", "poly", "vector", "int"] = "power",
    ) -> str:
        r"""
        Generates a finite field element representation table comparing the power, polynomial, vector, and
        integer representations.

        Arguments:
            element: An element to use as the exponent base in the power representation. The default is `None` which
                corresponds to :obj:`~galois.FieldArray.primitive_element`.
            sort: The sorting method for the table. The default is `"power"`. Sorting by `"power"` will order the rows
                of the table by ascending powers of `element`. Sorting by any of the others will order the rows in
                lexicographical polynomial/vector order, which is equivalent to ascending order of the integer
                representation.

        Returns:
            A string representation of the table comparing the power, polynomial, vector, and integer representations
            of each field element.

        Examples:
            Create a :obj:`~galois.FieldArray` subclass for $\mathrm{GF}(3^3)$.

            .. ipython:: python

                GF = galois.GF(3**3)
                print(GF.properties)

            Generate a representation table for $\mathrm{GF}(3^3)$. Since $x^3 + 2x + 1$ is a primitive
            polynomial, $x$ is a primitive element of the field. Notice, $\textrm{ord}(x) = 26$.

            .. ipython:: python

                print(GF.repr_table())
                GF("x").multiplicative_order()

            Generate a representation table for $\mathrm{GF}(3^3)$ using a different primitive element
            $2x^2 + 2x + 2$. Notice, $\textrm{ord}(2x^2 + 2x + 2) = 26$.

            .. ipython:: python

                print(GF.repr_table("2x^2 + 2x + 2"))
                GF("2x^2 + 2x + 2").multiplicative_order()

            Generate a representation table for $\mathrm{GF}(3^3)$ using a non-primitive element $x^2$.
            Notice, $\textrm{ord}(x^2) = 13 \ne 26$.

            .. ipython:: python

                print(GF.repr_table("x^2"))
                GF("x^2").multiplicative_order()

        Group:
            String representation

        Order:
            30
        """
        pass

    @classmethod
    def arithmetic_table(
        cls,
        operation: Literal["+", "-", "*", "/"],
        x: FieldArray | None = None,
        y: FieldArray | None = None,
    ) -> str:
        r"""
        Generates the specified arithmetic table for the finite field.

        Arguments:
            operation: The arithmetic operation.
            x: Optionally specify the $x$ values for the arithmetic table. The default is `None`
                which represents $\{0, \dots, p^m - 1\}$.
            y: Optionally specify the $y$ values for the arithmetic table. The default is `None`
                which represents $\{0, \dots, p^m - 1\}$ for addition, subtraction, and multiplication and
                $\{1, \dots, p^m - 1\}$ for division.

        Returns:
            A string representation of the arithmetic table.

        Examples:
            Arithmetic tables can be displayed using any element representation.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                print(GF.arithmetic_table("+"))

            An arithmetic table may also be constructed from arbitrary $x$ and $y$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF([7, 2, 8]); x
                y = GF([1, 4, 5, 3]); y
                print(GF.arithmetic_table("+", x=x, y=y))

        Group:
            String representation

        Order:
            30
        """
        pass

    @classmethod
    def primitive_root_of_unity(cls, n: int) -> Self:
        r"""
        Finds a primitive $n$-th root of unity in the finite field.

        Arguments:
            n: The root of unity.

        Returns:
            The primitive $n$-th root of unity, a 0-D scalar array.

        Raises:
            ValueError: If no primitive $n$-th roots of unity exist. This happens when $n$ is not a
                divisor of $p^m - 1$.

        Notes:
            A primitive $n$-th root of unity $\omega_n$ is such that $\omega_n^n = 1$ and
            $\omega_n^k \ne 1$ for all $1 \le k \lt n$.

            In $\mathrm{GF}(p^m)$, a primitive $n$-th root of unity exists when $n$ divides
            $p^m - 1$. Then, the primitive root is $\omega_n = \alpha^{(p^m - 1)/n}$ where $\alpha$
            is a primitive element of the field.

        Examples:
            In $\mathrm{GF}(31)$, primitive roots exist for all divisors of 30.

            .. ipython:: python

                GF = galois.GF(31)
                GF.primitive_root_of_unity(2)
                GF.primitive_root_of_unity(5)
                GF.primitive_root_of_unity(15)

            However, they do not exist for $n$ that do not divide 30.

            .. ipython:: python
                :okexcept:

                GF.primitive_root_of_unity(7)

            For $\omega_5$, one can see that $\omega_5^5 = 1$ and $\omega_5^k \ne 1$ for
            $1 \le k \lt 5$.

            .. ipython:: python

                root = GF.primitive_root_of_unity(5); root
                powers = np.arange(1, 5 + 1); powers
                root ** powers

        Group:
            Elements

        Order:
            22
        """
        pass

    @classmethod
    def primitive_roots_of_unity(cls, n: int) -> Self:
        r"""
        Finds all primitive $n$-th roots of unity in the finite field.

        Arguments:
            n: The root of unity.

        Returns:
            All primitive $n$-th roots of unity, a 1-D array. The roots are sorted in lexicographical order.

        Raises:
            ValueError: If no primitive $n$-th roots of unity exist. This happens when $n$ is not a
                divisor of $p^m - 1$.

        Notes:
            A primitive $n$-th root of unity $\omega_n$ is such that $\omega_n^n = 1$ and
            $\omega_n^k \ne 1$ for all $1 \le k \lt n$.

            In $\mathrm{GF}(p^m)$, a primitive $n$-th root of unity exists when $n$ divides
            $p^m - 1$. Then, the primitive root is $\omega_n = \alpha^{(p^m - 1)/n}$ where $\alpha$
            is a primitive element of the field.

        Examples:
            In $\mathrm{GF}(31)$, primitive roots exist for all divisors of 30.

            .. ipython:: python

                GF = galois.GF(31)
                GF.primitive_roots_of_unity(2)
                GF.primitive_roots_of_unity(5)
                GF.primitive_roots_of_unity(15)

            However, they do not exist for $n$ that do not divide 30.

            .. ipython:: python
                :okexcept:

                GF.primitive_roots_of_unity(7)

            For $\omega_5$, one can see that $\omega_5^5 = 1$ and $\omega_5^k \ne 1$ for
            $1 \le k \lt 5$.

            .. ipython:: python

                root = GF.primitive_roots_of_unity(5); root
                powers = np.arange(1, 5 + 1); powers
                np.power.outer(root, powers)

        Group:
            Elements

        Order:
            22
        """
        pass

    ###############################################################################
    # Instance methods
    ###############################################################################

    def additive_order(self) -> int | np.ndarray:
        r"""
        Computes the additive order of each element in $x$.

        Returns:
            An integer array of the additive order of each element in $x$. The return value is a single integer
            if the input array $x$ is a scalar.

        Notes:
            The additive order $a$ of $x$ in $\mathrm{GF}(p^m)$ is the smallest integer $a$
            such that $x a = 0$. With the exception of 0, the additive order of every element is
            the finite field's characteristic.

        Examples:
            Compute the additive order of each element of $\mathrm{GF}(3^2)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF.elements; x
                order = x.additive_order(); order
                x * order
        """
        pass

    def multiplicative_order(self) -> int | np.ndarray:
        r"""
        Computes the multiplicative order $\textrm{ord}(x)$ of each element in $x$.

        Returns:
            An integer array of the multiplicative order of each element in $x$. The return value is a single
            integer if the input array $x$ is a scalar.

        Raises:
            ArithmeticError: If zero is provided as an input. The multiplicative order of 0 is not defined. There is
                no power of 0 that ever results in 1.

        Notes:
            The multiplicative order $\textrm{ord}(x) = a$ of $x$ in $\mathrm{GF}(p^m)$ is the
            smallest power $a$ such that $x^a = 1$. If $a = p^m - 1$, $a$ is said to be a
            generator of the multiplicative group $\mathrm{GF}(p^m)^\times$.

            Note, :func:`multiplicative_order` should not be confused with :obj:`order`. The former returns the
            multiplicative order of :obj:`~galois.FieldArray` elements. The latter is a property of the field, namely
            the finite field's order or size.

        Examples:
            Compute the multiplicative order of each non-zero element of $\mathrm{GF}(3^2)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF.units; x
                order = x.multiplicative_order(); order
                x ** order

            The elements with $\textrm{ord}(x) = 8$ are multiplicative generators of
            $\mathrm{GF}(3^2)^\times$, which are also called primitive elements.

            .. ipython-with-reprs:: int,poly,power

                GF.primitive_elements
        """
        pass

    def is_square(self) -> bool | np.ndarray:
        r"""
        Determines if the elements of $x$ are squares in the finite field.

        Returns:
            A boolean array indicating if each element in $x$ is a square. The return value is a single boolean
            if the input array $x$ is a scalar.

        See Also:
            squares, non_squares

        Notes:
            An element $x$ in $\mathrm{GF}(p^m)$ is a *square* if there exists a $y$ such that
            $y^2 = x$ in the field.

            In fields with characteristic 2, every element is a square (with two identical square roots). In fields
            with characteristic greater than 2, exactly half of the nonzero elements are squares (with two unique
            square roots).

        References:
            - Section 3.5.1 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.

        Examples:
            Since $\mathrm{GF}(2^3)$ has characteristic 2, every element has a square root.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(2**3)
                x = GF.elements; x
                x.is_square()
                @suppress
                GF.repr()

            In $\mathrm{GF}(11)$, the characteristic is greater than 2 so only half of the elements have square
            roots.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(11)
                x = GF.elements; x
                x.is_square()
        """
        pass

    def row_reduce(self, ncols: int | None = None, eye: Literal["left", "right"] = "left") -> Self:
        r"""
        Performs Gaussian elimination on the matrix to achieve reduced row echelon form (RREF).

        Arguments:
            ncols: The number of columns to perform Gaussian elimination over. The default is `None` which represents
                the number of columns of the matrix.
            eye: The location of the identity matrix $\mathbf{I}$, either on the left or the right.

        Returns:
            The reduced row echelon form of the input matrix.

        Notes:
            The elementary row operations in Gaussian elimination are:

            1. Swap the position of any two rows.
            2. Multiply any row by a non-zero scalar.
            3. Add any row to a scalar multiple of another row.

        Examples:
            Perform Gaussian elimination to get the reduced row echelon form of $\mathbf{A}$.

            .. ipython:: python

                GF = galois.GF(31)
                A = GF([[16, 12, 1, 25], [1, 10, 27, 29], [1, 0, 3, 19]]); A
                A.row_reduce()
                np.linalg.matrix_rank(A)

            Perform Gaussian elimination to get an $\mathbf{I}$ on the right side of $\mathbf{A}$.

            .. ipython:: python

                A.row_reduce(eye="right")

            Or only perform Gaussian elimination over 2 columns.

            .. ipython:: python

                A.row_reduce(ncols=2)

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def lu_decompose(self) -> tuple[Self, Self]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns:
            - The lower triangular matrix.
            - The upper triangular matrix.

        Notes:
            The LU decomposition of $\mathbf{A}$ is defined as $\mathbf{A} = \mathbf{L} \mathbf{U}$.

        Examples:
            .. ipython:: python

                GF = galois.GF(31)
                # Not every square matrix has an LU decomposition
                A = GF([[22, 11, 25, 11], [30, 27, 10, 3], [21, 16, 29, 7]]); A
                L, U = A.lu_decompose()
                L
                U
                assert np.array_equal(A, L @ U)

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def plu_decompose(self) -> tuple[Self, Self, Self]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns:
            - The column permutation matrix.
            - The lower triangular matrix.
            - The upper triangular matrix.

        Notes:
            The PLU decomposition of $\mathbf{A}$ is defined as
            $\mathbf{A} = \mathbf{P} \mathbf{L} \mathbf{U}$. This is equivalent to
            $\mathbf{P}^T \mathbf{A} = \mathbf{L} \mathbf{U}$.

        Examples:
            .. ipython:: python

                GF = galois.GF(31)
                A = GF([[0, 29, 2, 9], [20, 24, 5, 1], [2, 24, 1, 7]]); A
                P, L, U = A.plu_decompose()
                P
                L
                U
                assert np.array_equal(A, P @ L @ U)
                assert np.array_equal(P.T @ A, L @ U)

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def row_space(self) -> Self:
        r"""
        Computes the row space of the matrix $\mathbf{A}$.

        Returns:
            The row space basis matrix. The rows of the basis matrix are the basis vectors that span the row space.
            The number of rows of the basis matrix is the dimension of the row space.

        Notes:
            Given an $m \times n$ matrix $\mathbf{A}$ over $\mathrm{GF}(q)$, the *row space* of
            $\mathbf{A}$ is the vector space $\{\mathbf{x} \in \mathrm{GF}(q)^n\}$ defined by all linear
            combinations of the rows of $\mathbf{A}$. The row space has at most dimension
            $\textrm{min}(m, n)$.

            The row space has properties $\mathcal{R}(\mathbf{A}) = \mathcal{C}(\mathbf{A}^T)$ and
            $\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m$.

        Examples:
            The :func:`row_space` method defines basis vectors (its rows) that span the row space of
            $\mathbf{A}$.

            .. ipython:: python

                m, n = 5, 3
                GF = galois.GF(31)
                A = GF.Random((m, n)); A
                R = A.row_space(); R

            The dimension of the row space and left null space sum to $m$.

            .. ipython:: python

                LN = A.left_null_space(); LN
                assert R.shape[0] + LN.shape[0] == m

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def column_space(self) -> Self:
        r"""
        Computes the column space of the matrix $\mathbf{A}$.

        Returns:
            The column space basis matrix. The rows of the basis matrix are the basis vectors that span the column
            space. The number of rows of the basis matrix is the dimension of the column space.

        Notes:
            Given an $m \times n$ matrix $\mathbf{A}$ over $\mathrm{GF}(q)$, the *column space* of
            $\mathbf{A}$ is the vector space $\{\mathbf{x} \in \mathrm{GF}(q)^m\}$ defined by all linear
            combinations of the columns of $\mathbf{A}$. The column space has at most dimension
            $\textrm{min}(m, n)$.

            The column space has properties $\mathcal{C}(\mathbf{A}) = \mathcal{R}(\mathbf{A}^T)$  and
            $\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n$.

        Examples:
            The :func:`column_space` method defines basis vectors (its rows) that span the column space of
            $\mathbf{A}$.

            .. ipython:: python

                m, n = 3, 5
                GF = galois.GF(31)
                A = GF.Random((m, n)); A
                C = A.column_space(); C

            The dimension of the column space and null space sum to $n$.

            .. ipython:: python

                N = A.null_space(); N
                assert C.shape[0] + N.shape[0] == n

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def left_null_space(self) -> Self:
        r"""
        Computes the left null space of the matrix $\mathbf{A}$.

        Returns:
            The left null space basis matrix. The rows of the basis matrix are the basis vectors that span the
            left null space. The number of rows of the basis matrix is the dimension of the left null space.

        Notes:
            Given an $m \times n$ matrix $\mathbf{A}$ over $\mathrm{GF}(q)$, the *left null space*
            of $\mathbf{A}$ is the vector space $\{\mathbf{x} \in \mathrm{GF}(q)^m\}$ that annihilates the
            rows of $\mathbf{A}$, i.e. $\mathbf{x}\mathbf{A} = \mathbf{0}$.

            The left null space has properties $\mathcal{LN}(\mathbf{A}) = \mathcal{N}(\mathbf{A}^T)$ and
            $\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m$.

        Examples:
            The :func:`left_null_space` method defines basis vectors (its rows) that span the left null space of
            $\mathbf{A}$.

            .. ipython:: python

                m, n = 5, 3
                GF = galois.GF(31)
                A = GF.Random((m, n)); A
                LN = A.left_null_space(); LN

            The left null space is the set of vectors that sum the rows to 0.

            .. ipython:: python

                LN @ A

            The dimension of the row space and left null space sum to $m$.

            .. ipython:: python

                R = A.row_space(); R
                assert R.shape[0] + LN.shape[0] == m

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def null_space(self) -> Self:
        r"""
        Computes the null space of the matrix $\mathbf{A}$.

        Returns:
            The null space basis matrix. The rows of the basis matrix are the basis vectors that span the null space.
            The number of rows of the basis matrix is the dimension of the null space.

        Notes:
            Given an $m \times n$ matrix $\mathbf{A}$ over $\mathrm{GF}(q)$, the *null space* of
            $\mathbf{A}$ is the vector space $\{\mathbf{x} \in \mathrm{GF}(q)^n\}$ that annihilates the
            columns of $\mathbf{A}$, i.e. $\mathbf{A}\mathbf{x} = \mathbf{0}$.

            The null space has properties $\mathcal{N}(\mathbf{A}) = \mathcal{LN}(\mathbf{A}^T)$ and
            $\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n$.

        Examples:
            The :func:`null_space` method defines basis vectors (its rows) that span the null space of
            $\mathbf{A}$.

            .. ipython:: python

                m, n = 3, 5
                GF = galois.GF(31)
                A = GF.Random((m, n)); A
                N = A.null_space(); N

            The null space is the set of vectors that sum the columns to 0.

            .. ipython:: python

                A @ N.T

            The dimension of the column space and null space sum to $n$.

            .. ipython:: python

                C = A.column_space(); C
                assert C.shape[0] + N.shape[0] == n

        Group:
            Linear algebra

        Order:
            51
        """
        pass

    def field_trace(self) -> FieldArray:
        r"""
        Computes the field trace $\mathrm{Tr}_{L / K}(x)$ of the elements of $x$.

        Returns:
            The field trace of $x$ in the prime subfield $\mathrm{GF}(p)$.

        Notes:
            The `self` array $x$ is over the extension field $L = \mathrm{GF}(p^m)$. The field trace of
            $x$ is over the subfield $K = \mathrm{GF}(p)$. In other words,
            $\mathrm{Tr}_{L / K}(x) : L \rightarrow K$.

            For finite fields, since $L$ is a Galois extension of $K$, the field trace of $x$ is
            defined as a sum of the Galois conjugates of $x$.

            $$\mathrm{Tr}_{L / K}(x) = \sum_{i=0}^{m-1} x^{p^i}$$

        References:
            - https://en.wikipedia.org/wiki/Field_trace

        Examples:
            Compute the field trace of the elements of $\mathrm{GF}(3^2)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF.elements; x
                y = x.field_trace(); y
        """
        pass

    def field_norm(self) -> FieldArray:
        r"""
        Computes the field norm $\mathrm{N}_{L / K}(x)$ of the elements of $x$.

        Returns:
            The field norm of $x$ in the prime subfield $\mathrm{GF}(p)$.

        Notes:
            The `self` array $x$ is over the extension field $L = \mathrm{GF}(p^m)$. The field norm of
            $x$ is over the subfield $K = \mathrm{GF}(p)$. In other words,
            $\mathrm{N}_{L / K}(x) : L \rightarrow K$.

            For finite fields, since $L$ is a Galois extension of $K$, the field norm of $x$ is
            defined as a product of the Galois conjugates of $x$.

            $$\mathrm{N}_{L / K}(x) = \prod_{i=0}^{m-1} x^{p^i} = x^{(p^m - 1) / (p - 1)}$$

        References:
            - https://en.wikipedia.org/wiki/Field_norm

        Examples:
            Compute the field norm of the elements of $\mathrm{GF}(3^2)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF.elements; x
                y = x.field_norm(); y
        """
        pass

    def characteristic_poly(self) -> Poly:
        r"""
        Computes the characteristic polynomial of a finite field element $a$ or a square matrix
        $\mathbf{A}$.

        Returns:
            For scalar inputs, the degree-$m$ characteristic polynomial $c_a(x)$ of $a$ over
            $\mathrm{GF}(p)$. For square $n \times n$ matrix inputs, the degree-$n$ characteristic
            polynomial $c_A(x)$ of $\mathbf{A}$ over $\mathrm{GF}(p^m)$.

        Raises:
            ValueError: If the array is not a single finite field element (scalar 0-D array) or a
                square $n \times n$ matrix (2-D array).

        Notes:
            **For a scalar element $a \in \mathrm{GF}(p^m)$:**

            This method returns the degree-$m$ characteristic polynomial $c_a(x) \in \mathrm{GF}(p)[x]$ of the
            $\mathrm{GF}(p)$-linear operator $x \mapsto a x$ on $\mathrm{GF}(p^m)$. It has the form

            $$
            c_a(x) = \prod_{i=0}^{m-1} \left(x - a^{p^i}\right),
            $$

            where $p$ is the characteristic of the field and $a^{p^i}$ are the Frobenius conjugates
            of $a$. In particular, $c_a(x)$ has coefficients in the prime subfield $\mathrm{GF}(p)$
            and satisfies

            $$
            c_a(a) = 0 \quad \text{in } \mathrm{GF}(p^m).
            $$

            In a prime field $\mathrm{GF}(p)$ (i.e., when $m = 1$), the characteristic polynomial of
            $a$ is simply

            $$
            c_a(x) = x - a.
            $$

            .. info::
                :title: Relationship to the minimal polynomial (scalar case)

                For $a \in \mathrm{GF}(p^m)$, let $m_a(x)$ denote its standard minimal polynomial
                over $\mathrm{GF}(p)$.

                The characteristic polynomial $c_a(x)$ of the multiplication map satisfies

                $$
                c_a(x) = m_a(x)^{\,m / \deg m_a}.
                $$

                Therefore:

                - $m_a(x)$ divides $c_a(x)$.
                - The two polynomials coincide if and only if $a$ does not lie in a proper subfield
                  (i.e., its Frobenius orbit has length $m$).
                - If $a \in \mathrm{GF}(p^d)$ with $d \mid m$, then $\deg m_a = d$ and
                  $c_a(x) = m_a(x)^{m/d}$.

                Thus the minimal polynomial measures the Frobenius-orbit size of $a$, while the
                characteristic polynomial always has fixed degree $m$.

            **For a square matrix $\mathbf{A} \in \mathrm{GF}(p^m)^{n \times n}$:**

            The characteristic polynomial is

            $$
            c_A(x) = \det(x \mathbf{I} - \mathbf{A}),
            $$

            where $\mathbf{I}$ is the $n \times n$ identity matrix. The constant coefficient is

            $$
            c_A(0) = \det(-\mathbf{A}),
            $$

            and the $x^{n-1}$ coefficient is

            $$
            -\operatorname{Tr}(\mathbf{A}).
            $$

            The characteristic polynomial annihilates the matrix:

            $$
            c_A(\mathbf{A}) = \mathbf{0}.
            $$

            The characteristic polynomial always has degree $n$, and for matrices it plays the same
            role here as in standard linear algebra: it is the determinant of $x \mathbf{I} - \mathbf{A}$
            and contains information about eigenvalues, invariant factors, and the minimal polynomial.

        References:
            - https://en.wikipedia.org/wiki/Characteristic_polynomial

        Examples:
            The characteristic polynomial of the element $a$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**5)
                a = GF.Random(); a
                poly = a.characteristic_poly(); poly

                # The characteristic polynomial annihilates a
                poly(a, field=GF)

            The characteristic polynomial of the square matrix $\mathbf{A}$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**5)
                A = GF.Random((3,3)); A
                poly = A.characteristic_poly(); poly

                # The x^0 coefficient is det(-A)
                poly.coeffs[-1] == np.linalg.det(-A)

                # The x^(n-1) coefficient is -Tr(A)
                poly.coeffs[1] == -np.trace(A)

                # The characteristic polynomial annihilates the matrix A
                poly(A, elementwise=False)
        """
        pass

    def minimal_poly(self) -> Poly:
        r"""
        Computes the minimal polynomial of a finite field element $a$ or a square matrix
        $\mathbf{A}$.

        .. important::

            This function computes the **standard** minimal polynomial. The **linearized** minimal
            polynomial, which detects Frobenius-linear relations and is used to test for normal
            elements, is implemented in the :meth:`linearized_minimal_poly`.

        Returns:
            For scalar inputs, the minimal polynomial $m_a(x)$ of $a$ over $\mathrm{GF}(p)$.
            For square matrix inputs, the minimal polynomial $m_A(x)$ of $\mathbf{A}$ over
            $\mathrm{GF}(p^m)$.

        Raises:
            ValueError: If the array is not a single finite field element (scalar 0-D array) or a
                square $n \times n$ matrix (2-D array).

        Notes:
            **For a field element $a \in \mathrm{GF}(p^m)$:**

            The minimal polynomial $m_a(x) \in \mathrm{GF}(p)[x]$ is the *monic polynomial of least
            degree* whose evaluation at $a$ in $\mathrm{GF}(p^m)$ satisfies

            $$
            m_a(a) = 0.
            $$

            In a prime field $\mathrm{GF}(p)$, the minimal polynomial is simply

            $$
            m_a(x) = x - a.
            $$

            For extension fields $\mathrm{GF}(p^m)$, the minimal polynomial is constructed from the
            Frobenius conjugates of $a$:

            $$
            m_a(x) = \prod_{i=0}^{r-1} \left(x - a^{p^i}\right),
            $$

            where $\{a, a^p, a^{p^2}, \dots\}$ are the distinct conjugates of $a$ under the Frobenius
            map $x \mapsto x^p$. The degree $r$ equals the size of this conjugacy orbit. All
            coefficients of $m_a(x)$ lie in the prime subfield $\mathrm{GF}(p)$.

            .. info::
                :title: Relationship to the characteristic polynomial (scalar case)

                Let $a \in \mathrm{GF}(p^m)$ and let $m_a(x)$ denote its standard minimal polynomial
                over $\mathrm{GF}(p)$, and $c_a(x)$ the characteristic polynomial of the
                $\mathrm{GF}(p)$-linear operator $x \mapsto a x$ on $\mathrm{GF}(p^m)$.

                The two polynomials satisfy

                $$
                c_a(x) = m_a(x)^{\,m / \deg m_a}.
                $$

                Thus:

                - If $a$ does *not* lie in any proper subfield, then $\deg m_a = m$ and
                  $c_a(x) = m_a(x)$.
                - If $a$ lies in a proper subfield $\mathrm{GF}(p^d)$ with $d \mid m$, then
                  $\deg m_a = d$ and the characteristic polynomial has multiplicity $m/d$.

                In particular, $m_a(x)$ always divides $c_a(x)$ in $\mathrm{GF}(p)[x]$.

            **For a square matrix $\mathbf{A} \in \mathrm{GF}(p^m)^{n \times n}$:**

            The minimal polynomial $m_A(x) \in \mathrm{GF}(p^m)[x]$ is the *monic polynomial of least
            degree* such that

            $$
            m_A(\mathbf{A}) = \mathbf{0}.
            $$

            It always divides the characteristic polynomial

            $$
            c_A(x) = \det(x \mathbf{I} - \mathbf{A})
            $$

            and can be written in factored form as

            $$
            m_A(x) = \prod_i f_i(x)^{e_i},
            $$

            where $\{f_i(x)\}$ are the distinct monic irreducible factors of $c_A(x)$, and each
            exponent $e_i$ equals the size of the largest Jordan (or Frobenius) block of $\mathbf{A}$
            associated with $f_i$.

            Programmatically, this is determined by examining the sequence of null spaces

            $$
            \ker(f_i(\mathbf{A})),\;
            \ker(f_i(\mathbf{A})^2),\;
            \ker(f_i(\mathbf{A})^3),\;\dots
            $$

            and finding the smallest $k$ for which the nullity stabilizes. That $k$ is the exponent
            $e_i$ in the minimal polynomial.

        References:
            - https://en.wikipedia.org/wiki/Minimal_polynomial_(field_theory)
            - https://en.wikipedia.org/wiki/Minimal_polynomial_(linear_algebra)

        Examples:
            The minimal polynomial of the element $a$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**5)
                a = GF.Random(); a
                poly = a.minimal_poly(); poly

                # The minimal polynomial annihilates a
                poly(a, field=GF)

                # The minimal polynomial always divides the characteristic polynomial
                divmod(a.characteristic_poly(), poly)

            The minimal polynomial of a square matrix $\mathbf{A}$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                A = GF.Random((3, 3)); A
                poly = A.minimal_poly(); poly

                # The minimal polynomial annihilates the matrix
                poly(A, elementwise=False)

                # The minimal polynomial always divides the characteristic polynomial
                divmod(A.characteristic_poly(), poly)
        """
        pass

    def log(self, base: ElementLike | ArrayLike | None = None) -> int | np.ndarray:
        r"""
        Computes the discrete logarithm of the array $x$ base $\beta$.

        Arguments:
            base: A primitive element or elements $\beta$ of the finite field that is the base of the logarithm.
                The default is `None` which uses :obj:`~FieldArray.primitive_element`.

                .. slow-performance::

                    If the :obj:`FieldArray` is configured to use lookup tables (`ufunc_mode == "jit-lookup"`) and
                    this method is invoked with a base different from :obj:`~FieldArray.primitive_element`, then
                    explicit calculation will be used (which is slower than using lookup tables).

        Returns:
            An integer array $i$ of powers of $\beta$ such that $\beta^i = x$. The return array
            shape obeys NumPy broadcasting rules.

        Examples:
            Compute the logarithm of $x$ with default base $\alpha$, which is the specified primitive
            element of the field.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**5)
                alpha = GF.primitive_element; alpha
                x = GF.Random(10, low=1); x
                i = x.log(); i
                assert np.array_equal(alpha ** i, x)

            With the default argument, :func:`numpy.log` and :func:`~FieldArray.log` are equivalent.

            .. ipython:: python

                assert np.array_equal(np.log(x), x.log())

            Compute the logarithm of $x$ with a different base $\beta$, which is another primitive element
            of the field.

            .. ipython-with-reprs:: int,poly,power

                beta = GF.primitive_elements[-1]; beta
                i = x.log(beta); i
                assert np.array_equal(beta ** i, x)

            Compute the logarithm of a single finite field element base all of the primitive elements of the field.

            .. ipython-with-reprs:: int,poly,power

                x = GF.Random(low=1); x
                bases = GF.primitive_elements
                i = x.log(bases); i
                assert np.all(bases ** i == x)
        """
        pass

    ###############################################################################
    # Display methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        Displays the array specifying the class and finite field order.

        Notes:
            This function prepends `GF(` and appends `, order=p^m)`.

        Examples:
            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF([4, 2, 7, 5])
                x
        """
        return self._display("repr", separator=", ")

    def __str__(self) -> str:
        """
        Displays the array without specifying the class or finite field order.

        Notes:
            This function does not prepend `GF(` and or append `, order=p^m)`. It also omits the comma separators.

        Examples:
            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(3**2)
                x = GF([4, 2, 7, 5])
                print(x)
        """
        return self._display("str", separator=" ")

    def __format__(self, format_spec: str) -> str:
        """
        Formats the array using either the `__repr__` or `__str__` methods depending on the format specification.
        """
        if format_spec == "!r":
            return repr(self)
        else:
            return str(self)

    def _display(self, mode: Literal["repr", "str"], separator=", ") -> str:
        # View the array as an ndarray so that the scalar -> 0-D array conversion in __array_finalize__() for
        # Galois field arrays isn't continually invoked. This improves performance slightly.
        x = self.view(np.ndarray)
        field = type(self)

        prefix = "GF(" if mode == "repr" else ""
        order = f"order={field._order_str}" if mode == "repr" else ""
        suffix = ")" if mode == "repr" else ""
        formatter = field._formatter(self)

        field._element_fixed_width = None  # Do not print with fixed-width
        field._element_fixed_width_counter = 0  # Reset element width counter

        string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        if formatter:
            # We are using special print methods and must perform element alignment ourselves. We will print each
            # element a second time use the max width of any element observed on the first array2string() call.
            field._element_fixed_width = field._element_fixed_width_counter

            string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        field._element_fixed_width = None
        field._element_fixed_width_counter = 0

        # Determine the width of the last line in the string
        if mode == "repr":
            idx = string.rfind("\n") + 1
            last_line_width = len(string[idx:] + ", " + order + suffix)

            if last_line_width <= np.get_printoptions()["linewidth"]:
                output = prefix + string + ", " + order + suffix
            else:
                output = prefix + string + ",\n" + " " * len(prefix) + order + suffix
        else:
            output = prefix + string + suffix

        return output

    @classmethod
    def _formatter(cls, array):
        """
        Returns a NumPy printoptions "formatter" dictionary.
        """
        pass

    @classmethod
    def _print_int(cls, element: Self) -> str:
        """
        Prints a single element in the integer representation. This is only needed for dtype=object arrays.
        """
        pass

    @classmethod
    def _print_poly(cls, element: Self) -> str:
        """
        Prints a single element in the polynomial representation.
        """
        pass

    @classmethod
    def _print_power(cls, element: Self) -> str:
        """
        Prints a single element in the power representation.
        """
        pass


def _poly_det(A: np.ndarray) -> Poly:
    """
    Computes the determinant of a matrix of `Poly` objects.
    """
    pass


def _characteristic_poly_element(a: FieldArray) -> Poly:
    """
    Computes the characteristic polynomial of the finite field element `a` over its prime subfield.
    """
    pass


def _characteristic_poly_matrix(A: FieldArray) -> Poly:
    """
    Computes the characteristic polynomial of the Galois field matrix `A`.
    """
    pass


def _minimal_poly_element(a: FieldArray) -> Poly:
    r"""
    Compute the standard minimal polynomial of the finite field element `a` over its prime subfield $\mathrm{GF}(p)$.
    """
    pass


def _minimal_poly_matrix(A: FieldArray) -> Poly:
    r"""
    Computes the standard minimal polynomial of the finite field matrix `A`.
    """
    pass


def _nullity(A: FieldArray) -> int:
    """
    Computes the nullity of the matrix A, i.e. the dimension of its null space.
    """
    pass
