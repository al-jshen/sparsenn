import functools as ft
from collections.abc import Callable
from typing import (
    Any,
    cast,
    Literal,
    overload,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec

from jaxtyping import Array, ArrayLike, Complex, Float, PyTree

import equinox as eqx

from .custom_types import sentinel
from .ad import value_and_grad as sparse_value_and_grad


_P = ParamSpec("_P")
_T = TypeVar("_T")


class _ValueAndGradWrapper(eqx.Module):
    _fun: Callable
    _has_aux: bool
    _gradkwargs: dict[str, Any]

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, x, /, *args, **kwargs):
        @ft.partial(sparse_value_and_grad, has_aux=self._has_aux, **self._gradkwargs)
        def fun_value_and_grad(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = eqx.combine(_diff_x, _nondiff_x)
            return self._fun(_x, *_args, **_kwargs)

        diff_x, nondiff_x = eqx.partition(x, eqx.is_array)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return eqx.Partial(self, instance)


class _GradWrapper(eqx.Module):
    _fun_value_and_grad: _ValueAndGradWrapper
    _has_aux: bool

    @property
    def __wrapped__(self):
        return self._fun_value_and_grad

    def __call__(self, /, *args, **kwargs):
        value, grad = self._fun_value_and_grad(*args, **kwargs)
        if self._has_aux:
            _, aux = value
            return grad, aux
        else:
            return grad

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return eqx.Partial(self, instance)


_Scalar = Union[float, complex, Float[ArrayLike, ""], Complex[ArrayLike, ""]]


@overload
def filter_value_and_grad(
    *, has_aux: Literal[False] = False
) -> Callable[[Callable[_P, _Scalar]], Callable[_P, tuple[_Scalar, PyTree]]]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, _Scalar], *, has_aux: Literal[False] = False
) -> Callable[_P, tuple[_Scalar, PyTree]]:
    ...


@overload
def filter_value_and_grad(
    *, has_aux: Literal[True] = True
) -> Callable[
    [Callable[_P, tuple[_Scalar, _T]]], Callable[_P, tuple[tuple[_Scalar, _T], PyTree]]
]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, tuple[_Scalar, _T]], *, has_aux: Literal[True] = True
) -> Callable[_P, tuple[tuple[_Scalar, _T], PyTree]]:
    ...


@overload
def filter_value_and_grad(
    fun: Callable[_P, _T], *, has_aux: bool = False
) -> Callable[_P, tuple[_T, PyTree]]:
    ...


def filter_value_and_grad(
    fun=sentinel, *, has_aux: bool = False, **gradkwargs
) -> Callable:
    """Creates a function that evaluates both `fun` and the gradient of `fun`.

    The gradient will be computed with respect to all floating-point JAX/NumPy arrays
    in the first argument. (Which should be a PyTree.)

    Any nondifferentiable leaves in the first argument will have `None` as the gradient.

    **Arguments:**

    - `fun` is a pure function to differentiate.
    - `has_aux`: if `True` then `fun` should return a pair; the first element is the
        output to be differentiated and the second element is auxiliary data.

    **Returns:**

    A function with the same arguments as `fun`, that evaluates both `fun` and computes
    the derivative of `fun` with respect to its first input. Any nondifferentiable
    leaves will have `None` as the gradient.

    If `has_aux` is `True` then a nested tuple `((value, aux), gradient)` is returned.
    If `has_aux` is `False` then the pair `(value, gradient)` is returned.
    """

    if fun is sentinel:
        return ft.partial(filter_value_and_grad, has_aux=has_aux, **gradkwargs)

    argnums = gradkwargs.pop("argnums", None)
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    return eqx.module_update_wrapper(
        _ValueAndGradWrapper(fun, has_aux, gradkwargs), fun
    )


@overload
def filter_grad(
    *, has_aux: Literal[False] = False
) -> Callable[[Callable[_P, _Scalar]], Callable[_P, PyTree[Float[Array, "..."]]]]:
    ...


@overload
def filter_grad(
    fun: Callable[_P, _Scalar], *, has_aux: Literal[False] = False
) -> Callable[_P, PyTree[Float[Array, "..."]]]:
    ...


@overload
def filter_grad(
    *, has_aux: Literal[True] = True
) -> Callable[
    [Callable[_P, tuple[_Scalar, _T]]],
    Callable[_P, tuple[PyTree[Float[Array, "..."]], _T]],
]:
    ...


@overload
def filter_grad(
    fun: Callable[_P, tuple[_Scalar, _T]], *, has_aux: Literal[True] = True
) -> Callable[_P, tuple[PyTree[Float[Array, "..."]], _T]]:
    ...


@overload
def filter_grad(fun: Callable[_P, Any], *, has_aux: bool = False) -> Callable[_P, Any]:
    ...


def filter_grad(fun=sentinel, *, has_aux: bool = False, **gradkwargs):
    """Creates a function that computes the gradient of `fun`.

    The gradient will be computed with respect to all floating-point JAX/NumPy arrays
    in the first argument. (Which should be a PyTree.)

    Any nondifferentiable leaves in the first argument will have `None` as the gradient.

    **Arguments:**

    - `fun` is a pure function to differentiate.
    - `has_aux`: if `True` then `fun` should return a pair; the first element is the
        output to be differentiated and the second element is auxiliary data.

    **Returns:**

    A function with the same arguments as `fun`, that computes the derivative of `fun`
    with respect to its first input. Any nondifferentiable leaves will have `None` as
    the gradient.

    If `has_aux` is `True` then a pair `(gradient, aux)` is returned. If `has_aux` is
    `False` then just the `gradient` is returned.

    !!! tip

        If you need to differentiate multiple objects, then put them together into a
        tuple and pass that through the first argument:
        ```python
        # We want to differentiate `func` with respect to both `x` and `y`.
        def func(x, y):
            ...

        @equinox.filter_grad
        def grad_func(x__y):
            x, y = x__y
            return func(x, y)
        ```

    !!! info

        See also [`equinox.apply_updates`][] for a convenience function that applies
        non-`None` gradient updates to a model.

    """

    if fun is sentinel:
        return ft.partial(filter_grad, has_aux=has_aux, **gradkwargs)

    fun_value_and_grad = filter_value_and_grad(fun, has_aux=has_aux, **gradkwargs)
    fun_value_and_grad = cast(_ValueAndGradWrapper, fun_value_and_grad)
    return eqx.module_update_wrapper(_GradWrapper(fun_value_and_grad, has_aux), fun)
