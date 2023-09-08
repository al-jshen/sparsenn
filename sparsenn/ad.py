import itertools
from typing import Any, Callable, Sequence, Union

import jax
import jax.numpy as jnp
from jax import tree_util
from jax._src import core
from jax._src.api_util import _ensure_index, _ensure_index_tuple
from jax._src.traceback_util import api_boundary
from jax._src.util import split_list, wraps
from jax.experimental.sparse._base import JAXSparse
from jax.flatten_util import ravel_pytree
from jax.util import safe_zip

is_sparse = lambda x: isinstance(x, JAXSparse)


def flatten_fun_for_sparse_ad(fun, argnums: Union[int, tuple[int]], args: tuple[Any]):
    argnums_tup = _ensure_index_tuple(argnums)
    assert all(0 <= argnum < len(args) for argnum in argnums_tup)

    # We do a two-step flattening to figure out how argnums maps to args_flat.
    # First, flatten arguments to a list containing sparse and dense objects.
    args_flat1, tree1 = tree_util.tree_flatten(args, is_leaf=is_sparse)
    *leaf_argnums1, end = split_list(
        range(tree1.num_leaves), [child.num_leaves for child in tree1.children()]
    )
    assert not end
    argnums_flat1 = list(
        itertools.chain.from_iterable(
            nums for i, nums in enumerate(leaf_argnums1) if i in argnums_tup
        )
    )

    # Next, fully flatten to a list of dense buffers.
    args_flat, tree2 = tree_util.tree_flatten(args_flat1)
    *leaf_argnums2, end = split_list(
        range(tree2.num_leaves), [child.num_leaves for child in tree2.children()]
    )
    assert not end
    # For sparse args, we only mark the first buffer (the data) for differentiation.
    leaf_argnums2 = [
        nums[:1] if is_sparse(arg) else nums
        for arg, nums in safe_zip(args_flat1, leaf_argnums2)
    ]
    argnums_flat = tuple(
        itertools.chain.from_iterable(
            nums for i, nums in enumerate(leaf_argnums2) if i in argnums_flat1
        )
    )

    def fun_flat(*args_flat, **kwargs):
        args = tree_util.tree_unflatten(
            tree1, tree_util.tree_unflatten(tree2, args_flat)
        )
        return fun(*args, **kwargs)

    def reconstruct(i, grad_out):
        bufs, tree = tree_util.tree_flatten(args_flat1[i])
        f_recons = lambda g: tree_util.tree_unflatten(tree, [g, *bufs[1:]])
        for _ in range(grad_out.ndim - bufs[0].ndim):
            f_recons = jax.vmap(f_recons)
        return f_recons(grad_out)

    single_argnum = isinstance(argnums, int)

    _, unravel = ravel_pytree(tuple((args[i] for i in argnums_tup)))

    def postprocess_gradients(grads_out):
        out = [reconstruct(*args) for args in safe_zip(argnums_flat1, grads_out)]
        out_flat, _ = ravel_pytree(out)
        result = unravel(out_flat)
        if single_argnum:
            return result[0]
        return result

    return fun_flat, argnums_flat, args_flat, postprocess_gradients


def value_and_grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux=False, **kwargs
) -> Callable[..., tuple[Any, Any]]:
    """Sparse-aware version of :func:`jax.value_and_grad`

    Arguments and return values are the same as :func:`jax.value_and_grad`, but when
    taking the gradient with respect to a :class:`jax.experimental.sparse` array, the
    gradient is computed in the subspace defined by the array's sparsity pattern.

    Example:

      >>> from jax.experimental import sparse
      >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
      >>> y = jnp.ones(6)
      >>> sparse.value_and_grad(lambda X, y: X @ y)(X, y)
      (Array(15., dtype=float32), BCOO(float32[6], nse=5))
    """
    raw_value_and_grad_fun = jax.value_and_grad(
        fun, argnums=argnums, has_aux=has_aux, **kwargs
    )
    argnums = core.concrete_or_error(_ensure_index, argnums)

    @wraps(fun, docstr=raw_value_and_grad_fun.__doc__, argnums=argnums)
    @api_boundary
    def value_and_grad_fun(*args, **kwargs):
        (
            fun_flat,
            argnums_flat,
            args_flat,
            postprocess_gradients,
        ) = flatten_fun_for_sparse_ad(fun, argnums, args)
        val_out, grad_out = jax.value_and_grad(
            fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs
        )(*args_flat)
        return val_out, postprocess_gradients(grad_out)

    return value_and_grad_fun


def grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux=False, **kwargs
) -> Callable:
    """Sparse-aware version of :func:`jax.grad`

    Arguments and return values are the same as :func:`jax.grad`, but when taking
    the gradient with respect to a :class:`jax.experimental.sparse` array, the
    gradient is computed in the subspace defined by the array's sparsity pattern.

    Example:

      >>> from jax.experimental import sparse
      >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
      >>> y = jnp.ones(6)
      >>> sparse.grad(lambda X, y: X @ y)(X, y)
      BCOO(float32[6], nse=5)
    """
    raw_grad_fun = jax.grad(fun, argnums=argnums, **kwargs)
    argnums = core.concrete_or_error(_ensure_index, argnums)

    @wraps(fun, docstr=raw_grad_fun.__doc__, argnums=argnums)
    @api_boundary
    def grad_fun(*args, **kwargs):
        (
            fun_flat,
            argnums_flat,
            args_flat,
            postprocess_gradients,
        ) = flatten_fun_for_sparse_ad(fun, argnums, args)
        out = jax.grad(fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs)(
            *args_flat
        )
        if has_aux:
            return postprocess_gradients(out[0]), out[1]

        return postprocess_gradients(out)

    return grad_fun


import functools as ft
from collections.abc import Callable
from typing import Any, Literal, TypeVar, Union, cast, overload

import equinox as eqx
from jaxtyping import Array, ArrayLike, Complex, Float, PyTree
from typing_extensions import ParamSpec

from .custom_types import sentinel

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
        @ft.partial(value_and_grad, has_aux=self._has_aux, **self._gradkwargs)
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
