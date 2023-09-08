# mostly taken from netket

import sys
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.api_util import argnums_partial


class HashablePartial(partial):
    """A class behaving like functools.partial, but that retains it's hash if
    it's created with a lexically equivalent (the same) function and with the
    same partially applied arguments and keywords.

    It also stores the computed hash for faster hashing.
    """

    # TODO remove when dropping support for Python < 3.10
    def __new__(cls, func, *args, **keywords):
        # In Python 3.10+ if func is itself a functools.partial instance,
        # functools.partial.__new__ would merge the arguments of this HashablePartial
        # instance with the arguments of the func
        # Pre 3.10 this does not happen, so here we emulate this behaviour recursively
        # This is necessary since functools.partial objects do not have a __code__
        # property which we use for the hash
        # For python 3.10+ we still need to take care of merging with another HashablePartial
        while isinstance(
            func, partial if sys.version_info < (3, 10) else HashablePartial
        ):
            original_func = func
            func = original_func.func
            args = original_func.args + args
            keywords = {**original_func.keywords, **keywords}
        return super(HashablePartial, cls).__new__(cls, func, *args, **keywords)

    def __init__(self, *args, **kwargs):
        self._hash = None

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.func.__code__ == other.func.__code__
            and self.args == other.args
            and self.keywords == other.keywords
        )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (self.func.__code__, self.args, frozenset(self.keywords.items()))
            )

        return self._hash

    def __repr__(self):
        return f"<hashable partial {self.func.__name__} with args={self.args} and kwargs={self.keywords}, hash={hash(self)}>"


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unchunk(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, residual = divmod(n, chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by chunk_size."
            + f"\n            Got x.shape={x.shape} but chunk_size={chunk_size}."
        )
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


def scanmap(fun, scan_fun, argnums=0):
    """A helper function to wrap f with a scan_fun.

    Example:
        import jax.numpy as jnp
        from functools import partial

        from netket.jax import scanmap, scan_append_reduce

        scan_fun = partial(scan_append_reduce, append_cond=(True, False, False))

        @partial(scanmap, scan_fun=scan_fun, argnums=1)
        def f(c, x):
             y = jnp.sin(x) + c
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)
        c = 1.


        y, s, s2 = f(c, x)
        mean = s/N
        var = s2/N - mean**2
    """

    def f_(*args, **kwargs):
        f = jax.linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_


_tree_add = partial(jax.tree_map, jax.lax.add)
_tree_zeros_like = partial(jax.tree_map, lambda x: jnp.zeros(x.shape, dtype=x.dtype))


def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)


def scan_append_reduce(f, x, append_cond, op=_tree_add):
    """Evaluate f element by element in x while appending and/or reducing the
    results.

    Args:
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of bools (if f returns multiple values)
            which indicates whether the individual result should be appended or reduced
        op: a function to (pairwise) reduce the specified results. Defaults to a sum.
    Returns:
        returns the (tuple of) results corresponding to the output of f
        where each result is given by:
        if append_cond is True:
            a (pytree of) array(s) with leading dimension same as x,
            containing the evaluation of f at each element in x
        else (append_cond is False):
            a (pytree of) array(s) with the same shape as the corresponding output of f,
            containing the reduction over op of f evaluated at each x


    Example:

        import jax.numpy as jnp
        from netket.jax import scan_append_reduce

        def f(x):
             y = jnp.sin(x)
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)

        y, s, s2 = scan_append_reduce(f, x, (True, False, False))
        mean = s/N
        var = s2/N - mean**2
    """
    # TODO: different op for each result

    x0 = jax.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_util.tree_leaves(x)[0].shape[0] == 1:
        return _multimap(
            lambda c, x: jnp.expand_dims(x, 0) if c else x, append_cond, f(x0)
        )

    # the original idea was to use pytrees, however for now just operate on the return value tuple
    _get_append_part = partial(_multimap, lambda c, x: x if c else None, append_cond)
    _get_op_part = partial(_multimap, lambda c, x: x if not c else None, append_cond)
    _tree_select = partial(_multimap, lambda c, t1, t2: t1 if c else t2, append_cond)

    carry_init = True, _get_op_part(_tree_zeros_like(jax.eval_shape(f, x0)))

    def f_(carry, x):
        is_first, y_carry = carry
        y = f(x)
        y_op = _get_op_part(y)
        y_append = _get_append_part(y)
        # select here to avoid the user having to specify the zero element for op
        y_reduce = jax.tree_map(
            partial(jax.lax.select, is_first), y_op, op(y_carry, y_op)
        )
        return (False, y_reduce), y_append

    (_, res_op), res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the reduced and appended parts in the two trees
    return _tree_select(res_append, res_op)


scan_append = partial(scan_append_reduce, append_cond=True)
scan_reduce = partial(scan_append_reduce, append_cond=False)


def _fun(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
    n_chunks, n_rest = divmod(n_elements, chunk_size)

    if n_chunks == 0 or chunk_size >= n_elements:
        y = vmapped_fun(*args, **kwargs)
    else:
        # split inputs
        def _get_chunks(x):
            x_chunks = jax.tree_map(lambda x_: x_[: n_elements - n_rest, ...], x)
            x_chunks = _chunk(x_chunks, chunk_size)
            return x_chunks

        def _get_rest(x):
            x_rest = jax.tree_map(lambda x_: x_[n_elements - n_rest :, ...], x)
            return x_rest

        args_chunks = [
            _get_chunks(a) if i in argnums else a for i, a in enumerate(args)
        ]
        args_rest = [_get_rest(a) if i in argnums else a for i, a in enumerate(args)]

        y_chunks = _unchunk(
            scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
        )

        if n_rest == 0:
            y = y_chunks
        else:
            y_rest = vmapped_fun(*args_rest, **kwargs)
            y = jax.tree_map(lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest)
    return y


def _chunk_vmapped_function(
    vmapped_fun: Callable, chunk_size: Optional[int], argnums=0
) -> Callable:
    """takes a vmapped function and computes it in chunks."""

    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)

    return HashablePartial(_fun, vmapped_fun, chunk_size, argnums)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def _vmap_chunked(
    f: Callable, in_axes: int | Tuple[int, ...] = 0, *, chunk_size: Optional[int] = None
) -> Callable:
    """Behaves like jax.vmap but uses scan to chunk the computations in smaller
    chunks.

    This function is essentially equivalent to:

    .. code-block:: python

        nk.jax.apply_chunked(jax.vmap(f, in_axes), in_axes, chunk_size)

    Some limitations to `in_axes` apply.

    Args:
        f: The function to be vectorised.
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`,
            chunking is disabled

    Returns:
        A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)


def vmap_chunked(
    f: Callable,
    in_axes: int | Tuple[int, ...] = 0,
    chunk_size: Optional[int] = None,
    devices: int = 1,
):
    """
    Behaves like jax.vmap but
        1. uses pmap to chunk the computations across multiple devices
        2. uses scan to chunk the device-level computations into even
           smaller chunks for memory efficiency

    Requires that the length of the first dimension of input to be chunked
    is divisible by the number of devices. Does NOT require that this length
    is also divisible by the chunk size.

    Inputs:
        f: Callable
            The function to be vectorised.
        in_axes: int | Tuple[int, ...]
            The axes that should be scanned along. Only supports `0` or `None`.
            Defaults to 0. This should be the same length as the number of
            arguments to `f`.
        chunk_size: Optional[int]
            The maximum size of the chunks to be used per batch (and per device).
            If it is `None`, chunking is disabled.
        devices: int
            The number of devices to split the computation across. Defaults to 1.

    Returns:
        A vectorised and chunked function that has inputs and outputs like `f`
    """

    if devices == 1:
        return _vmap_chunked(f, in_axes, chunk_size=chunk_size)
    else:
        assert devices > 0
        in_axes, argnums = _parse_in_axes(in_axes)

        def inner(*args):
            args_chunked = jax.tree_map(
                lambda x: _chunk(x, x.shape[0] // devices), args
            )

            result = jax.pmap(
                _vmap_chunked(f, in_axes, chunk_size=chunk_size),
                devices=jax.devices()[:devices],
            )(*args_chunked)

            return _unchunk(result)

        return inner
