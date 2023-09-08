import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax._src.base as base
from jax.flatten_util import ravel_pytree


def apply_updates(sparse_model, updates):
    splitter = jtu.tree_map(eqx.is_inexact_array, sparse_model)

    # intermediate bcoo will be invalid here because the indices
    # will be set to None but that doesn't matter because the leaves
    # are still valid
    model_diff, model_static = eqx.partition(sparse_model, splitter)

    updates_diff, _ = eqx.partition(updates, splitter)
    model_diff_new = jtu.tree_map(lambda p, g: p + g, model_diff, updates_diff)
    return eqx.combine(model_diff_new, model_static)


def flatten(inner: base.GradientTransformation) -> base.GradientTransformationExtraArgs:
    """Flattens parameters and gradients for init and update of inner
    transform.

    This can reduce the overhead of performing many calculations on lots of small variables, at the cost of slightly increased memory usage.

    Args:
      inner: Inner transformation to flatten inputs for.

    Returns:
      New ``GradientTransformationExtraArgs``
    """

    inner = base.with_extra_args_support(inner)

    def _flatten(params):
        """Flattens and concatenates all tensors in params to a single
        vector."""
        # params, _ = jtu.tree_flatten(params)
        # return jnp.concatenate([jnp.reshape(param, [-1]) for param in params])
        params = eqx.filter(params, eqx.is_inexact_array)
        return ravel_pytree(params)[0]

    def _unflatten(updates, flat):
        """Extracts tensors from flat, using the structure and shapes of
        params."""
        updates_flat, treedef = jtu.tree_flatten(updates)
        offsets = []
        for update in updates_flat:
            size = np.prod(update.shape)
            if offsets:
                offsets.append(size + offsets[-1])
            else:
                offsets.append(size)
        del offsets[-1]
        flat_split = jnp.split(flat, offsets)
        reshaped = [
            jnp.reshape(flat_update, update.shape)
            for flat_update, update in zip(flat_split, updates_flat)
        ]
        return jtu.tree_unflatten(treedef, reshaped)

    def init_fn(params):
        flat = _flatten(params)
        return inner.init(flat)

    def update_fn(updates, state, params=None, **extra_args):
        splitter = jtu.tree_map(eqx.is_inexact_array, updates)
        updates_diff, updates_static = eqx.partition(updates, splitter)
        if params is not None:
            params = _flatten(params)
        updates_new_flat, state = inner.update(
            _flatten(updates_diff), state, params, **extra_args
        )
        updates_new = _unflatten(updates_diff, updates_new_flat)
        updates = eqx.combine(updates_new, updates_static)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
