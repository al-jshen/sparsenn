import equinox as eqx
import jax.tree_util as jtu


def apply_updates(sparse_model, updates):
    splitter = jtu.tree_map(eqx.is_inexact_array, sparse_model)

    # intermediate bcoo will be invalid here because the indices
    # will be set to None but that doesn't matter because the leaves
    # are still valid
    model_diff, model_static = eqx.partition(sparse_model, splitter)

    updates_diff, _ = eqx.partition(updates, splitter)
    model_diff_new = jtu.tree_map(lambda p, g: p + g, model_diff, updates_diff)
    return eqx.combine(model_diff_new, model_static)
