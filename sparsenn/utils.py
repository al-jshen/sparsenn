from jax.flatten_util import ravel_pytree
import equinox as eqx


def nparams(model):
    """Number of parameters in a model."""
    return len(ravel_pytree(eqx.filter(model, eqx.is_inexact_array))[0])
