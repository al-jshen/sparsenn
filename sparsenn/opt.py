import equinox as eqx
from jax.flatten_util import ravel_pytree
from .filter_ad import filter_value_and_grad


def make_sgd_step(fun, lr):
    @eqx.filter_jit
    def wrap(model, *args, **kwargs):
        loss_value, grads = filter_value_and_grad(fun)(model, *args, **kwargs)
        grads_flat, unravel = ravel_pytree(grads)
        diff_model, static_model = eqx.partition(model, eqx.is_array)
        diff_model_flat, _ = ravel_pytree(diff_model)
        diff_model_flat_new = diff_model_flat - lr * grads_flat
        return eqx.combine(unravel(diff_model_flat_new), static_model), loss_value

    return wrap
