# sparsenn

Sparse neural networks. Fully compatible with Equinox and Optax. To get started, see the [example](examples/regression.ipynb).

## tl;dr

1. Wrap `optax` optimizer with `sparsenn.flatten`
2. Replace `eqx.filter_value_and_grad` with `sparsenn.filter_value_and_grad`
3. Replace `eqx.apply_updates` with `sparsenn.apply_updates`

## vmap
Use `sparsenn.vmap_chunked(f, in_axes=..., chunk_size=..., devices=...)` instead of `jax.vmap(f, in_axes=...)` to do memory-limited (chunked with `scan`) multi-GPU `vmap`.
