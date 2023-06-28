from typing import Callable
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import equinox as eqx
from .custom_types import Array, Key


class SparseLinear(eqx.Module):
    """Linear layer with sparse weights and dense biases."""

    W: sparse.BCOO
    B: Array

    def __init__(
        self,
        rng: Key,
        in_dims: int,
        out_dims: int,
        dense_rows: int | Array,
        dense_cols: int | Array,
        sparsity: float = 0.8,
    ):
        """Initialize a sparse linear layer.

        Inputs:
        =======
        rng: Key
            Random number generator key.
        in_dims: int
            Number of input dimensions.
        out_dims: int
            Number of output dimensions.
        dense_rows: int | Array
            Number of dense rows or indices of dense rows. If an integer is passed,
            randomly selects that many rows to be dense.
        dense_cols: int | Array
            Number of dense columns or indices of dense columns. If an integer is passed,
            randomly selects that many columns to be dense.
        sparsity: float
            Probability of a weight being zero, not including the dense rows and columns.

        Outputs:
        ========
        None
        """
        keys = jax.random.split(rng, 5)
        weights = jax.random.normal(keys[0], (in_dims, out_dims)) / jnp.sqrt(
            in_dims * out_dims
        )
        sparse_mask = jax.random.choice(
            keys[1], 2, (in_dims, out_dims), p=jnp.array([sparsity, 1 - sparsity])
        )

        if isinstance(dense_rows, Array):
            sparse_mask = sparse_mask.at[dense_rows].set(1)
        elif isinstance(dense_rows, int):
            fill_rows = jax.random.choice(
                keys[2], in_dims, (dense_rows,), replace=False
            )
            sparse_mask = sparse_mask.at[fill_rows].set(1)

        if isinstance(dense_cols, Array):
            sparse_mask = sparse_mask.at[:, dense_cols].set(1)
        elif isinstance(dense_cols, int):
            fill_cols = jax.random.choice(
                keys[3], out_dims, (dense_cols,), replace=False
            )
            sparse_mask = sparse_mask.at[:, fill_cols].set(1)

        self.W = sparse.BCOO.fromdense(weights * sparse_mask)
        self.B = jax.random.normal(keys[4], (out_dims,))

    def __call__(self, x, key=None):
        """Compute the output of the linear layer.

        Inputs:
        =======
        x: Array
            Input array.

        Outputs:
        ========
        out: Array
            Output array.
        """
        return self.W.T @ x + self.B

    def n_params(self):
        return self.W.nse + self.B.size


class SparseMLP(eqx.Module):
    """Multi-layer perceptron with sparse linear layers (see SparseLinear)."""

    layers: list

    def __init__(
        self,
        rng: Key,
        in_dims: int,
        out_dims: int,
        hidden_dims: tuple[int, ...] = (100, 100),
        act: Callable = jax.nn.leaky_relu,
        act_final: Callable = lambda x: x,
        sparsity: float = 0.95,
        dense_rows: int = 0,
        dense_cols: int = 0,
    ):
        """Initialize a sparse MLP.

        Inputs:
        =======
        rng: Key
            Random number generator key.
        in_dims: int
            Number of input dimensions.
        out_dims: int
            Number of output dimensions.
        hidden_dims: tuple[int, ...]
            Dimensions of hidden layers.
        act: Callable
            Activation function for hidden layers. Defaults to leaky ReLU.
        act_final: Callable
            Activation function for final layer. Defaults to identity.
        sparsity: float
            Sparsity of SparseLinear layers.
        dense_rows: int
            Number of dense rows in each SparseLinear layer.
        dense_cols: int
            Number of dense columns in each SparseLinear layer.

        Outputs:
        ========
        None
        """

        layers = []

        dims = [in_dims, *hidden_dims, out_dims]
        depth = len(dims) - 1

        keys = jax.random.split(rng, depth)
        activations = [eqx.nn.Lambda(act)] * (depth - 1) + [eqx.nn.Lambda(act_final)]

        for i in range(depth):
            layers.append(
                SparseLinear(
                    keys[i],
                    dims[i],
                    dims[i] + 1,
                    dense_rows,
                    dense_cols,
                    sparsity=sparsity,
                )
            )
            layers.append(activations[i])

        self.layers = layers

    def __call__(self, x, key=None):
        """Compute the output of the MLP.

        Inputs:
        =======
        x: Array
            Input array.

        Outputs:
        ========
        out: Array
            Output array.
        """
        for layer in self.layers:
            x = layer(x, key)
        return x
