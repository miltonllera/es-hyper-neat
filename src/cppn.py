import numpy as np
import jax
import jax.nn as jnn
import jax.random as jr
import jax.lax as lax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import Callable, Literal, Mapping, Sequence
from jaxtyping import Array, Bool, Float, Int


def gaussian_fn(x, a=1, b=0, c=1):
    return a * jnp.exp(-(x - b) ** 2) / (2 * c ** 2)


default_act_fns = {
    'identity': lambda x: x,
    'gaussian': gaussian_fn,
    'sin': lambda x: jnp.sin(jnp.pi * x),
    'cos': lambda x: jnp.cos(jnp.pi * x),
    'tanh': jnn.tanh,
    'sigmoid': jnn.sigmoid,
    'abs': jnp.abs,
    'relu': jnn.relu,
    'step': lambda x: (x > 0.0).astype(jnp.float32),
    'flip': lambda x: -x,
}


class CPPN(eqx.Module):
    n_inputs: int
    n_outputs: int
    act_fns: Sequence[Callable[[Float[Array, "H"]], Float[Array, "H"]]]
    act_fns_names: Sequence[str]
    relative_positions: Literal['no', 'to_origin', 'to_target']

    def __call__(
        self,
        inputs: np.ndarray | Float[Array, "..."],
        key=None
    )-> Float[Array, "..."]:
        raise NotImplementedError


#--------------------------------------- Graph CPPN ----------------------------------------------

class GraphCPPN(CPPN):
    n_hidden: int
    nodes: Bool[Array, "N"]
    adjacency: Float[Array, "N N"]
    bias: Float[Array, "N"]
    node_fn: Int[Array, "N"]

    """
    CPPN implemented as a directed acyclyc graph (DAG).

    A CPPN is represented using:
        - A node mask that determines which nodes are being used.
        - An activation function index which specifies which activation function to apply to each
        node after the linear transformation from the acttivation function list.
        - A bias term for each node.
        - An adjacency matrix that encodes the weights between connections.

    Note that it is assumed that the adjacency matrix respects the DAG property as well as the
    abscence of input-to-input and output-to-output connections. We can change the number of input
    units to reflect different CPPN designs and the number of hidden and output units can be
    specified during construction.

    The __call__ method runs the CPPN as a standard graph neural network. Because nodes could
    be layed out in sequence, we always run the maximum of N - 4 iterations to ensure that all
    updates propagate correctly. In the future it would be better to use a while loop.
    """
    def __call__(
        self,
        inputs: np.ndarray | Float[Array, "N"],
        key: jax.Array | None = None
    ) -> Float[Array, "..."]:
        assert len(inputs) == self.n_inputs

        h = jnp.zeros((self.adjacency.shape[0],))
        h = h.at[:len(inputs)].set(inputs)
        h = self.apply_non_linearities(h)

        # TODO: replace this by a while loop
        def update(h, _):
            aggregate = self.adjacency.T @ h + self.bias
            h = self.apply_non_linearities(aggregate)
            h = h.at[:len(inputs)].set(inputs)  # type: ignore
            return h, h

        h = lax.scan(update, h, xs=None, length=h.shape[0] - len(inputs))[0]

        last_idx = jnp.where(self.nodes, jnp.arange(len(self.nodes)), -1).argmax()
        return lax.dynamic_slice(h, [last_idx - (self.n_outputs - 1)], [self.n_outputs])

    def apply_non_linearities(self, h):
        return jax.vmap(lax.switch, in_axes=(0, None, 0))(self.node_fn, self.act_fns, h)

    @property
    def n_nodes(self):
        return self.n_inputs + self.n_outputs + self.n_hidden

    @classmethod
    def from_hyperparameters(
        cls,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        act_fn_defs: Sequence[str] | Mapping[str, Callable[[jax.Array], jax.Array]] | None = None,
        relative_positions: Literal['no', 'to_origin', 'to_target'] = 'no',
        *,
        key:jax.Array
    ) -> "GraphCPPN":
        if act_fn_defs is None:
            act_fn_defs = default_act_fns
        elif isinstance(act_fn_defs, list) and all(map(lambda x: isinstance(x, str), act_fn_defs)):
            act_fn_defs = {x: default_act_fns[x] for x in act_fn_defs}
        else:
            raise RuntimeError("Invalid activation function list")

        act_fns_names, act_fns = list(zip(*act_fn_defs.items()))  # type: ignore
        total_nodes = n_inputs + n_hidden + n_outputs

        idx = jnp.asarray(list(range(n_inputs)) + list(range(n_inputs + n_hidden, total_nodes)))
        nodes = jnp.zeros((total_nodes,), dtype=bool).at[idx].set(True)
        # random activation functions for each node
        node_fn = jr.categorical(key, jnp.ones(len(act_fns)) / len(act_fns), shape=(total_nodes,))
        # output nodes should not have activation functions
        node_fn = node_fn.at[-n_outputs:].set(0)
        node_fn = node_fn.at[:n_inputs].set(0)

        # Minimal random connectivity from all inputs to all outputs
        adj = jr.normal(key, (n_inputs, n_outputs))
        adj = jnp.zeros((total_nodes, total_nodes)).at[:n_inputs, -n_outputs:].set(adj)
        bias = jnp.zeros((total_nodes,))

        return cls(
            n_inputs,
            n_outputs,
            act_fns,
            act_fns_names,
            relative_positions,
            n_hidden,
            nodes,
            adj,
            bias,
            node_fn,
        )
