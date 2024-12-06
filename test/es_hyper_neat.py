import pyrootutils

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tensorneat.common import ACT, AGG
from src.neat import NEAT
from src.dev import DevelopedFeedForwardPolicy, HyperCubeDevModel, NEATForwardPolicyShaper
from src.cppn import GraphCPPN
from src.substrate import MultiResolutionPoolingInitializer

import matplotlib.pyplot as plt
from src.analysis.hypercube_dev import plot_cppn


# Algorithm

neat_args = {
    'popsize': 10,
    'species_size': 1,
    'max_stagnation': 0,
    'species_elitism': 1,
    'spawn_number_change_rate': 0.5,
    'genome_elitism': 1,
    'survival_threshold': 0.1,
    'min_species_size': 1,
    'compatibility_threshold': 2.0,
    'species_fitness_func': jnp.max,
}

genome_args = {
    'genome_type': 'default',
    'max_nodes': 10,
    'max_conns': 10,
    'num_inputs': 2,
    'num_outputs': 1,
    'init_hidden_layers': tuple(),
    'node_add': 0.5,
    'node_delete': 0.0,
    'conn_add': 0.5,
    'conn_delete': 0.0,
    'output_transform': ACT.identity,  # type: ignore
}

node_gene_args = {
    'gene_type': 'bias',
    'bias_init_std': 0.1,
    'bias_mutate_power': 0.05,
    'bias_mutate_rate': 0.5,
    'bias_replace_rate': 0.0,
    'activation_options': ['identity', 'tanh', 'gaussian', 'sin'],  # type: ignore
    'aggregation_options': [AGG.sum],  # type: ignore
}

conn_gene_args = {
    'gene_type': 'default',
    'weight_init_mean': 0.0,
    'weight_init_std': 0.1,
    'weight_mutate_power': 0.05,
    'weight_mutate_rate': 0.5,
    'weight_replace_rate': 0.0,
}


neat = NEAT(neat_args, genome_args, node_gene_args, conn_gene_args)  # type: ignore
cppn = GraphCPPN.from_hyperparameters(
    n_inputs=2,
    n_hidden=6,
    n_outputs=1,
    act_fn_defs=['identity', 'identity', 'identity', 'identity'],
    key=jr.key(0),

)
substrate = MultiResolutionPoolingInitializer(max_depth=6, threshold=0.01, return_aux=False)
hyperdev = HyperCubeDevModel(
    input_pos=3,
    output_pos=1,
    cppn=cppn,
    substrate=substrate,
    grid_size=(64, 64),
)

# Policy network with developmental component
dev_policy = DevelopedFeedForwardPolicy(
    n_observations=5,
    n_actions=1,
    n_layers=1,
    output_fn=jax.nn.tanh,
    dev_model=hyperdev,
    params_formatter=NEATForwardPolicyShaper(),
)

params, static = dev_policy.partition()

state = neat.init(jr.key(19))

params, _ = neat.ask(jr.key(10), state)
_, nodes, _, _ = params
state = neat.tell(params, jr.normal(jr.key(32), (len(nodes),)), state)
params, _ = neat.ask(jr.key(32), state)

seqs, nodes, conns, u_conns = params

i = 5
ex_cppn = static.instantiate((seqs[i], nodes[i], conns[i], u_conns[i])).dev_model.cppn

# print(seqs[i])
# print(ex_cppn.nodes)
# print(ex_cppn.node_fn)
print(ex_cppn.adjacency)

result = ex_cppn(jnp.asarray([1.0]), jnp.asarray([1.0]))
print(np.asarray(result))

# plot_cppn(ex_cppn)
# plt.show()

