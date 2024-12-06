import jax
import jax.numpy as jnp
import jax.tree as tree
from tensorneat.algorithm.neat import NEAT as WrappedNEAT, State
from tensorneat.genome import (
    BiasNode,
    DefaultDistance,
    DefaultConn,
    DefaultGenome,
    DefaultMutation,
)
from tensorneat.genome.default import ACT, AGG


def gaussian_fn(x, a=1, b=0, c=1):
    return a * jnp.exp(-(x - b) ** 2) / (2 * c ** 2)

if not 'gaussian' in ACT.name2jnp:
    ACT.add_func('gaussian', gaussian_fn)


def init_genome(node_gene, conn_gene, **kwargs):
    genome_type = kwargs.pop('genome_type')
    mutation = DefaultMutation(
        conn_add=kwargs.pop('conn_add'),
        conn_delete=kwargs.pop('conn_delete'),
        node_add=kwargs.pop('node_add'),
        node_delete=kwargs.pop('node_delete')
    )
    distance = DefaultDistance(
        compatibility_disjoint=kwargs.pop('compatibility_disjoint'),
        compatibility_weight=kwargs.pop('compatibility_weight'),
    )
    if genome_type == 'default':
        return DefaultGenome(
            mutation=mutation,
            node_gene=node_gene,
            conn_gene=conn_gene,
            distance=distance,
            **kwargs
        )
    else:
        raise NotImplementedError


def init_node_gene(**kwargs):
    gene_type = kwargs.pop('gene_type')
    if gene_type == 'bias':
        return BiasNode(**kwargs)
    else:
        raise NotImplementedError()


def init_conn_gene(**kwargs):
    gene_type = kwargs.pop('gene_type')
    if gene_type == 'default':
        return DefaultConn(**kwargs)
    else:
        raise NotImplementedError()


def get_activations(activations):
    def _get(act):
        if callable(act):
            return act
        elif isinstance(act, str):
            return getattr(ACT, act)
        else:
            raise RuntimeError

    if isinstance(activations, (list, tuple)):
        return [_get(act) for act in activations]

    return _get(activations)


def get_aggregations(aggregations):
    def _get(agg):
        if callable(agg):
            return agg
        elif isinstance(agg, str):
            return getattr(AGG, agg)
        else:
            raise RuntimeError

    if isinstance(aggregations, (list, tuple)):
        return [_get(agg) for agg in aggregations]



class NEAT:
    def __init__(
        self,
        args: dict,
        genome_args: dict,
        node_args: dict,
        conn_args: dict,
    ) -> None:
        fitness_fn = args['species_fitness_func']
        if fitness_fn == "max":
            fitness_fn = jnp.max
        else:
            fitness_fn = jnp.min

        # NOTE: It's probably not necessary to have these as hyperparameters since we are not using
        # tensorneat's forward propagation anyways. A list of indices is probably enough.
        genome_args['output_transform'] = get_activations(genome_args['output_transform'])
        node_args['activation_options'] = get_activations(node_args['activation_options'])
        node_args['aggregation_options'] = get_aggregations(node_args['aggregation_options'])

        args['species_fitness_func'] = fitness_fn

        # self.args = args
        # self.genome_args = genome_args
        # self.node_args = node_args
        # self.conn_args = conn_args

        node_gene = init_node_gene(**node_args)
        conn_gene = init_conn_gene(**conn_args)
        genome = init_genome(node_gene, conn_gene, **genome_args)

        args['pop_size'] = args.pop('popsize')

        self._neat = WrappedNEAT(genome, **args)


    def init(self, rng: jax.Array):
        state = State(best_fitness=-jnp.inf, randkey=rng)
        state = self._neat.setup(state)
        best_member = self._neat.transform(state, (state.pop_nodes[0], state.pop_conns[0]))

        state = state.register(
            best_member=best_member,
        )

        return state

    def ask(self, key: jax.Array, state: State):  # type: ignore
        params = (state.pop_nodes, state.pop_conns)
        return jax.vmap(self._neat.transform, in_axes=(None, 0))(state, params), state

    def tell(self, pop, fitness, state):   # type: ignore
        state = self._neat.tell(state, fitness)

        # TODO: this is hardcoded to a max function, but it could be min
        max_idx = jnp.argmax(fitness)
        replace_best = fitness[max_idx] > state.best_fitness

        best_fitness = jax.lax.select(
            replace_best, fitness[max_idx], state.best_fitness
        )

        best_pop = tree.map(lambda x: x[max_idx], pop)

        def select(x, y):
            return jax.lax.select(replace_best, x, y)

        best_member = tree.map(select, best_pop, state.best_member)

        state = state.update(
            best_fitness=best_fitness,
            best_member=best_member,
        )

        return state

    @property
    def params_strategy(self):
        return None
