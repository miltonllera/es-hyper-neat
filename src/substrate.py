import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import Sequence
from jaxtyping import Array, Float, Int


class SubstrateInitializer(eqx.Module):
    def __call__(self, pattern, key=None):
        raise NotImplementedError


#------------------------------------- Substate Partition ----------------------------------------

class MultiResolutionPoolingInitializer(SubstrateInitializer):
    """
    Substrate intialization algorithm used for ES-HyperNeat which is compatible with Jax
    transformations, including `jit`.

    Parameters
    ----------
    min_depth: int
        The minimium depth at which the search will start. By default it' 0 i.e. the full pattern.
    max_depth : int
        The maximum depth by which the search will be expanded. If not provided will use the
        maximum resolution of the pattern.
    threshold: float
        The threshold of the variance below which pattern division will be stopped.
    return_aux: bool
        Whether to return the variance and index maps computed during the run. This is useful when
        debuging so it is disabled by default.

    This class replicates the functionality of the standard QuadTree used in ES-HyperNeat using
    a set of MaxPooling convolutions. This enables us to replicate the functionality of said in
    a way that is compatible with Jax transformations.

    Notice that as it stands, this version is only compatible with square patterns whose side is
    a power of 2. This simplies the underlaying implementation and enables useful optimisations to
    the code.
    """
    min_depth: int
    max_depth: int
    threshold: float
    return_aux: bool

    def __init__(
        self,
        min_depth: int = 0,
        max_depth: int = 6,
        threshold: float = 0.01,
        return_aux: bool = False
    ):
        self.threshold = threshold
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.return_aux = return_aux

    def __call__(
        self,
        pattern: Float[Array, "H W C"],
        key: jax.Array | None = None,
    ) -> Float[Array, "H*W 2"] | \
         tuple[Float[Array, "H*W 2"], tuple[Float[Array, "H W"], Int[Array, "H W"]]]:
        """
        This method approximates the QuadTree variance division algorithm by computing all possible
        quadrant variance values between min depth and max_depth - 1. It then uses these maps to
        compute the level at which the division would have stopped for each of quadrant. Cells in
        said quadrant are thus assigned to the same index, which enables the use of `jnp.unique`
        to filter out the repeated ones.

        Parameters
        ----------
        pattern: Float[Array, "H W C"]
            An array representing a pattern which is used to determine the weights and location of
            neurons in a substrate. This can be obtained for example from a CPPN.
        key: jax.Array | None
            Not used, included for compatibility.

        Returns
        -------
        node_grid_idx: Int[Array, "N*N 2"]
            Index of each realized node in the pattern's grid. Notice that to be compatible with
            Jax transformations, the size of this array must be fixed, in which case units that
            were not used will have index -1 i.e. they will be out of bounds.
        aux (optional): tuple[Float[Array, "H*W 2"], tuple[Float[Array, "H W"], Int[Array, "H W"]]]
            A tuple containing the variance maps at each scale along with the quadrant indices that
            were computed during the substrate initialisation. This is mainly useful for debugging
            and in fact no calling class should expect it as part of the output.
        """
        h, w = pattern.shape[:2]
        assert h == w  # enforce square patterns

        pattern_t = jnp.moveaxis(pattern, -1, 0)
        map_h = self.max_depth_grid_size(h)
        pools = self.get_pools(h)

        # use this to recover the final weights by averaging the values in the level
        # pooled_patterns = [p(pattern_t) for p in pools]
        # pooled_patterns = jnp.stack(
        #     [jax.image.resize(m, (1, map_h, map_h), 'nearest') for m in pooled_patterns]
        # )

        # Normalise the pattern
        pattern_t = (pattern_t - pattern_t.min()) / abs(pattern_t.max() - pattern_t.min())

        # Var(x) = E^2[x] - E[x^2]
        sqr_pattern_t = pattern_t ** 2
        var_maps = [p(sqr_pattern_t) - p(pattern_t) ** 2 for p in pools]
        var_maps = jnp.stack([
            jax.image.resize(m.squeeze(0), (map_h, map_h), 'nearest') for m in var_maps
        ])

        # NOTE: The final level is the guard, which means it should return true regardless of
        # whether it is greater that the variance threshold. Thus we do not need to compute this
        # final level to begin with. Notice that self.get_pools function returns pools from 0 to
        # max_depth - 1, so we can just concatenate a map of True values as the final comparison.
        is_below_threshold = jnp.concatenate([
            var_maps < self.threshold,
            jnp.ones((1, map_h, map_h), dtype=bool)
        ])

        arg_max_map = jnp.moveaxis(is_below_threshold.argmax(0, keepdims=True), 0, -1)

        # Compute center index position of each patch
        # NOTE: The output will be in Cartesian coordinates, meaning the output shape is reversed
        # with respect to the input. This is fine since the input to the CPPN should be x first
        # and y second.
        x = jnp.arange(0, w, dtype=int)
        y = jnp.arange(0, h, dtype=int)
        xy = jnp.stack(jnp.meshgrid(y, x, indexing='ij'), axis=-1)

        # NOTE: this is the tricky part of the algorithm, so I am leaving a short note
        # For each cell we have the depth at which division in the QuadTree would have stopped
        # stored 'arg_max_map' (we must correct for minimum tree depth). To compute the quadrant
        # to which they belong notice that we just need apply integer division to their position
        # (at the maximum depth) by the power of 2 corresponding to that cell in 'arg_max_map':
        #     - if depth == 0 => xy // (h // 2 ** 0) is 0 for all values, since in xy < h
        #     - if depth == 3 => xy // (h // 2 ** 3) then you get a position in the 8x8 grid
        #         e.g. if xy == (4, 4) and h == 64 => quadrant_xy == (0, 0)
        #         e.g. if xy == (48, 19) and h == 64 => quadrant_xy == (5, 3)
        # Note that we don't need to perform the double integer divion above, instead we can use
        # right/left shifts. We also need to move the position of each cell to the center of the
        # quadrant which we can do by adding the next smallest power of two in the scale.

        log2_h = jnp.log2(h).astype(int)
        depth_grid = arg_max_map + self.min_depth
        node_grid_idx = xy >> log2_h - depth_grid
        node_grid_idx = (node_grid_idx << log2_h - depth_grid) + (h >> depth_grid + 1)

        # Use unique to remove repeated values of a quadrant. Fill value is set to '[h, 0]' so that
        # unused nodes have grid index equal to the smallest possible value which is out of bounds.
        # This enables useful optimisations and simplifies the code when using 'jnp.take/unique'.
        node_grid_idx = node_grid_idx.reshape(-1, 2)
        node_grid_idx = jnp.unique(
            node_grid_idx, size=self.max_cells(h), fill_value=jnp.array([h, 0]), axis=0)

        if self.return_aux:  # for debugging
            return node_grid_idx, (var_maps, arg_max_map)

        return node_grid_idx

    def get_pools(self, pattern_size) -> Sequence[nn.AvgPool2d]:
        pools = []
        for depth in range(self.min_depth, self.max_depth):
            if (1 << depth) == pattern_size:
                break

            pools.append(nn.AvgPool2d(
                    kernel_size=pattern_size >> depth,
                    stride=pattern_size >> depth,
                    padding=0
                ))

        return pools

    def max_depth_grid_size(self, pattern_resolution):
        # return min(pattern_resolution, 1 << self.max_depth)
        return pattern_resolution // max(1, pattern_resolution >> self.max_depth)

    def max_cells(self, pattern_resolution):
        depth_grid_height = self.max_depth_grid_size(pattern_resolution)
        return depth_grid_height * depth_grid_height
