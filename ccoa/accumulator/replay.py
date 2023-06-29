"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import chex
import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


@flax.struct.dataclass
class ReplayBufferState:
    trajectories: chex.PyTreeDef
    next_slot: int
    full: bool


class ReplayBuffer:
    """
    Replay buffer that stores a batch of trajectories and allows to sample from them.
    """

    def __init__(self, max_size):
        self.max_size = max_size

    def reset(self, sample_trajectory: chex.PyTreeDef) -> ReplayBufferState:
        next_slot = 0
        full = False

        def batch_empty_like(x):
            return jnp.empty_like(x, shape=(self.max_size, *x.shape))

        trajectories = jtu.tree_map(batch_empty_like, sample_trajectory)

        return ReplayBufferState(trajectories, next_slot, full)

    def add(
        self, state_buffer: ReplayBufferState, trajectory_batch: chex.PyTreeDef
    ) -> ReplayBufferState:
        """
        Add a batch of trajectories to the store of trajectories and overwrite if full (FIFO).
        """
        chex.assert_equal_shape(jtu.tree_leaves(trajectory_batch), dims=0)
        batch_size = len(jtu.tree_leaves(trajectory_batch)[0])

        # If buffer is filled, start replacing values FIFO
        next_slots = jnp.mod(state_buffer.next_slot + jnp.arange(batch_size), self.max_size)

        def set_at_next_slots(x, y):
            return x.at[next_slots].set(y)

        trajectories = jtu.tree_map(set_at_next_slots, state_buffer.trajectories, trajectory_batch)

        # Check if buffer as been filled at least once
        full = jax.lax.cond(
            ((state_buffer.next_slot + batch_size) >= self.max_size),
            lambda _: True,
            lambda _: state_buffer.full,
            None,
        )

        return ReplayBufferState(trajectories, next_slots[-1] + 1, full)

    def sample(
        self, rng: chex.PRNGKey, state_buffer: ReplayBufferState, batch_size: int
    ) -> chex.PyTreeDef:
        """
        Sample a batch of trajectories.
        """
        # Determine range of indeces to sample from
        # NOTE: it is not possible to conditionally sample without replacement in a jit-compatible way
        idx = jax.lax.cond(
            state_buffer.full,
            lambda _: jax.random.randint(rng, (batch_size,), 0, maxval=self.max_size),
            lambda _: jax.random.randint(rng, (batch_size,), 0, maxval=state_buffer.next_slot),
            None,
        )

        return jtu.tree_map(lambda x: x[idx], state_buffer.trajectories)
