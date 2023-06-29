import jax
import jax.numpy as jnp
from flax import struct
import numpy as np

from .base import Environment, Transition


@struct.dataclass
class TreeState:
    done: bool
    depth: int
    position: int
    seed_offset: int


class Tree(Environment):
    """(Overlapping) tree MDP environmnet"""

    def __init__(
        self,
        length,
        branching,
        state_overlap,
        reward_overlap_fraction,
        reward_sparsify_fraction,
        only_reward_last_timestep,
        action_dependent_reward,
        number_different_rewards,
        seed,
        negative_rewards,
    ):
        super().__init__()

        assert branching - state_overlap >= 0 and state_overlap >= 0

        self.total_depth = length
        self.length = length
        self.branching = branching
        self.state_overlap = state_overlap
        self.total_number_states = int(self.compute_total_number_states())
        self.reward_overlap_fraction = reward_overlap_fraction
        self.reward_sparsify_fraction = reward_sparsify_fraction
        self.only_reward_last_timestep = only_reward_last_timestep
        if action_dependent_reward:
            self.large_prime = 9973
        else:
            self.large_prime = 0

        if self.only_reward_last_timestep:
            nb_states_last_timestep = int(self.number_states_at_depth(self.total_depth))
            self.reward_modulo = max(
                int(nb_states_last_timestep * (1 - self.reward_overlap_fraction)), 1
            )
        else:
            self.reward_modulo = max(
                int(self.total_number_states * (1 - self.reward_overlap_fraction)), 1
            )

        if number_different_rewards is not None:
            self.reward_modulo = number_different_rewards

        if negative_rewards:
            self.reward_offset = -max(0, int((self.reward_modulo - 1) // 2))
        else:
            self.reward_offset = 0
        self.negative_rewards = negative_rewards

        self._reward_values = np.arange(
            self.reward_offset, max(self.reward_modulo, 2) + self.reward_offset
        )
        # self.reward_values = jnp.array(self.reward_values, float)
        # if modulo is 1, we always give reward 1 instead of 0, and in case we sparsify, we should still have the option of reward 0.
        self.seed = seed

    @property
    def reward_values(self):
        return self._reward_values

    def number_states_at_depth(self, i):
        if self.branching - self.state_overlap == 1:
            return 1 + self.state_overlap * i
        else:
            return (
                (self.branching - 1) * (self.branching - self.state_overlap) ** i
                - self.state_overlap
            ) / (self.branching - self.state_overlap - 1)

    def compute_total_number_states(self):
        total = 0
        for i in range(self.total_depth):
            total += self.number_states_at_depth(i)
        return total

    def compute_total_number_states_until(self, i):
        base = self.branching - self.state_overlap
        if base == 1:
            return i + 1 + self.state_overlap * (i + 1) * i / 2
        else:
            return (self.branching - 1) / (base - 1) * (base ** (i + 1) - 1) / (base - 1) - (
                i + 1
            ) * self.state_overlap / (base - 1)

    def compute_state_idx(self, depth, position):
        return self.compute_total_number_states_until(depth - 1) + jnp.array([0, position]).max()

    def get_reward(self, state, action):
        depth = state.depth
        position = state.position
        seed_offset = state.seed_offset
        state_idx = jnp.array(self.compute_state_idx(depth, position), int)
        # state_idx = self.compute_state_idx(depth, position).astype(int)
        random_key = jax.random.PRNGKey(
            state_idx + seed_offset + action * self.large_prime
        )  # We are going to sparsify with a
        # certain probability,
        #  but the rewards need to be deterministic in the environment, hence using the state_idx as the seed
        sparsify = jax.random.bernoulli(random_key, p=1.0 - self.reward_sparsify_fraction)
        if self.only_reward_last_timestep:
            sparsify = jax.lax.cond(
                depth == self.total_depth - 1, lambda x: x, lambda x: jnp.array(0, bool), sparsify
            )
            state_idx = position
        if self.reward_modulo == 1:
            return jnp.array(sparsify, float)
        else:
            return jnp.array(
                sparsify
                * (
                    (state_idx + action * self.large_prime + seed_offset) % self.reward_modulo
                    + self.reward_offset
                ),
                float,
            )

    @property
    def num_actions(self):
        return self.branching

    @property
    def observation_shape(self):
        return (3,)

    def get_observation(self, state: TreeState):
        return jnp.array([state.depth, state.position, state.seed_offset])

    def reset(self, rng):
        rng = jax.random.PRNGKey(self.seed)
        seed_offset = jax.random.randint(rng, (), 0, self.total_number_states)
        init_state = TreeState(
            depth=0,
            position=0,
            done=False,
            seed_offset=seed_offset,
        )
        init_transition = Transition(
            observation=self.get_observation(init_state), reward=0.0, done=False, timestep=0
        )
        return init_state, init_transition

    def __step(self, state: TreeState, action):
        reward = self.get_reward(state, action)

        next_depth = state.depth + 1
        done = next_depth >= self.total_depth
        # nb_positions = self.number_states_at_depth(next_depth).astype(int)
        nb_positions = jnp.array(self.number_states_at_depth(next_depth), int)
        next_position = (
            state.position * (self.branching - self.state_overlap)
            + action
            - (self.branching - self.state_overlap) // 2
        ) % nb_positions  # modulo operation with offset such
        next_position = done * (-1) + (1 - done) * next_position
        # that the agent can move on a circle,
        # i.e. access low positions from high positions
        next_state = TreeState(
            done=done, depth=next_depth, position=next_position, seed_offset=state.seed_offset
        )
        return next_state, reward

    def _step(self, rng, state: TreeState, action):
        new_state, reward = self.__step(state, action)
        return new_state, Transition(
            self.get_observation(new_state), reward, new_state.done, new_state.depth
        )

    def _step_mdp(self, state, action):
        new_state, reward = self.__step(state, action)
        reward_probabilities = jnp.zeros(self.reward_values.shape)
        reward_probabilities = reward_probabilities.at[
            jnp.array(reward - self.reward_offset, int)
        ].set(1.0)
        # reward_probabilities = reward_probabilities.at[reward.astype(int)].set(1.)

        return (
            new_state,
            Transition(self.get_observation(new_state), reward, new_state.done, new_state.depth),
            reward_probabilities,
        )

    def step_mdp(self, state: TreeState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(state, action):
            """
            Only update time and give no reward.
            """
            new_state = state

            reward_probabilities = jnp.zeros(self.reward_values.shape)
            reward_probabilities = reward_probabilities.at[-self.reward_offset].set(1.0)

            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=state.depth,
            )
            return new_state, new_transition, reward_probabilities

        return jax.lax.cond(
            state.done,
            empty_step,
            self._step_mdp,
            state,
            action,
        )

    def step(self, rng, state: TreeState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(rng, state, action):
            """
            Only update time and give no reward.
            """
            new_state = state
            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=state.depth,
            )
            return new_state, new_transition

        return jax.lax.cond(
            state.done,
            empty_step,
            self._step,
            rng,
            state,
            action,
        )

    def info(self, state):
        return {}
