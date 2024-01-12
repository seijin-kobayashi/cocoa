"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .base import Environment, Transition


class ACTIONS(Enum):
    PICK_KEY = 0
    PICK_LEFT = 1
    PICK_RIGHT = 2
    OPEN_DOOR = 3


class THINGS(Enum):
    EMPTY = 0
    APPLE_LEFT = 1
    APPLE_RIGHT = 2
    DOOR = 3
    KEY = 4
    TREASURE = 5


@struct.dataclass
class ConveyorTreasureState:
    done: bool
    grid: jnp.ndarray
    total_distractors: int
    position: int
    timestep: int


class ConveyorTreasure(Environment):
    """Conveyor  environment."""

    def __init__(
        self,
        length,
        reward_distractor,
        reward_treasure,
        random_distractors,
        num_keys,
        seed,
        distractor_prob=2.0 / 3.0,
        reward_distractor_logits=None,
        reward_treasure_logits=None,
        treasure_at_door=False,
        ignore_door=False,
    ):
        super().__init__()

        if not isinstance(reward_distractor, list):
            reward_distractor = [reward_distractor]
        self.reward_distractor = np.array(reward_distractor)

        if reward_distractor_logits is None:
            reward_distractor_logits = np.zeros((len(reward_distractor)))
        self.reward_distractor_logits = np.array(reward_distractor_logits)

        if not isinstance(reward_treasure, list):
            reward_treasure = [reward_treasure]
        self.reward_treasure = np.array(reward_treasure)

        if reward_treasure_logits is None:
            reward_treasure_logits = np.zeros((len(reward_treasure)))
        self.reward_treasure_logits = np.array(reward_treasure_logits)

        reward_values = {0.0} | set(reward_distractor) | set(reward_treasure)

        self._reward_values = np.array(list(reward_values))

        self.distractor_to_reward_values = np.concatenate(
            [np.where(self._reward_values == d_r)[0] for d_r in self.reward_distractor]
        )
        self.treasure_to_reward_values = np.concatenate(
            [np.where(self._reward_values == t_r)[0] for t_r in self.reward_treasure]
        )
        self.zero_to_reward_values = np.where(self._reward_values == 0.0)[0]

        self.length = length
        self.seed = seed
        self.num_keys = num_keys
        self.key_pos = np.array([int(i * (length - 2) / num_keys) for i in range(num_keys)])
        self.grid = np.zeros((length,))
        self.distractor_prob = distractor_prob
        self.random_distractors = random_distractors
        self.treasure_at_door = treasure_at_door
        self.ignore_door = ignore_door

        if not random_distractors:
            # Place apples at fixed, random postions along the path
            rng_np = np.random.default_rng(seed)
            self.grid[1:-2] = rng_np.choice(
                a=[0, 1, 2],
                size=(self.length - 3,),
                p=[1 - distractor_prob, distractor_prob / 2, distractor_prob / 2],
            )

            # Place keys and door
            self.grid[self.key_pos] = THINGS.KEY.value
            self.grid[-2] = THINGS.DOOR.value
            self.grid[-1] = THINGS.EMPTY.value

    @property
    def reward_values(self):
        return self._reward_values

    @property
    def num_actions(self):
        return len(ACTIONS)

    @property
    def observation_shape(self):
        return (len(THINGS) + 2 + self.num_keys,)

    def get_observation(self, state: ConveyorTreasureState):
        relative_position = jnp.expand_dims(state.position / self.length, axis=0)
        key_counter = jnp.sum(jnp.equal(state.grid[self.key_pos], THINGS.EMPTY.value))
        key_counter_one_hot = jax.nn.one_hot(
            jnp.asarray(key_counter, dtype=jnp.int32), self.num_keys + 1
        )
        current_cell = state.grid[state.position]
        current_cell_one_hot = jax.nn.one_hot(current_cell, len(THINGS))

        return jnp.concatenate([relative_position, current_cell_one_hot, key_counter_one_hot])

    def reset(self, rng):
        if self.random_distractors:
            rng, rng_apples = jax.random.split(rng)
            grid = jnp.zeros((self.length,))

            # Place apples randomly along the path
            grid = grid.at[1:-2].set(
                jax.random.choice(
                    rng_apples,
                    a=np.array([0, 1, 2]),
                    shape=(self.length - 3,),
                    p=np.array(
                        [
                            1 - self.distractor_prob,
                            self.distractor_prob / 2,
                            self.distractor_prob / 2,
                        ]
                    ),
                )
            )

            # Place keys and door at their fixed positions
            grid = grid.at[self.key_pos].set(THINGS.KEY.value)
            grid = grid.at[-2].set(THINGS.DOOR.value)
            grid = grid.at[-1].set(THINGS.EMPTY.value)
        else:
            grid = jnp.asarray(self.grid)

        total_distractors = (grid == THINGS.APPLE_LEFT.value).sum() + (
            grid == THINGS.APPLE_RIGHT.value
        ).sum()
        init_state = ConveyorTreasureState(
            done=False, grid=grid, total_distractors=total_distractors, position=0, timestep=0
        )
        init_transition = Transition(
            observation=self.get_observation(init_state), reward=0.0, done=False, timestep=0
        )
        return init_state, init_transition

    def __step(self, state: ConveyorTreasureState, action):
        """Compute reward and new environment state"""

        current_cell = state.grid[state.position]

        # Compute the current reward
        get_distractor = jnp.logical_or(
            jnp.logical_and(
                jnp.equal(current_cell, THINGS.APPLE_LEFT.value),  # an apple to the left
                jnp.equal(action, ACTIONS.PICK_LEFT.value),  # was picked by the agent
            ),
            jnp.logical_and(
                jnp.equal(current_cell, THINGS.APPLE_RIGHT.value),  # an apple to the right
                jnp.equal(action, ACTIONS.PICK_RIGHT.value),  # was picked by the agent
            ),
        )
        # Check if all keys have been picked up
        has_all_keys = jnp.all(jnp.equal(state.grid[self.key_pos], THINGS.EMPTY.value))

        is_at_door_with_keys = jnp.logical_and(
            has_all_keys,  # keys have been picked up
            jnp.equal(current_cell, THINGS.DOOR.value),  # agent currently at door
        )
        open_door = jnp.logical_and(
            is_at_door_with_keys,
            jnp.equal(action, ACTIONS.OPEN_DOOR.value),  # agent opens the door
        )

        if self.ignore_door:
            open_door = is_at_door_with_keys
        if self.treasure_at_door:
            get_treasure = open_door
        else:
            get_treasure = jnp.equal(current_cell, THINGS.TREASURE.value)

        # Compute the next environment state
        remove_key = jnp.logical_and(
            jnp.equal(current_cell, THINGS.KEY.value),  # agent is at key
            jnp.equal(action, ACTIONS.PICK_KEY.value),  # and picks it up
        )

        remove_apple = get_distractor  # the agent picked up an apple

        remove_thing = jnp.logical_or(remove_key, remove_apple)
        grid = state.grid.at[state.position].set(
            remove_thing * THINGS.EMPTY.value + (1 - remove_thing) * current_cell
        )
        position = state.position + 1
        grid = jax.lax.cond(
            position == self.length - 1,
            lambda g: g.at[position].set(
                open_door * THINGS.TREASURE.value + (1 - open_door) * THINGS.EMPTY.value
            ),
            lambda g: g,
            grid,
        )
        done = position >= self.length
        new_state = ConveyorTreasureState(
            done, grid, state.total_distractors, position, state.timestep + 1
        )

        return new_state, (get_distractor, get_treasure), done, position

    def _step_mdp(self, state, action):
        new_state, (get_distractor, get_treasure), done, position = self.__step(state, action)

        reward_distractor = self.reward_distractor @ jax.nn.softmax(self.reward_distractor_logits)
        reward_treasure = self.reward_treasure @ jax.nn.softmax(self.reward_treasure_logits)

        reward = get_distractor * reward_distractor + get_treasure * reward_treasure
        reward_probabilities = jnp.zeros(self.reward_values.shape)
        for idx, prob in zip(
            self.distractor_to_reward_values,
            get_distractor * jax.nn.softmax(self.reward_distractor_logits),
        ):
            reward_probabilities = reward_probabilities.at[idx].set(
                reward_probabilities[idx] + prob
            )
        for idx, prob in zip(
            self.treasure_to_reward_values,
            get_treasure * jax.nn.softmax(self.reward_treasure_logits),
        ):
            reward_probabilities = reward_probabilities.at[idx].set(
                reward_probabilities[idx] + prob
            )
        reward_probabilities = reward_probabilities.at[self.zero_to_reward_values].set(
            reward_probabilities[self.zero_to_reward_values]
            + 1
            - jnp.logical_or(get_distractor, get_treasure)
        )

        return (
            new_state,
            Transition(self.get_observation(new_state), reward, done, position),
            reward_probabilities,
        )

    def step_mdp(self, state: ConveyorTreasureState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(state: ConveyorTreasureState, action):
            """
            Only update time and give no reward.
            """
            new_timestep = state.timestep + 1
            new_state = state.replace(timestep=new_timestep)

            reward_probabilities = jnp.zeros(self.reward_values.shape)
            reward_probabilities = reward_probabilities.at[self.zero_to_reward_values].set(1.0)

            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=new_timestep,
            )
            return new_state, new_transition, reward_probabilities

        return jax.lax.cond(
            state.done,
            empty_step,
            self._step_mdp,
            state,
            action,
        )

    def _step(self, rng, state: ConveyorTreasureState, action):
        """Compute reward and new environment state"""
        new_state, (get_distractor, get_treasure), done, position = self.__step(state, action)
        rng_d, rng_t = jax.random.split(rng)
        reward_distractor = self.reward_distractor @ jax.nn.one_hot(
            jax.random.categorical(rng_d, self.reward_distractor_logits),
            len(self.reward_distractor_logits),
        )
        reward_treasure = self.reward_treasure @ jax.nn.one_hot(
            jax.random.categorical(rng_t, self.reward_treasure_logits),
            len(self.reward_treasure_logits),
        )

        reward = get_distractor * reward_distractor + get_treasure * reward_treasure

        return new_state, Transition(self.get_observation(new_state), reward, done, position)

    def step(self, rng, state: ConveyorTreasureState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(rng, state: ConveyorTreasureState, action):
            """
            Only update time and give no reward.
            """
            new_timestep = state.timestep + 1
            new_state = state.replace(timestep=new_timestep)
            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=new_timestep,
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
        grid = state.grid
        missed_distractors = (grid == THINGS.APPLE_LEFT.value).sum() + (
            grid == THINGS.APPLE_RIGHT.value
        ).sum()
        picked_distractors = state.total_distractors - missed_distractors
        picked_treasure = (grid == THINGS.TREASURE.value).sum()
        return {
            "fraction_distractor_rewards": (picked_distractors / state.total_distractors),
            "percentage_treasure_rewards": picked_treasure,
        }
