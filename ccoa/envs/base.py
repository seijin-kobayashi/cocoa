"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
from typing import Union
from flax import struct
import jax.numpy as jnp
import numpy as onp

ndarray = Union[onp.ndarray, jnp.ndarray]

# Adapted from https://github.com/google/brax/blob/main/brax/envs/env.py


@struct.dataclass
class Transition:
    observation: ndarray
    reward: float
    done: bool
    timestep: int


class Environment(abc.ABC):
    @abc.abstractmethod
    def reset(self, rng):
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def get_observation(self, state):
        """Computes the observation of an environment at the given state."""

    @abc.abstractmethod
    def step(self, rng, state, action):
        """Run one timestep of the environment's dynamics. Returns the Transition and the Environment state."""

    @abc.abstractmethod
    def step_mdp(self, state, action):
        """Run one timestep of the environment's dynamics. Similar to step, but returns the expected reward as well as the probability for each reward values."""

    @property
    def reward_values(self):
        """List of possible rewards returned by the MDP"""

    @property
    def num_actions(self):
        """The number of possible actions"""

    @property
    def observation_shape(self):
        """The shape of the observation array"""

    @abc.abstractmethod
    def info(self, state):
        """Returns a dictionary of env info useful for logging"""
