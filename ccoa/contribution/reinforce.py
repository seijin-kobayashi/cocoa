"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any


import jax
import jax.numpy as jnp
import rlax
from flax import struct

from .base import Contribution
from .modules.qvalue import QValueGT


@struct.dataclass
class ReinforceContributionState:
    value: Any


class ReinforceContribution(Contribution):
    def __init__(self, num_actions, obs_shape, return_type, value_module):
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.return_type = return_type
        self.value_module = value_module

    def __call__(self, state, trajectory):
        if self.return_type == "action_value":
            cumulative_return = rlax.discounted_returns(
                r_t=trajectory.rewards,
                discount_t=1 - trajectory.dones[1:],
                v_t=jnp.zeros((len(trajectory.observations))),
            )
            return_contribution = jnp.expand_dims(cumulative_return, axis=-1)
        elif self.return_type == "advantage":
            value = self.get_value(state, trajectory.observations)
            cumulative_return = rlax.discounted_returns(
                r_t=trajectory.rewards,
                discount_t=1 - trajectory.dones[1:],
                v_t=value,
            )
            return_contribution = jnp.expand_dims(cumulative_return - value, axis=-1)
        else:
            raise ValueError

        traj_actions = jax.nn.one_hot(trajectory.actions, num_classes=self.num_actions)

        return jnp.where(traj_actions, return_contribution / jax.nn.softmax(trajectory.logits), 0)

    def reset(self, rng):
        rng_value = rng

        if self.return_type == "advantage":
            dummy_observation = jnp.zeros(self.obs_shape, dtype=jnp.float32)
            state_value = self.value_module.reset(rng_value, dummy_observation)
        else:
            state_value = dict()

        return ReinforceContributionState(value=state_value)

    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        rng_value = rng
        metrics = dict()

        # Update value network
        if self.return_type == "advantage":
            state_value, metrics_value = self.value_module.update(
                rng_value, state.value, batch_sampler, logits_fn
            )
            metrics.update({k + "_value": metrics_value[k] for k in metrics_value})
        else:
            state_value = state.value

        return ReinforceContributionState(value=state_value), metrics

    def get_value(self, state, observations):
        return self.value_module(state.value, observations)

    def expected_advantage(self, state, mdp, policy_prob):
        action_values = QValueGT.get_qvalue(mdp, policy_prob)

        if self.return_type == "advantage":
            value = self.get_value(state, mdp.mdp_observation)
            return action_values - jnp.expand_dims(value, -1)
        return action_values
