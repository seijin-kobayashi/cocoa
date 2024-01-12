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
class TrajCVContributionState:
    qvalue: Any


class TrajCVContribution(Contribution):
    def __init__(self, num_actions, obs_shape, return_type, qvalue_module):
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.return_type = return_type
        self.qvalue_module = qvalue_module

    def __call__(self, state, trajectory):
        cumulative_return = rlax.discounted_returns(
            r_t=trajectory.rewards,
            discount_t=1 - trajectory.dones[1:],
            v_t=jnp.zeros((len(trajectory.observations))),
        )
        returns = jnp.expand_dims(cumulative_return, axis=-1)

        q_values = self.get_action_value(state, trajectory.observations)
        traj_actions = jax.nn.one_hot(trajectory.actions, num_classes=self.num_actions)
        q_values_traj_actions = (q_values * traj_actions).sum(axis=-1)
        values = (q_values * jax.nn.softmax(trajectory.logits)).sum(axis=-1)
        cumulative_q_values = rlax.discounted_returns(
            r_t=q_values_traj_actions,
            discount_t = 1-trajectory.dones[1:],
            v_t = jnp.zeros((len(trajectory.observations))),
        )
        cumulative_q_values = jnp.expand_dims(cumulative_q_values, axis=-1)
        cumulative_values = rlax.discounted_returns(
            r_t=values,
            discount_t=1 - trajectory.dones[1:],
            v_t=jnp.zeros((len(trajectory.observations))),
        )
        cumulative_values = jnp.expand_dims(cumulative_values, axis=-1)
        cumulative_values_delayed = jnp.concatenate([cumulative_values[1:], jnp.zeros((1,cumulative_values.shape[-1]))])

        return_contribution = jnp.where(traj_actions, (returns - cumulative_q_values + cumulative_values_delayed) /
                                        jax.nn.softmax(trajectory.logits),0) + q_values
        return return_contribution

    def reset(self, rng):
        dummy_observation = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        state_qvalue = self.qvalue_module.reset(rng, dummy_observation)

        return TrajCVContributionState(qvalue=state_qvalue)

    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        metrics = dict()

        # Update qvalue network
        state_qvalue, metrics_qvalue = self.qvalue_module.update(
            rng, state.qvalue, batch_sampler, logits_fn
        )
        metrics.update({k + "_qvalue": metrics_qvalue[k] for k in metrics_qvalue})

        return TrajCVContributionState(qvalue=state_qvalue), metrics

    def get_action_value(self, state, observations):
        return self.qvalue_module(state.qvalue, observations)

    def expected_advantage(self, state, mdp, policy_prob):
        action_values = QValueGT.get_qvalue(mdp, policy_prob)
        if self.return_type == "advantage":
            value = (action_values * policy_prob).sum(axis=-1)
            return action_values - jnp.expand_dims(value, -1)
        return action_values
