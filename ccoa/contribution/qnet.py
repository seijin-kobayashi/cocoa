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
from flax import struct

from .base import Contribution


@struct.dataclass
class QCriticContributionState:
    qvalue: Any


class QCriticContribution(Contribution):
    def __init__(self, num_actions, obs_shape, return_type, qvalue_module):
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.return_type = return_type
        self.qvalue_module = qvalue_module

    def __call__(self, state, trajectory):
        done = trajectory.dones
        return_contribution = self._get_contribution(
            state, trajectory.observations, jax.nn.softmax(trajectory.logits)
        )
        mask = jnp.expand_dims(1 - done[:-1], axis=-1)
        return return_contribution * mask

    def _get_contribution(self, state, observations, policy_prob):
        return_contribution = self.get_action_value(state, observations)

        if self.return_type == "advantage":
            value = (return_contribution * policy_prob).sum(-1)
            return_contribution = return_contribution - jnp.expand_dims(value, axis=-1)

        return return_contribution

    def reset(self, rng):
        dummy_observation = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        state_qvalue = self.qvalue_module.reset(rng, dummy_observation)

        return QCriticContributionState(qvalue=state_qvalue)

    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        metrics = dict()

        # Update qvalue network
        state_qvalue, metrics_qvalue = self.qvalue_module.update(
            rng, state.qvalue, batch_sampler, logits_fn
        )
        metrics.update({k + "_qvalue": metrics_qvalue[k] for k in metrics_qvalue})

        return QCriticContributionState(qvalue=state_qvalue), metrics

    def get_action_value(self, state, observations):
        return self.qvalue_module(state.qvalue, observations)

    def expected_advantage(self, state, mdp, policy_prob):
        return self._get_contribution(state, mdp.mdp_observation, policy_prob)
