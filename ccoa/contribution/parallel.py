"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict

from flax import struct
import jax

from .base import Contribution


@struct.dataclass
class ParallelContributionState:
    state: Dict


class ParallelContribution(Contribution):
    def __init__(self, contribution_dict, main_contribution_id, reset_before_update):
        self.contribution_dict = contribution_dict
        self.main_contribution_id = main_contribution_id
        self.reset_before_update = reset_before_update

    def reset(self, rng):
        state_dict = dict()
        for k, v in self.contribution_dict.items():
            state_dict[k] = self.contribution_dict[k].reset(rng)
        return ParallelContributionState(state=state_dict)

    def __call__(self, contribution_state, trajectory, key=None):
        if key is None:
            key = self.main_contribution_id
        return self.contribution_dict[key](contribution_state.state[key], trajectory)

    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        if self.reset_before_update:
            rng, rng_reset = jax.random.split(rng)
            state = self.reset(rng_reset)

        state_dict = dict()
        metric = dict()
        for k, v in self.contribution_dict.items():
            state_dict[k], metric_k = self.contribution_dict[k].update(
                rng, state.state[k], batch_sampler, offline_batch_sampler, logits_fn
            )
            metric.update({_k + "_" + k: _v for (_k, _v) in metric_k.items()})
        return ParallelContributionState(state=state_dict), metric

    def expected_advantage(self, state, mdp, policy_prob, key=None):
        if key is None:
            key = self.main_contribution_id
        return self.contribution_dict[key].expected_advantage(state.state[key], mdp, policy_prob)
