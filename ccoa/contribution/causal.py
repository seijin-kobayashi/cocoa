"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any

from functools import partial
import jax
import jax.numpy as jnp
from flax import struct

from .base import Contribution


@struct.dataclass
class CausalContributionState:
    feature: Any
    counterfactual: Any


class CausalContribution(Contribution):
    def __init__(
        self,
        num_actions,
        obs_shape,
        return_type,
        counterfactual_module,
        feature_module,
    ):
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.return_type = return_type
        self.counterfactual_module = counterfactual_module
        self.feature_module = feature_module

    def __call__(self, state, trajectory):
        # This has shape (T, T, A), encoding the contribution of (s_t, s_t', a_t).
        # The diagonal is zero
        batched_coeffs = self.get_contribution_coeff(state=state, trajectory=trajectory)

        # NOTE: It is assumed the trajectory always ends.
        rewards_masked = trajectory.rewards * (1 - trajectory.dones[:-1])
        rewards_masked = jnp.expand_dims(rewards_masked, -1)
        advantages = jnp.sum(batched_coeffs * rewards_masked, axis=1)

        # The contribution s_t->s_t is not modeled, and the immediate reward not used.
        # We instead use the policy gradient contribution for the immediate reward.
        advantages = advantages + jnp.where(
            jax.nn.one_hot(trajectory.actions, self.num_actions),
            jnp.expand_dims(trajectory.rewards, -1) / jnp.exp(trajectory.logits),
            0,
        )

        return advantages

    def get_contribution_coeff(self, state, trajectory):
        """
        Computes the contribution coefficients (s_t, s_t', s_a) using the trajectory.
        The returned tensor has dimension (T,T,A) where T is the length of the trajectory.
        The diagonal (contribution s_t -> s_t) is zero when using the greg trick.
        """
        num_timesteps = trajectory.observations.shape[0]

        hindsight_objects = jax.vmap(self.get_hindsight_object, in_axes=(None, 0, 0, 0))(
            state, trajectory.observations, trajectory.actions, trajectory.rewards
        )

        batched_coeffs_norm = self.get_coefficient(
            state, trajectory.observations, trajectory.logits, hindsight_objects
        )

        # Do not return the contribution of s_t on (s_t, a_t).
        mask_tril = jnp.expand_dims(1 - jnp.tri(num_timesteps, num_timesteps, k=0), axis=-1)

        if self.return_type == "advantage":
            return (batched_coeffs_norm - 1.0) * mask_tril

        return batched_coeffs_norm * mask_tril

    def reset(self, rng):
        rng_features, rng_counterfactual = jax.random.split(rng, 2)
        dummy_observation = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        dummy_action = 0
        dummy_reward = 0.0
        dummy_logits = jnp.zeros((self.num_actions,))

        features_state = self.feature_module.reset(rng_features, dummy_observation)

        dummy_hindsight_object = self.feature_module(
            features_state, dummy_observation, dummy_action, dummy_reward
        )

        # Initialise the hindsight model
        counterfactual_state = self.counterfactual_module.reset(
            rng_counterfactual, dummy_observation, dummy_hindsight_object, dummy_logits
        )

        return CausalContributionState(feature=features_state, counterfactual=counterfactual_state)

    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        rng_features, rng_counterfactual = jax.random.split(rng, 2)
        metrics = dict()

        # Update feature network
        state_features, metrics_features = self.feature_module.update(
            rng_features, state.feature, offline_batch_sampler
        )
        metrics.update({k + "_features": metrics_features[k] for k in metrics_features})

        state_counterfactual, metrics_counterfactual = self.counterfactual_module.update(
            rng_counterfactual,
            state.counterfactual,
            batch_sampler,
            partial(self.get_hindsight_object, state),
            logits_fn,
        )
        metrics.update(
            {k + "_counterfactual": metrics_counterfactual[k] for k in metrics_counterfactual}
        )

        state = CausalContributionState(
            feature=state_features,
            counterfactual=state_counterfactual,
        )

        return state, metrics

    def get_coefficient(self, state, observations, logits, hindsight_objects):
        return self.counterfactual_module(
            state.counterfactual, observations, logits, hindsight_objects
        )

    def get_hindsight_object(self, state, observation, action, reward):
        """
        Returns the credit assignment feature associated with (s, a, r).
        NOTE: This is not batched, i.e. assumes a single observation, action and reward
        """
        return self.feature_module(state.feature, observation, action, reward)

    def expected_advantage(self, state, mdp, policy_prob):
        all_hindsight_objects = jax.vmap(
            jax.vmap(
                jax.vmap(self.get_hindsight_object, in_axes=(None, None, None, 0)),
                in_axes=(None, None, 0, None),
            ),
            in_axes=(None, 0, None, None),
        )(state, mdp.mdp_observation, jnp.arange(mdp.num_actions), mdp.mdp_reward_values)
        all_hindsight_state = jax.vmap(
            jax.vmap(
                jax.vmap(mdp.hindsight_object_to_hindsight_state, in_axes=(0, None)),
                in_axes=(0, None),
            ),
            in_axes=(0, None),
        )(all_hindsight_objects, all_hindsight_objects)

        # P(s_k|s_t, a_t)
        s_a_scurr = jax.vmap(mdp.get_state_action_successor, in_axes=(0, None, None), out_axes=-1)(
            jnp.eye(mdp.num_state), policy_prob, mdp.max_trial
        )

        # s is s_t, k is s_k, b is a_k, r is r_k, u is u_k
        # P(k|s)
        s_scurr = jnp.einsum("sbk,sb->sk", s_a_scurr, policy_prob)

        # P(u, r | s) = SUM_{k, b} P(k|s) pi(b|k) p(r| k, b) p(u|k,b,r)
        s_rcurr_ucurr = jnp.einsum(
            "sk,kb,kbr,kbru->sru", s_scurr, policy_prob, mdp.mdp_reward_probs, all_hindsight_state
        )

        contribution = self.get_coefficient(
            state,
            mdp.mdp_observation,
            jnp.log(policy_prob),
            jnp.reshape(all_hindsight_objects, (-1, *all_hindsight_objects.shape[3:])),
        )

        if self.return_type == "advantage":
            contribution = contribution - 1.0
            avr_rewards = jax.vmap(lambda a, b: a @ b)(policy_prob, mdp.mdp_reward)
            immediate_advantage = mdp.mdp_reward - jnp.expand_dims(avr_rewards, -1)
        else:
            immediate_advantage = mdp.mdp_reward

        # SUM_{r,u} P(u, r | s_t) w(s,a,u) r
        future_advantage = jnp.einsum(
            "sua,sru,r->sa", contribution, s_rcurr_ucurr, mdp.mdp_reward_values
        )

        return immediate_advantage + future_advantage
