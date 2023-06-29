"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax
from flax import struct

from .base import Agent


@struct.dataclass
class PolicyGradientState:
    params: hk.Params
    optimizer_state: optax.OptState


class PolicyGradient(Agent):
    def __init__(
        self,
        num_actions,
        obs_shape,
        policy,
        optimizer,
        loss_type,
        num_sample,
        entropy_reg,
        max_grad_norm,
        epsilon,
        epsilon_at_eval,
    ):
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.policy = policy
        self.num_sample = num_sample
        self.entropy_reg = entropy_reg

        if max_grad_norm is not None and max_grad_norm > 0:
            # Optional gradient clipping
            self.optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)
        else:
            self.optimizer = optimizer

        self.init_policy, forward_policy = hk.without_apply_rng(hk.transform(self.policy))

        if epsilon == 0:
            self.forward_policy_train = forward_policy
        else:
            self.forward_policy_train = lambda p, x: jnp.log(
                (1 - epsilon) * jax.nn.softmax(forward_policy(p, x)) + epsilon / self.num_actions
            )

        if epsilon_at_eval:
            self.forward_policy_eval = self.forward_policy_train
        else:
            self.forward_policy_eval = forward_policy

        if loss_type == "sum":
            self._loss = self.pg_loss_sum
        elif loss_type == "sample":
            self._loss = self.pg_loss_sample
        else:
            raise ValueError('Loss type "{}" not defined.'.format(loss_type))

    def act(self, rng, agent_state, observation, eval=False):
        action_logits = self.get_logits(agent_state, observation, eval)

        action = jax.random.categorical(rng, action_logits).squeeze()

        return action.astype(int), action_logits

    def get_logits(self, agent_state, observation, eval=False):
        action_logits = jax.lax.cond(
            eval,
            lambda x: self.forward_policy_eval(
                agent_state.params, jnp.expand_dims(observation, axis=0)
            ).squeeze(),
            lambda x: self.forward_policy_train(
                agent_state.params, jnp.expand_dims(observation, axis=0)
            ).squeeze(),
            None,
        )
        return jax.nn.log_softmax(action_logits)

    def reset(self, rng):
        dummy_observation = jnp.zeros((1, *self.obs_shape), dtype=jnp.float32)
        initial_params = self.init_policy(rng, dummy_observation)
        initial_opt_state = self.optimizer.init(initial_params)
        initial_state = PolicyGradientState(
            params=initial_params, optimizer_state=initial_opt_state
        )

        return initial_state

    def grad(self, params, rng, trajectory, return_contribution, entropy_reg):
        def batch_loss(params, rng, trajectory, return_contribution):
            batch_size = trajectory.observations.shape[0]
            rngs = jax.random.split(rng, batch_size)
            loss = jax.vmap(self._loss, in_axes=(None, 0, 0, 0, None))(
                params, rngs, trajectory, return_contribution, entropy_reg
            )

            return jnp.mean(loss)

        return jax.value_and_grad(batch_loss)(params, rng, trajectory, return_contribution)

    def update(self, rng, agent_state, trajectory, return_contribution):
        loss, grads = self.grad(
            agent_state.params, rng, trajectory, return_contribution, self.entropy_reg
        )
        params_update, next_optimizer_state = self.optimizer.update(
            grads, agent_state.optimizer_state, agent_state.params
        )
        next_params = optax.apply_updates(agent_state.params, params_update)

        next_agent_state = PolicyGradientState(
            params=next_params, optimizer_state=next_optimizer_state
        )
        metrics = {"loss": loss, "gradnorm": optax.global_norm(grads)}

        return next_agent_state, metrics

    def pg_loss_sum(self, params, rng, trajectory, return_contribution, entropy_reg):
        logits = self.forward_policy_train(params, trajectory.observations)

        mask = 1 - jnp.expand_dims(trajectory.dones[:-1], -1)
        loss_pg_terms = jax.nn.softmax(logits) * return_contribution
        loss_pg = -(loss_pg_terms * mask).sum(-1).mean()

        loss_entropy = rlax.entropy_loss(logits, 1.0 - trajectory.dones[:-1])

        return loss_pg + entropy_reg * loss_entropy

    def pg_loss_sample(self, params, rng, trajectory, return_contribution, entropy_reg):
        observations = trajectory.observations
        episode_len = observations.shape[0]
        dones = jnp.expand_dims(trajectory.dones[:-1], -1)

        logits = self.forward_policy_train(params, observations)

        @jax.vmap
        def sample_actions(rng, logit):
            actions = jax.random.categorical(rng, logit)
            actions_onehot = jax.nn.one_hot(actions, num_classes=self.num_actions, dtype=jnp.int_)

            return actions_onehot

        @jax.vmap
        def sample_pg_loss(rng_sample):
            rngs_sample = jax.random.split(rng_sample, episode_len)

            sampled_action = sample_actions(rngs_sample, logits).squeeze()
            loss_terms = jax.nn.log_softmax(logits) * (sampled_action * return_contribution)
            mask = 1 - dones

            return (loss_terms * mask).sum() / mask.sum()

        rngs = jax.random.split(rng, self.num_sample)
        loss_pg = -sample_pg_loss(rngs).mean()
        loss_entropy = rlax.entropy_loss(logits, 1.0 - trajectory.dones[:-1])

        return loss_pg + entropy_reg * loss_entropy
