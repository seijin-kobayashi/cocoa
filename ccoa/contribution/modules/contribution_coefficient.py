"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from flax import struct


class ContributionCoefficientModule(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state, observations, policy_logits, hindsight_objects):
        pass

    @abc.abstractmethod
    def reset(self, rng, dummy_observation, dummy_hindsight_object, dummy_logits):
        pass

    @abc.abstractmethod
    def update(self, rng, state, batch_sampler, hindsight_object_fn, logits_fn):
        pass


@struct.dataclass
class CoefficientBaseState:
    params: hk.Params
    optim: optax.OptState


class CoefficientBase(ContributionCoefficientModule):
    def __init__(
        self,
        model,
        optimizer,
        steps,
        mask_zero_reward_loss,
        clip_contrastive,
        max_grad_norm,
    ):
        self.hindsight_model = hk.without_apply_rng(hk.transform(model))
        if max_grad_norm is not None:
            assert max_grad_norm > 0
            # Optional gradient clipping
            self.hindsight_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optimizer
            )
        else:
            self.hindsight_optimizer = optimizer
        self.hindsight_steps = steps
        self.mask_zero_reward_loss = mask_zero_reward_loss
        self.clip_contrastive = clip_contrastive

    @abc.abstractmethod
    def __call__(self, state, observations, policy_logits, hindsight_objects):
        pass

    def reset(self, rng, dummy_observation, dummy_hindsight_object, dummy_logits):
        # Initialise the hindsight model
        initial_hindsight_params = self.hindsight_model.init(
            rng, dummy_observation, dummy_hindsight_object, dummy_logits
        )

        initial_hindsight_opt_state = self.hindsight_optimizer.init(initial_hindsight_params)

        return CoefficientBaseState(
            params=initial_hindsight_params,
            optim=initial_hindsight_opt_state,
        )

    def update(self, rng, state, batch_sampler, hindsight_object_fn, logits_fn):
        def mean_batch_loss(params, observations, hindsight_objects, actions, logits, rewards):
            batch_loss = jax.vmap(self.loss, in_axes=(None, 0, 0, 0, 0, 0))(
                params, observations, hindsight_objects, actions, logits, rewards
            )
            return jnp.mean(batch_loss)

        def update_step(carry, rng_t):
            params, optimizer_state = carry

            # Sample a batch of trajectories from the replay buffer
            batch_trajectory = batch_sampler(rng_t)

            # Obtain the hindsight objects
            observations = batch_trajectory.observations
            actions = batch_trajectory.actions
            logits = batch_trajectory.logits
            rewards = batch_trajectory.rewards

            hindsight_object_fn_batched = jax.vmap(jax.vmap(hindsight_object_fn))

            hindsight_objects = hindsight_object_fn_batched(
                batch_trajectory.observations,
                batch_trajectory.actions,
                batch_trajectory.rewards,
            )

            # Compute loss
            batch_loss, grads = jax.value_and_grad(mean_batch_loss)(
                params, observations, hindsight_objects, actions, logits, rewards
            )

            # Update params
            params_update, next_optimizer_state = self.hindsight_optimizer.update(
                grads, optimizer_state, params
            )
            next_params = optax.apply_updates(params, params_update)

            metrics = {"loss": batch_loss, "gradnorm": optax.global_norm(grads)}

            return [next_params, next_optimizer_state], metrics

        rng_init, rng_scan = jax.random.split(rng)

        carry, metrics = jax.lax.scan(
            f=update_step,
            init=[state.params, state.optim],
            xs=jax.random.split(rng_scan, self.hindsight_steps),
        )

        # Only select the last element from metrics.
        metrics_summary = {k + "_start": metrics[k][0] for k in metrics}
        metrics_summary.update({k + "_end": metrics[k][-1] for k in metrics})

        (params_hindsight, optim_hindsight) = carry
        state = CoefficientBaseState(
            params=params_hindsight,
            optim=optim_hindsight,
        )
        return state, metrics_summary

    def loss(self, params, observations, hindsight_objects, actions, action_logits, rewards):
        "Loss over a single trajectory"
        batched_loss = jax.vmap(
            jax.vmap(self._loss, in_axes=(None, 0, None, 0, 0)),
            in_axes=(None, None, 0, None, None),
            out_axes=1,
        )
        loss = batched_loss(params, observations, hindsight_objects, actions, action_logits)

        causal_mask = jnp.triu(jnp.ones_like(loss), k=1)

        reward_mask = jnp.ones((rewards.shape[0]))
        if self.mask_zero_reward_loss:
            # Only consider non-zero rewards
            reward_mask = jnp.where(rewards == 0, 0, reward_mask)

        reward_mask = jnp.expand_dims(reward_mask, axis=0)
        normalizer = jnp.sum(causal_mask * reward_mask)
        normalizer = normalizer * (normalizer != 0) + (normalizer == 0)
        return jnp.sum(loss * causal_mask * reward_mask) / normalizer


class HindsightCoefficient(CoefficientBase):
    def __init__(
        self,
        model,
        optimizer,
        steps,
        hindsight_loss_type,
        mask_zero_reward_loss,
        max_grad_norm,
        modulate_with_policy,
    ):
        super().__init__(
            model,
            optimizer,
            steps,
            hindsight_loss_type,
            mask_zero_reward_loss,
            max_grad_norm,
        )
        self.modulate_with_policy = modulate_with_policy

    def __call__(self, state, observations, policy_logits, hindsight_objects):
        """compute the contribution coefficients"""

        logit_normed = jnp.expand_dims(policy_logits, axis=1)

        if not self.modulate_with_policy:
            logit_normed = 0 * logit_normed

        batched_coeffs = jax.vmap(
            jax.vmap(self.hindsight_model.apply, in_axes=(None, None, 0, None)),
            in_axes=(None, 0, None, 0),
        )(state.params, observations, hindsight_objects, policy_logits)

        batched_coeffs = jax.nn.log_softmax(logit_normed + batched_coeffs, axis=-1)
        batched_coeffs_norm = jnp.exp(batched_coeffs - jnp.expand_dims(policy_logits, axis=1))

        return batched_coeffs_norm

    def _loss(self, params, observation, hindsight_object, action, action_logits):
        logits = self.hindsight_model.apply(params, observation, hindsight_object, action_logits)

        if self.modulate_with_policy:
            logits = logits + action_logits

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, action)
        return loss


class ContrastiveCoefficient(CoefficientBase):
    def __init__(
        self,
        model,
        optimizer,
        steps,
        hindsight_loss_type,
        mask_zero_reward_loss,
        max_grad_norm,
        clip_contrastive,
    ):
        super().__init__(
            model,
            optimizer,
            steps,
            hindsight_loss_type,
            mask_zero_reward_loss,
            max_grad_norm,
        )
        self.clip_contrastive = clip_contrastive

    def __call__(self, state, observations, policy_logits, hindsight_objects):
        """compute the contribution coefficients"""

        batched_coeffs = jax.vmap(
            jax.vmap(self.hindsight_model.apply, in_axes=(None, None, 0, None)),
            in_axes=(None, 0, None, 0),
        )(state.params, observations, hindsight_objects, policy_logits)

        if self.clip_contrastive:
            lower = -jnp.expand_dims(policy_logits, axis=1) * jnp.ones_like(batched_coeffs)
            batched_coeffs_norm = jnp.exp(jnp.min(jnp.stack([batched_coeffs, lower]), axis=0))
        else:
            batched_coeffs_norm = jnp.exp(batched_coeffs)

        return batched_coeffs_norm

    def _loss(self, params, observation, hindsight_object, action, action_logits):
        discriminator_logits = self.hindsight_model.apply(
            params, observation, hindsight_object, action_logits
        )

        log_sigmoid = jax.nn.log_sigmoid(discriminator_logits)
        one_hot_action = jax.nn.one_hot(action, action_logits.shape[0])
        loss = -(
            one_hot_action * log_sigmoid
            + jnp.exp(action_logits) * (-discriminator_logits + log_sigmoid)
        ).sum()

        return loss


@struct.dataclass
class CoefficientGTState:
    policy_prob: jnp.ndarray
    hindsight_objects: jnp.ndarray


class CoefficientGT(ContributionCoefficientModule):
    def __init__(self, mdp):
        self.mdp = mdp

    @staticmethod
    def get_coefficient(mdp, policy_prob, hindsight_objects):
        s_a_scurr = jax.vmap(mdp.get_state_action_successor, in_axes=(0, None, None), out_axes=-1)(
            jnp.eye(mdp.num_state), policy_prob, mdp.max_trial
        )

        all_hindsight_state = jax.vmap(
            jax.vmap(
                jax.vmap(mdp.hindsight_object_to_hindsight_state, in_axes=(0, None)),
                in_axes=(0, None),
            ),
            in_axes=(0, None),
        )(hindsight_objects, hindsight_objects)

        # k: s_k, b: a_k, r: r_k, s: s_t, a: a_t, u: u_k,
        s_a_ucurr = jnp.einsum(
            "sak,kb,kbr,kbru->sua",
            s_a_scurr,
            policy_prob,
            mdp.mdp_reward_probs,
            all_hindsight_state,
        )
        s_ucurr = jnp.expand_dims(jnp.einsum("sua,sa->su", s_a_ucurr, policy_prob), axis=-1)

        coeff = jnp.where(
            s_ucurr == 0,
            1,
            s_a_ucurr / s_ucurr,
        )
        return coeff

    def __call__(self, state, observations, policy_logits, hindsight_objects):
        coeff = CoefficientGT.get_coefficient(self.mdp, state.policy_prob, state.hindsight_objects)

        states = jax.vmap(self.mdp.observation_to_state)(observations)
        hindsight_states = jax.vmap(
            self.mdp.hindsight_object_to_hindsight_state, in_axes=(0, None)
        )(hindsight_objects, state.hindsight_objects)
        return jnp.einsum("ts,sua,ku->tka", states, coeff, hindsight_states)

    def reset(self, rng, dummy_observation, dummy_hindsight_object, dummy_logits):
        return CoefficientGTState(
            policy_prob=jnp.ones((self.mdp.num_state, self.mdp.num_actions)) / self.mdp.num_actions,
            hindsight_objects=jnp.zeros(
                (
                    self.mdp.num_state,
                    self.mdp.num_actions,
                    self.mdp.mdp_reward_values.shape[0],
                    *dummy_hindsight_object.shape,
                )
            ),
        )

    def update(self, rng, state, batch_sampler, hindsight_object_fn, logits_fn):
        policy_logits = logits_fn(self.mdp.mdp_observation, False)

        hindsight_objects = jax.vmap(
            jax.vmap(
                jax.vmap(hindsight_object_fn, in_axes=(None, None, 0)), in_axes=(None, 0, None)
            ),
            in_axes=(0, None, None),
        )(self.mdp.mdp_observation, jnp.arange(self.mdp.num_actions), self.mdp.mdp_reward_values)

        return (
            CoefficientGTState(
                policy_prob=jax.nn.softmax(policy_logits), hindsight_objects=hindsight_objects
            ),
            dict(),
        )
