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
import rlax


class QValueModule(abc.ABC):
    """
    Base class for submodules used by Contribution models (e.g. value function, feature extractor)
    """

    @abc.abstractmethod
    def __call__(self, state, observations):
        pass

    @abc.abstractmethod
    def reset(self, rng, dummy_observation):
        pass

    @abc.abstractmethod
    def update(self, rng, state, batch_sampler, logits_fn):
        pass


@struct.dataclass
class QValueState:
    params: hk.Params
    optimizer_state: optax.OptState


class QValue(QValueModule):
    def __init__(self, model, optimizer, steps, td_lambda):
        self.action_value_optimizer = optimizer
        self.action_value_steps = steps
        self.action_value_model = hk.without_apply_rng(hk.transform(model))
        self.td_lambda = td_lambda

    def __call__(self, state, observations):
        return jax.vmap(self.action_value_model.apply, in_axes=(None, 0))(
            state.params, observations
        )

    def reset(self, rng, dummy_observation):
        initial_params = self.action_value_model.init(rng, dummy_observation)
        initial_opt_state = self.action_value_optimizer.init(initial_params)
        initial_state = QValueState(params=initial_params, optimizer_state=initial_opt_state)

        return initial_state

    def update(self, rng, state, batch_sampler, logits_fn):
        target_params = state.params

        # Compute average loss and grads across batch
        def mean_batch_loss(params, batch_trajectory):
            batch_loss = jax.vmap(self.loss, in_axes=(None, None, 0))(
                params, target_params, batch_trajectory
            )
            return jnp.mean(batch_loss)

        def update_once(carry, rng_t):
            params, optimizer_state = carry

            # Sample a batch_trajectory from the replay buffer
            batch_trajectory = batch_sampler(rng_t)
            batch_loss, grads = jax.value_and_grad(mean_batch_loss)(params, batch_trajectory)

            # Update params
            params_update, next_optimizer_state = self.action_value_optimizer.update(
                grads, optimizer_state, params
            )
            next_params = optax.apply_updates(params, params_update)

            metrics = {"loss": batch_loss, "gradnorm": optax.global_norm(grads)}

            return [next_params, next_optimizer_state], metrics

        carry, metrics = jax.lax.scan(
            f=update_once,
            init=[state.params, state.optimizer_state],
            xs=jax.random.split(rng, self.action_value_steps),
        )

        # Only select the last element from metrics.
        metrics_summary = {k + "_start": metrics[k][0] for k in metrics}
        metrics_summary.update({k + "_end": metrics[k][-1] for k in metrics})

        params, optimizer_state = carry
        next_contribution_state = QValueState(params=params, optimizer_state=optimizer_state)

        return next_contribution_state, metrics_summary

    def loss(self, params, target_params, trajectory):
        q_values_current = jax.vmap(self.action_value_model.apply, in_axes=(None, 0))(
            params, trajectory.observations
        )
        q_values_target = jax.vmap(self.action_value_model.apply, in_axes=(None, 0))(
            target_params, trajectory.next_observations
        )

        td_error = rlax.q_lambda(
            q_tm1=q_values_current.squeeze(),
            a_tm1=trajectory.actions,
            r_t=trajectory.rewards,
            discount_t=1 - trajectory.dones[1:],
            q_t=q_values_target.squeeze(),
            lambda_=jnp.ones_like(trajectory.rewards) * self.td_lambda,
        )

        loss = rlax.l2_loss(td_error)

        return jnp.sum(loss * (1 - trajectory.dones[:-1])) / jnp.sum(1 - trajectory.dones[:-1])


@struct.dataclass
class QValueGTState:
    policy_prob: jnp.ndarray


class QValueGT(QValueModule):
    @staticmethod
    def get_qvalue(mdp, policy_prob):
        s_a_scurr = jax.vmap(mdp.get_state_action_successor, in_axes=(0, None, None), out_axes=-1)(
            jnp.eye(mdp.num_state), policy_prob, mdp.max_trial
        )
        avr_rewards = jax.vmap(lambda a, b: a @ b)(policy_prob, mdp.mdp_reward)
        future_action_values = jnp.einsum("ijk,k->ij", s_a_scurr, avr_rewards)
        action_values = mdp.mdp_reward + future_action_values
        return action_values

    def __init__(self, mdp):
        self.mdp = mdp

    def __call__(self, state, observations):
        action_values = QValueGT.get_qvalue(self.mdp, state.policy_prob)
        states = jax.vmap(self.mdp.observation_to_state)(observations)
        return states @ action_values

    def reset(self, rng, dummy_observation):
        return QValueGTState(
            policy_prob=jnp.ones((self.mdp.num_state, self.mdp.num_actions)) / self.mdp.num_actions
        )

    def update(self, rng, state, batch_sampler, logits_fn):
        policy_logits = logits_fn(self.mdp.mdp_observation, False)
        return QValueGTState(policy_prob=jax.nn.softmax(policy_logits)), dict()
