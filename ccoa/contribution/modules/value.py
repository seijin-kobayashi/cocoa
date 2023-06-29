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
import rlax
from flax import struct


class ValueModule(abc.ABC):
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
class ValueFunctionState:
    params: hk.Params
    optim: optax.OptState


class ValueFunction(ValueModule):
    def __init__(self, model, optimizer, steps, td_lambda) -> None:
        self.model = hk.without_apply_rng(hk.transform(model))
        self.optimizer = optimizer
        self.steps = steps
        self.td_lambda = td_lambda

    def __call__(self, state, observations):
        return jax.vmap(self.model.apply, in_axes=(None, 0))(state.params, observations).squeeze(-1)

    def reset(self, rng, dummy_observation):
        initial_params = self.model.init(rng, dummy_observation)
        initial_opt_state = self.optimizer.init(initial_params)

        return ValueFunctionState(params=initial_params, optim=initial_opt_state)

    def update(self, rng, state, batch_sampler, logits_fn):
        target_params = state.params

        def mean_batch_loss(params, batch_trajectory):
            """
            Compute average loss and grads across batch
            """
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
            params_update, next_optimizer_state = self.optimizer.update(
                grads, optimizer_state, params
            )
            next_params = optax.apply_updates(params, params_update)

            metrics = {"loss": batch_loss, "gradnorm": optax.global_norm(grads)}

            return [next_params, next_optimizer_state], metrics

        carry, metrics = jax.lax.scan(
            f=update_once,
            init=[state.params, state.optim],
            xs=jax.random.split(rng, self.steps),
        )

        # Only select the last element from metrics.
        metrics_summary = {k + "_start": metrics[k][0] for k in metrics}
        metrics_summary.update({k + "_end": metrics[k][-1] for k in metrics})

        params, optimizer_state = carry
        next_state = ValueFunctionState(params=params, optim=optimizer_state)

        return next_state, metrics_summary

    def loss(self, params, target_params, trajectory):
        value_model_traj = jax.vmap(self.model.apply, in_axes=(None, 0))
        values = value_model_traj(params, trajectory.observations).squeeze(axis=1)
        target_values = value_model_traj(target_params, trajectory.next_observations).squeeze(
            axis=1
        )
        # TD-lambda temporal difference error
        td_errors = rlax.td_lambda(
            v_tm1=values,
            r_t=trajectory.rewards,
            discount_t=1 - trajectory.dones[1:],
            v_t=target_values,
            lambda_=jnp.ones_like(trajectory.rewards) * self.td_lambda,
        )
        return jnp.sum(td_errors**2 * (1 - trajectory.dones[:-1])) / jnp.sum(
            1 - trajectory.dones[:-1]
        )


@struct.dataclass
class ValueGTState:
    policy_prob: jnp.ndarray


class ValueGT(ValueModule):
    @staticmethod
    def get_value(mdp, policy_prob):
        s_a_scurr = jax.vmap(mdp.get_state_action_successor, in_axes=(0, None, None), out_axes=-1)(
            jnp.eye(mdp.num_state), policy_prob, mdp.max_trial
        )
        avr_rewards = jax.vmap(lambda a, b: a @ b)(policy_prob, mdp.mdp_reward)
        future_values = jnp.einsum("ij,ijk,k->i", policy_prob, s_a_scurr, avr_rewards)
        values = avr_rewards + future_values
        return values

    def __init__(self, mdp) -> None:
        self.mdp = mdp

    def __call__(self, state, observations):
        values = ValueGT.get_value(self.mdp, state.policy_prob)
        states = jax.vmap(self.mdp.observation_to_state)(observations)
        return states @ values

    def reset(self, rng, dummy_observation):
        policy_prob = jnp.ones((self.mdp.num_state, self.mdp.num_actions)) / self.mdp.num_actions
        return ValueGTState(policy_prob=policy_prob)

    def update(self, rng, state, batch_sampler, logits_fn):
        policy_logits = logits_fn(self.mdp.mdp_observation, False)
        return ValueGTState(policy_prob=jax.nn.softmax(policy_logits)), dict()
