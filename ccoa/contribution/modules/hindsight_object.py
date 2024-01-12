"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import struct

from ccoa.utils.utils import flatcat


class HindsightObjectModule(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state, observation, action, reward):
        pass

    @abc.abstractmethod
    def reset(self, rng, dummy_input):
        pass

    @abc.abstractmethod
    def update(self, rng, state, batch_sampler):
        pass


class StateObject(HindsightObjectModule):
    def __call__(self, state, observation, action, reward):
        return observation

    def reset(self, rng, dummy_input):
        return dict()

    def update(self, rng, state, batch_sampler):
        return dict(), dict()


NUM_DECIMAL = 4

class RewardObject(HindsightObjectModule):
    def __init__(
        self,
        reward_values,
    ):
        self.reward_values = reward_values

    def __call__(self, state, observation, action, reward):
        index = jnp.nonzero(jnp.round(reward - self.reward_values, NUM_DECIMAL)==0, size=1)[0]
        return jax.nn.one_hot(index, len(self.reward_values)).squeeze()

    def reset(self, rng, dummy_input):
        return dict()

    def update(self, rng, state, batch_sampler):
        return dict(), dict()


@struct.dataclass
class FeatureObjectState:
    backbone: hk.Params
    optim: optax.OptState
    readout: hk.Params
    trained: bool


class FeatureObject(HindsightObjectModule):
    """
    Args:
        backbone: hk.Module that maps (observation, action) -> feature without batching
    """

    def __init__(
        self,
        num_actions,
        model,
        optimizer,
        steps,
        reward_values,
        per_action_readout,
        l1_reg_params,
        l2_reg_readout,
        train_once=True
    ) -> None:
        self.num_actions = num_actions
        self.backbone = hk.without_apply_rng(hk.transform(model))
        self.readout = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1)(x)))
        self.optimizer = optimizer
        self.steps = steps
        self.reward_values = reward_values
        self.per_action_readout = per_action_readout
        self.l1_reg_params = l1_reg_params
        self.l2_reg_readout = l2_reg_readout
        self.train_once = train_once

    def __call__(self, state, observation, action, reward):
        return (self.backbone.apply(state.backbone, observation, action) > 0) * 1.0

    def reset(self, rng, dummy_observation):
        rng_backbone, rng_readout = jax.random.split(rng)
        params_backbone = self.backbone.init(rng_backbone, dummy_observation, 0)
        dummy_features = self.backbone.apply(params_backbone, dummy_observation, 0)

        if self.per_action_readout:
            # One readout per action
            rngs_readout = jax.random.split(rng_readout, self.num_actions)
            params_readout = jax.vmap(self.readout.init, in_axes=(0, None))(
                rngs_readout, dummy_features
            )
        else:
            params_readout = self.readout.init(
                rng_readout, dummy_features
            )

        optim = self.optimizer.init([params_backbone, params_readout])

        return FeatureObjectState(
            backbone=params_backbone, optim=optim, readout=params_readout, trained=False
        )

    def update(self, rng, state, batch_sampler):
        def batch_loss(rng, params, observations, rewards, actions):
            rngs = jax.random.split(rng, observations.shape[0] * observations.shape[1]).reshape(
                observations.shape[0], observations.shape[1], -1
            )
            # vmap over both batch_size and steps in trajectory
            loss_fn_batched = jax.vmap(
                jax.vmap(self.loss_fn, in_axes=(0, None, None, 0, 0, 0)),
                in_axes=(0, None, None, 0, 0, 0),
            )

            loss, metrics = loss_fn_batched(
                rngs, params[0], params[1], observations, rewards, actions
            )
            metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), metrics)

            for i, value in enumerate(self.reward_values):
                metrics["loss_reward_{}".format(value)] = jnp.where(
                    metrics["num_class_{}".format(value)] > 0,
                    metrics["loss_reward_{}".format(value)] / metrics["num_class_{}".format(value)],
                    0,
                )
            loss = jnp.mean(loss, axis=(0, 1))

            return loss, metrics

        def update_step(carry, rng_t):
            params, optim = carry
            rng_loss, rng_sample = jax.random.split(rng_t)

            # Sample a batch of trajectories from the replay buffer
            batch_trajectory = batch_sampler(rng_sample)
            observations = batch_trajectory.observations
            rewards = batch_trajectory.rewards
            actions = batch_trajectory.actions

            # Compute loss
            (loss, metrics), grads = jax.value_and_grad(batch_loss, argnums=1, has_aux=True)(
                rng_loss, params, observations, rewards, actions
            )
            # Update params
            params_update, optim = self.optimizer.update(grads, optim, params)
            next_params = optax.apply_updates(params, params_update)

            params_backbone, params_readout = next_params[0], next_params[1]

            # We call this l1, but we can just use L2 since the parameters are assumed to be squared
            if self.l1_reg_params > 0:
                params_backbone = jax.tree_util.tree_map(
                    lambda p: p - self.l1_reg_params * p, params_backbone
                )

            if self.l2_reg_readout > 0:
                params_readout = jax.tree_util.tree_map(
                    lambda p: p - self.l2_reg_readout * p, params_readout
                )

            # NOTE: disabling metric logging to make the jax.lax.cond work.
            metrics = {
                "loss": loss,
                "gradnorm": optax.global_norm(grads),
                "readout_norm": jnp.sqrt(jnp.sum(flatcat(params_readout) ** 2)),
                "backbone_norm": jnp.sqrt(jnp.sum(flatcat(params_backbone) ** 2)),
                **metrics,
            }

            return [[params_backbone, params_readout], optim], dict()

        carry, metrics = jax.lax.cond(
            state.trained,
            lambda: ([[state.backbone, state.readout], state.optim], dict()),
            lambda: jax.lax.scan(
                f=update_step,
                init=[[state.backbone, state.readout], state.optim],
                xs=jax.random.split(rng, self.steps),
            ),
        )

        # Only select the last element from metrics.
        metrics_summary = dict()
        metrics_summary.update({k + "_end": metrics[k][-1] for k in metrics})

        params, optim = carry
        state = FeatureObjectState(backbone=params[0], optim=optim, readout=params[1], trained=self.train_once)

        return state, metrics_summary

    def loss_fn(
        self,
        rng: chex.PRNGKey,
        params_backbone: hk.Params,
        params_readout: hk.Params,
        observation: jnp.ndarray,
        reward: jnp.ndarray,
        action: jnp.ndarray,
    ):
        """
        Loss is defined for a single example (no batching over batch_size or num_steps).
        """
        metrics = dict()
        features = self.backbone.apply(params_backbone, observation, action)

        # Select readout params according to action
        if self.per_action_readout:
            params_readout_action = jtu.tree_map(lambda p: p[action], params_readout)
        else:
            params_readout_action = params_readout
        prediction = self.readout.apply(params_readout_action, features)

        log_dict = {}
        loss = optax.l2_loss(prediction.squeeze(), reward)
        metrics.update({"loss": loss})

        label = jnp.nonzero(reward == jnp.array(self.reward_values), size=1)[0].squeeze()
        for i, value in enumerate(self.reward_values):
            num_class = ((label == i) * 1.0).sum()
            log_dict["loss_reward_{}".format(value)] = loss * (label == i)
            log_dict["num_class_{}".format(value)] = num_class
        metrics.update(log_dict)

        return loss, metrics


class TreeGroupingObject(HindsightObjectModule):
    def __init__(
        self,
        reward_values,
        reward_modulo,
        rewarding_outcome_nb_groups,
        reward_offset,
        compute_state_idx,
        large_prime,
    ):
        self.reward_values = reward_values
        self.reward_modulo = reward_modulo
        self.rewarding_outcome_nb_groups = rewarding_outcome_nb_groups
        self.reward_offset = reward_offset
        self.compute_state_idx = compute_state_idx
        self.large_prime = large_prime

    def __call__(self, state, observation, action, reward):
        x = observation
        assert len(x) == 3

        depth = x[0]
        position = x[1]
        seed_offset = x[2]
        state_idx = jnp.array(self.compute_state_idx(depth, position), int)

        # Quick hack to handle the terminal state in the mdp, which has a position of -1, and should have a
        #  reward of zero
        return (1 + jnp.array([0, position]).min()) * (
            jnp.array(
                (state_idx + action * self.large_prime + seed_offset)
                % (self.reward_modulo * self.rewarding_outcome_nb_groups),
                int,
            )
            + self.reward_offset
        ) + jnp.zeros_like(observation)

    def reset(self, rng, dummy_input):
        return dict()

    def update(self, rng, state, batch_sampler):
        return dict(), dict()
