"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import pickle
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from absl import flags
from flax import struct

from ccoa import utils
from ccoa.accumulator.base import Trajectory


@struct.dataclass
class ExperimentState:
    agent: Any
    buffer: Any
    contribution: Any
    episode: int
    reward: jnp.array
    reward_final: jnp.array
    rng: chex.PRNGKey
    offline_buffer: Any


@struct.dataclass
class CallbackState:
    agent: Any
    buffer: Any
    contribution: Any
    env: Any
    episode: int
    trajectory: Trajectory
    offline_buffer: Any


class Event(Enum):
    EVAL_EPISODE = 0


class Experiment:
    def __init__(
        self,
        agent,
        contribution,
        env,
        env_switched,
        env_switch_episode,
        buffer,
        offline_buffer,
        num_episodes,
        max_trials,
        batch_size,
        offline_batch_size,
        burnin_episodes,
        logger,
        logdir,
        eval_interval_episodes,
        eval_batch_size,
        log_level=0,
    ):
        self.agent = agent
        self.contribution = contribution
        self.env = env
        self.buffer = buffer
        self.env_switched = env_switched
        self.env_switch_episode = env_switch_episode
        self.offline_buffer = offline_buffer

        self.num_episodes = num_episodes
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.offline_batch_size = offline_batch_size
        self.burnin_episodes = burnin_episodes

        self.logger = logger
        self.logdir = logdir
        self.eval_interval_episodes = eval_interval_episodes
        self.eval_batch_size = eval_batch_size

        self.callbacks = defaultdict(list)
        self.log_level = log_level

        self.log_training = log_level > 2

        self.rollout_eval = jax.jit(
            jax.vmap(
                lambda x, y, z: self._episode_rollout(x, y, eval=True, episode=z),
                in_axes=(0, None, None),
            )
        )

    def reset_env(self, episode, *args):
        return jax.lax.cond(
            self.env_switch_episode > 0 and self.env_switch_episode <= episode,
            self.env_switched.reset,
            self.env.reset,
            *args
        )

    def step_env(self, episode, *args):
        return jax.lax.cond(
            self.env_switch_episode > 0 and self.env_switch_episode <= episode,
            self.env_switched.step,
            self.env.step,
            *args
        )

    def add_callback(self, onevent: Event, callback, log_level=0):
        """
        Add a callback function triggerent on the specified event.

        Args:
            onevent: the `Event` at which to trigger the callback
            callback: callback function that should take as inputs (rng, ctx, CallbackState)
                where ctx is the current context of the Experiment (self)
        """
        self.callbacks[onevent].append((callback, log_level))

    def trigger_callbacks(self, rng, onevent: Event, callback_state: CallbackState):
        for callback_loglvl in self.callbacks.get(onevent, []):
            if callback_loglvl[1] <= self.log_level:
                rng, rng_callback = jax.random.split(rng)
                yield callback_loglvl[0](rng=rng_callback, ctx=self, state=callback_state)

    def reset(self, rng, from_dir=None):
        if from_dir is not None:
            return pickle.load(open(os.path.join(from_dir, "runner_state.pkl"), "wb"))

        rng_agent, rng_buffer, rng_contribution, rng_experiment = jax.random.split(rng, 4)
        agent_state = self.agent.reset(rng_agent)
        (sample_trajectory, _), _ = self._episode_rollout(
            rng_buffer, agent_state, eval=False, episode=0
        )
        buffer_state = self.buffer.reset(sample_trajectory)
        offline_buffer_state = self.offline_buffer.reset(sample_trajectory)
        contribution_state = self.contribution.reset(rng_contribution)

        return ExperimentState(
            agent=agent_state,
            buffer=buffer_state,
            contribution=contribution_state,
            episode=0,
            reward=0.0,
            reward_final=0.0,
            rng=rng_experiment,
            offline_buffer=offline_buffer_state,
        )

    def save(self, runner_state):
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        flags.FLAGS.append_flags_into_file(os.path.join(self.logdir, "flagfile.cfg"))
        pickle.dump(runner_state, open(os.path.join(self.logdir, "runner_state.pkl"), "wb"))

    def run(self, runner_state):
        start_episode = runner_state.episode
        for episode in range(start_episode, self.num_episodes):
            runner_state, metrics = self.step(
                runner_state,
                episode,
                update_agent=episode >= self.burnin_episodes,
                update_contribution=episode >= self.burnin_episodes,
            )

            if self.log_training:
                self.log(metrics)

            if episode % self.eval_interval_episodes == 0:
                metrics_eval = self.eval(runner_state)
                self.log(metrics_eval, prefix="eval/")
                runner_state = runner_state.replace(reward=metrics_eval["reward_running_avg"],
                                                    reward_final=metrics_eval["reward_final_running_avg"])
                self.save(runner_state)

        metrics_eval = self.eval(runner_state)
        self.log(metrics_eval, prefix="eval/")
        self.save(runner_state)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def step(self, runner_state, episode, update_agent=True, update_contribution=True):
        (
            rng_rollout,
            rng_update_agent,
            rng_update_contribution,
            next_rng,
        ) = jax.random.split(runner_state.rng, 4)

        # Rollout a batch of trajectories
        rngs_rollout = jax.random.split(rng_rollout, self.batch_size)
        episode_rollout_vmap = jax.vmap(
            lambda x, y, z: self._episode_rollout(x, y, eval=False, episode=z),
            in_axes=(0, None, None),
        )
        (trajectory, _), metrics = episode_rollout_vmap(rngs_rollout, runner_state.agent, episode)

        # Update the replay buffer
        state_buffer = self.buffer.add(runner_state.buffer, trajectory)
        offline_buffer_state = self.offline_buffer.add(runner_state.offline_buffer, trajectory)

        # Give contribution module access to sampling from the replay buffer
        batch_sampler = partial(
            self.buffer.sample, state_buffer=state_buffer, batch_size=self.batch_size
        )
        offline_batch_sampler = partial(
            self.offline_buffer.sample,
            state_buffer=offline_buffer_state,
            batch_size=self.offline_batch_size,
        )

        # Give contribution module access to sampling policy logits
        logits_fn = partial(
            jax.vmap(self.agent.get_logits, in_axes=(None, 0, None)), runner_state.agent
        )

        if update_contribution:
            # Update the contribution model's state
            state_contribution, metrics_contribution = self.contribution.update(
                rng=rng_update_contribution,
                state=runner_state.contribution,
                batch_sampler=batch_sampler,
                offline_batch_sampler=offline_batch_sampler,
                logits_fn=logits_fn,
            )
            metrics.update(**utils.prepend_keys(metrics_contribution, "contribution"))
        else:
            state_contribution = runner_state.contribution

        # Update the agent's state
        if update_agent:
            return_contribution = jax.vmap(self.contribution.__call__, in_axes=(None, 0))(
                state_contribution, trajectory
            )
            state_agent, metrics_agent = self.agent.update(
                rng_update_agent, runner_state.agent, trajectory, return_contribution
            )
            metrics.update(**utils.utils.prepend_keys(metrics_agent, "agent"))
        else:
            state_agent = runner_state.agent

        metrics["reward_final"] = metrics["reward"][:, -1].mean()
        metrics["reward"] = metrics["reward"].sum(-1).mean(0)
        metrics["action_entropy"] = metrics["action_entropy"].sum() / (1 - trajectory.dones).sum()
        metrics["episode"] = runner_state.episode

        new_runner_state = ExperimentState(
            agent=state_agent,
            buffer=state_buffer,
            contribution=state_contribution,
            episode=runner_state.episode + 1,
            reward=runner_state.reward,
            reward_final=runner_state.reward_final,
            rng=next_rng,
            offline_buffer=offline_buffer_state,
        )

        return new_runner_state, metrics

    def eval(self, runner_state):
        rng_callback, rng_rollout = jax.random.split(runner_state.rng, 2)

        # Temporarily update only the contribution module, not the agent, to sync both
        runner_state, metrics = self.step(
            runner_state,
            runner_state.episode,
            update_agent=False,
            update_contribution=(runner_state.episode >= (self.burnin_episodes + 1)).item(),
        )

        # Rollout a batch of episodes in parallel
        rngs_rollout = jax.random.split(rng_rollout, self.eval_batch_size)
        (trajectory, env_states), metrics = self.rollout_eval(
            rngs_rollout, runner_state.agent, runner_state.episode
        )

        metrics["reward_final"] = metrics["reward"][:, -1].mean()
        metrics["reward"] = metrics["reward"].sum(-1).mean(0)

        metrics["reward_running_avg"] = runner_state.reward + (
            (metrics["reward"] - runner_state.reward) / (runner_state.episode + 1)
        )
        metrics["reward_final_running_avg"] = runner_state.reward_final + (
            (metrics["reward_final"] - runner_state.reward_final) / (runner_state.episode + 1)
        )

        metrics.update(
            {
                k: v.sum() / (1 - trajectory.dones).sum()
                for (k, v) in metrics.items()
                if onp.prod(v.shape) > 1
            }
        )
        metrics["episode"] = runner_state.episode

        # Trigger callbacks
        for m in self.trigger_callbacks(
            rng_callback,
            Event.EVAL_EPISODE,
            callback_state=CallbackState(
                agent=runner_state.agent,
                buffer=runner_state.buffer,
                offline_buffer=runner_state.offline_buffer,
                contribution=runner_state.contribution,
                env=env_states,
                episode=runner_state.episode,
                trajectory=trajectory,
            ),
        ):
            metrics.update(m)

        return metrics

    @partial(jax.jit, static_argnums=0)
    def _episode_rollout(self, rng, agent_state, eval, episode):
        def _episode_step(carry, t):
            """
            Episode step closing over agent_state
            """
            rng, env_state, env_emission = carry
            rng_next, rng_action, rng_env = jax.random.split(rng, 3)

            # Choose an action
            action, action_logits = self.agent.act(
                rng_action, agent_state, env_emission.observation, eval
            )

            # Act in the environment and observe next emission
            env_state, env_emission_next = self.step_env(episode, rng_env, env_state, action)

            transition = Trajectory(
                observations=env_emission.observation,
                next_observations=env_emission_next.observation,
                rewards=env_emission_next.reward,
                dones=env_emission_next.done,
                actions=action,
                logits=action_logits,
            )

            carry = [rng_next, env_state, env_emission_next]

            metrics = {
                "action_entropy": (-jnp.exp(action_logits) * action_logits).sum(-1)
                * (1 - env_emission.done),
                "reward": env_emission_next.reward * (1 - env_emission.done),
            }

            return carry, (transition, metrics)

        rng_env, rng_rollout = jax.random.split(rng)

        # Randomly initalise the environment
        env_state, env_emission = self.reset_env(episode, rng_env)

        # Rollout the whole episode
        carry, (trajectory, metrics) = jax.lax.scan(
            _episode_step,
            [rng_rollout, env_state, env_emission],
            jnp.arange(self.max_trials),
        )
        _, env_state, _ = carry

        # NOTE: `done` for starting state was not recorded, so it is manually added here
        trajectory = trajectory.replace(dones=jnp.concatenate((jnp.zeros(1), trajectory.dones)))

        return (trajectory, env_state), metrics

    def log(self, log_dict, prefix=""):
        def versiontuple(v):
            return tuple(map(int, (v.split("."))))

        if versiontuple(jax.__version__) >= versiontuple("0.4.1"):
            log_dict = {
                key: val if not isinstance(val, jax.Array) else onp.array(val)
                for key, val in log_dict.items()
            }

        self.logger.log({prefix + k: log_dict[k] for k in log_dict})
