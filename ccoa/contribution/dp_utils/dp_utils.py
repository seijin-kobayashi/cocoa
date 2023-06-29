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
import numpy as np


def get_idx(curr_obs, observations):
    match = jax.vmap(lambda x, y: jnp.all(jnp.abs(x - y) < 0.00001), in_axes=(0, None))(
        jnp.stack(observations), curr_obs
    )
    return jax.lax.cond(
        jnp.sum(match) > 0,
        lambda _: jnp.argmax(match, axis=0),
        lambda _: jnp.ones((), dtype=int) * (-1),
        None,
    )


def dfs(env, observe, step_mdp, curr_state, observations, transition, reward, reward_probs):
    curr_obs = observe(curr_state)
    curr_obs_idx = get_idx(curr_obs, observations) if len(observations) > 0 else -1

    if curr_obs_idx > 0:
        return
    curr_obs_idx = len(observations)
    observations.append(curr_obs)
    reward.append([0] * env.num_actions)
    transition.append([[0] * env.num_actions for _ in range(curr_obs_idx)])
    reward_probs.append([[] for _ in range(env.num_actions)])

    for i in range(curr_obs_idx + 1):
        transition[i].append([0] * env.num_actions)

    if curr_state.done:
        print("done reached!")
        zero_idx = np.where(env.reward_values == 0)[0][0]
        transition[curr_obs_idx][curr_obs_idx] = [0] * env.num_actions
        reward_probs[curr_obs_idx] = [
            [0] * zero_idx + [1.0] + [0.0] * (len(env.reward_values) - 1 - zero_idx)
            for _ in range(env.num_actions)
        ]

        return

    for action in range(env.num_actions):
        next_state, next_transition, next_reward_probs = step_mdp(curr_state, action)
        dfs(env, observe, step_mdp, next_state, observations, transition, reward, reward_probs)

        next_obs_idx = get_idx(next_transition.observation, observations)
        assert next_obs_idx >= 0
        reward[curr_obs_idx][action] = next_transition.reward
        reward_probs[curr_obs_idx][action] = list(next_reward_probs)
        transition[curr_obs_idx][next_obs_idx][action] = 1
    return


def get_mdp(env):
    print("Building MDP")
    observations = []
    init_state, _ = env.reset(None)
    transition = []
    reward = []  # average rewards
    reward_probs = []  # probabilities for each reward
    observe = jax.jit(env.get_observation)
    step_mdp = jax.jit(env.step_mdp)
    dfs(env, observe, step_mdp, init_state, observations, transition, reward, reward_probs)
    print("Done")
    return (
        jnp.stack(observations),
        jnp.transpose(jnp.array(transition), (0, 2, 1)),
        jnp.array(reward),
        jnp.array(reward_probs),
    )


class MDP:
    def __init__(self, env):
        self.max_trial = env.length
        observation, transition, reward, reward_probs = get_mdp(env)
        self.mdp_observation = observation
        self.mdp_transition = transition
        self.mdp_reward = reward  # average rewards of state-action pairs
        self.mdp_reward_probs = reward_probs
        self.mdp_reward_values = jnp.array(env.reward_values)
        self.num_state = observation.shape[0]
        self.num_actions = env.num_actions
        self.init_state = jax.nn.one_hot(0, self.num_state)

    def observation_to_state(self, observation):
        return jax.nn.one_hot(get_idx(observation, self.mdp_observation), self.num_state)

    def hindsight_object_to_hindsight_state(self, hindsight_object, all_hindsight_objects):
        """
        Gets a hindsight object as input, and return the unique one hot encoding associated to it.
        """
        all_hindsight_objects = jnp.reshape(
            all_hindsight_objects, (-1, *all_hindsight_objects.shape[3:])
        )

        hs_idx = get_idx(hindsight_object, all_hindsight_objects)
        return jax.nn.one_hot(
            hs_idx, self.num_state * self.num_actions * len(self.mdp_reward_values)
        )

    def get_state_action_successor(self, curr_state, policy_prob, horizon):
        """
        curr_state: one hot encoding
        policy: 2d tensor of (s, a) logits

        returns: sum_k=t+1 P(s_k=curr_state|s_t,a_t)
        """
        contribution = jnp.zeros((self.num_state, self.num_actions))

        batch_inner_prod = jax.vmap(lambda a, b: a @ b)

        def _get_contribution(curr_contrib, timestep):
            next_contrib = self.mdp_transition @ (
                curr_state + batch_inner_prod(curr_contrib, policy_prob)
            )
            return next_contrib, None

        contribution, _ = jax.lax.scan(_get_contribution, contribution, jnp.arange(horizon))

        return contribution
