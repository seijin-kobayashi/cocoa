"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import wandb

from ccoa import utils, contribution
from ccoa.contribution.modules.qvalue import QValueGT
from ccoa.contribution.modules.value import ValueGT
from ccoa.experiment import CallbackState


def get_policy_gradient_analyses_callback(mdp, first_state_only, prefix):
    if mdp is None:
        return lambda x, y, z: dict()

    def get_policy_gradient_mean(agent, agent_state, policy_prob, advantage):
        policy_transition = jax.vmap(lambda a, b: a @ b)(policy_prob, mdp.mdp_transition)
        batch_inner_prod = jax.vmap(lambda a, b: a @ b)

        def get_loss(params):
            def get_summed_loss(curr_state, timestep):
                zero_advantage = first_state_only * (timestep > 0)
                _policy_prob = jax.nn.softmax(
                    agent.forward_policy_train(params, mdp.mdp_observation)
                )
                curr_loss = curr_state @ batch_inner_prod(_policy_prob, advantage * (1 - zero_advantage))
                next_state = curr_state @ policy_transition
                return next_state, curr_loss

            carry, timestep_loss = jax.lax.scan(
                get_summed_loss, mdp.init_state, jnp.arange(mdp.max_trial)
            )
            return -timestep_loss.mean()

        return jax.value_and_grad(get_loss)(agent_state.params)[1]

    @partial(jax.jit, static_argnums=1)
    def callback_policy_grad_var(rng, ctx, state: CallbackState):
        """
        Compute the variance of the policy gradient within a batch averaged over parameters.
        """
        (rng_grad,) = jax.random.split(rng, 1)
        policy_prob = jax.nn.softmax(
            ctx.agent.forward_policy_train(state.agent.params, mdp.mdp_observation)
        )
        gt_advantage = QValueGT.get_qvalue(mdp, policy_prob) - jnp.expand_dims(
            ValueGT.get_value(mdp, policy_prob), -1
        )
        gt_grad = get_policy_gradient_mean(ctx.agent, state.agent, policy_prob, gt_advantage)
        trajectory = state.trajectory
        batch_size = trajectory.observations.shape[0]

        fn_dict = dict()
        if isinstance(ctx.contribution, contribution.parallel.ParallelContribution):
            for k in ctx.contribution.contribution_dict.keys():
                fn_dict[k] = (
                    partial(ctx.contribution.expected_advantage, key=k),
                    partial(ctx.contribution.__call__, key=k),
                )
        else:
            fn_dict[""] = (ctx.contribution.expected_advantage, ctx.contribution.__call__)

        all_metric = dict()
        for key, (expected_advantage, call) in fn_dict.items():
            expected_advantage = expected_advantage(state.contribution, mdp, policy_prob)

            expected_grad = get_policy_gradient_mean(
                ctx.agent, state.agent, policy_prob, expected_advantage
            )

            # Get the current batch of trajectories and compute corresponding return contributions
            return_contribution = jax.vmap(call, in_axes=(None, 0))(state.contribution, trajectory)
            if first_state_only:
                return_contribution = return_contribution.at[:, 1:].set(0)

            # Compute the agent's parameter grad for each trajectory in the batch individually
            trajectory_expanded = jtu.tree_map(partial(jnp.expand_dims, axis=1), trajectory)
            return_contribution_expanded = jnp.expand_dims(return_contribution, axis=1)
            rngs_grad = jax.random.split(rng_grad, batch_size)
            batched_agent_grad = jax.vmap(ctx.agent.grad, in_axes=(None, 0, 0, 0, None))
            _, grads = batched_agent_grad(
                state.agent.params, rngs_grad, trajectory_expanded, return_contribution_expanded, 0
            )

            def cosine_similarity(a, b):
                c = a @ b
                return jnp.where(c == 0, 0, c / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))

            def get_metrics(flat_gt_grad, flat_expected_grad, flat_grads):
                bias_cosine_sim = cosine_similarity(flat_expected_grad, flat_gt_grad)

                sample_cosine_sim = cosine_similarity(jnp.mean(flat_grads, 0), flat_gt_grad)

                metric = {
                    **{
                        "advantage_gt_{}".format(a): gt_advantage[0, a]
                        for a in range(mdp.num_actions)
                    },
                    **{
                        "advantage_expected_{}".format(a): expected_advantage[0, a]
                        for a in range(mdp.num_actions)
                    },
                    "policy_grad_var": jnp.mean((flat_grads - flat_expected_grad) ** 2)
                    / jnp.mean(flat_gt_grad**2),
                    "policy_grad_var_dB": 10
                    / jnp.log(10)
                    * (
                        -jnp.log(jnp.mean(flat_gt_grad**2))
                        + jnp.log(jnp.mean((flat_grads - flat_expected_grad) ** 2))
                    ),
                    "policy_grad_bias": jnp.mean((flat_gt_grad - flat_expected_grad) ** 2)
                    / jnp.mean(flat_gt_grad**2),
                    "policy_grad_bias_dB": 20
                    / jnp.log(10)
                    * (
                        -jnp.log(jnp.linalg.norm(flat_gt_grad))
                        + jnp.log(jnp.linalg.norm(flat_gt_grad - flat_expected_grad))
                    ),
                    "policy_grad_cos_sim_bias": bias_cosine_sim,
                    "policy_grad_snr": (
                        jnp.linalg.norm(flat_gt_grad) ** 2
                        / (jnp.mean(jnp.linalg.norm(flat_grads - flat_gt_grad, axis=1) ** 2))
                    ),
                    "policy_grad_snr_dB": 10
                    / jnp.log(10)
                    * (
                        jnp.log(jnp.linalg.norm(flat_gt_grad) ** 2)
                        - jnp.log(jnp.mean(jnp.linalg.norm(flat_grads - flat_gt_grad, axis=1) ** 2))
                    ),
                    "policy_grad_normalized_snr_dB": 10
                    / jnp.log(10)
                    * (
                        - jnp.log(jnp.mean(jnp.linalg.norm(flat_grads/jnp.linalg.norm(flat_expected_grad, keepdims=True) - flat_gt_grad/jnp.linalg.norm(flat_gt_grad, keepdims=True), axis=1) ** 2))
                    ),
                    "policy_grad_var_sample": jnp.mean(jnp.var(flat_grads, axis=0))
                    / jnp.mean(flat_gt_grad**2),
                    "policy_grad_bias_sample": jnp.mean(
                        (jnp.mean(flat_grads, axis=0) - flat_gt_grad) ** 2
                    )
                    / jnp.mean(flat_gt_grad**2),
                    "policy_grad_cos_sim_bias_sample": sample_cosine_sim,
                }
                return metric

            batched_flatcat = jax.vmap(utils.flatcat)
            gt_grad = utils.flatcat(gt_grad)
            expected_grad = utils.flatcat(expected_grad)
            grads = batched_flatcat(grads)

            metric = get_metrics(gt_grad, expected_grad, grads)
            all_metric.update({prefix + _k + "_" + key: _v for (_k, _v) in metric.items()})

        return all_metric

    return callback_policy_grad_var


def env_info_callback(rng, ctx, state: CallbackState):
    info_dict = jax.vmap(ctx.env.info)(state.env)
    log_dict = {k: v.mean() for (k, v) in info_dict.items()}
    return log_dict

def get_parallel_causal(rng, ctx, state, get_metric_fn):
    causal_dict = dict()
    if isinstance(ctx.contribution, contribution.parallel.ParallelContribution):
        for k in ctx.contribution.contribution_dict.keys():
            if isinstance(ctx.contribution.contribution_dict[k], contribution.causal.CausalContribution):
                causal_dict[k] = (ctx.contribution.contribution_dict[k], state.contribution.state[k])
    else:
        causal_dict[""] = (ctx.contribution, state.contribution)

    all_metric = dict()
    for key, (ctx_contribution, state_contribution) in causal_dict.items():
        all_metric.update({_k + "_" + key: _v for (_k, _v) in get_metric_fn(state, ctx_contribution, state_contribution).items()})

    return all_metric


def get_task_disentangling_callback(env):
    if True:
        @partial(jax.jit, static_argnums=1)
        def get_task_disentangling_matrix_single_model(state, ctx_contribution, state_contribution):
            trajectories = state.trajectory
            contribution_coeffs = jax.vmap(ctx_contribution.get_contribution_coeff, in_axes=(None,0))(state_contribution, trajectories)
            contribution_coeffs = jnp.einsum("BTtA,BTA -> BTt", jnp.abs(contribution_coeffs), jnp.exp(trajectories.logits))

            def get_mask(trajectory, ctx_query, ctx_ans):
                traj_len = trajectory.observations.shape[0]
                ctx_query = jax.nn.one_hot(ctx_query, env.num_contexts).reshape([-1])
                ctx_ans = jax.nn.one_hot(ctx_ans, env.num_contexts).reshape([-1])
                context_mask_query = jnp.expand_dims(jnp.all(trajectory.observations[:,:env.num_contexts] == jnp.expand_dims(ctx_query,axis=0), axis=-1), axis=1)*jnp.ones((traj_len, traj_len))
                context_mask_ans = jnp.expand_dims(jnp.all(trajectory.observations[:,:env.num_contexts] == jnp.expand_dims(ctx_ans,axis=0), axis=-1), axis=0)*jnp.ones((traj_len, traj_len))
                causal_mask = 1 - jnp.tri(traj_len, traj_len, k=0)
                query_mask = jnp.expand_dims(trajectory.observations[:, env.num_contexts] == 0, axis=1)*jnp.ones((traj_len, traj_len))
                answer_mask = jnp.expand_dims(trajectory.observations[:, env.num_contexts] == 1, axis=0)*jnp.ones((traj_len, traj_len))
                reward_mask = jnp.expand_dims(trajectory.rewards != 0, axis=0)*jnp.ones((traj_len, traj_len))
                return context_mask_query*context_mask_ans*causal_mask*query_mask*answer_mask*reward_mask

            def get_element_matrix(ctx_query, ctx_ans):
                mask = jax.vmap(get_mask, in_axes=(0,None,None))(trajectories, ctx_query, ctx_ans)
                selected_coeff = mask*contribution_coeffs
                return selected_coeff.sum()/mask.sum()

            return {"disentangling_matrix": (jax.vmap(jax.vmap(get_element_matrix, in_axes=(None, 0)),in_axes=(0,None))(
                jnp.arange(start=0, stop=env.num_contexts), jnp.arange(start=0, stop=env.num_contexts)
            ))}

    def callback_task_disentangling(rng, ctx, state: CallbackState):
        tensor_dict = get_parallel_causal(rng, ctx, state, get_task_disentangling_matrix_single_model)
        return {k: wandb.Image(np.array(v/jnp.sum(v,axis=-1,keepdims=True))) for k,v in tensor_dict.items()}

    return callback_task_disentangling







