"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import copy
import os
import random
from functools import partial
from pathlib import Path

import jax
import ml_collections as mlc
import optax
from absl import app, flags
from ml_collections import config_flags

import wandb
from callback import get_policy_gradient_analyses_callback, env_info_callback, get_task_disentangling_callback
from ccoa import agents, envs
from ccoa.accumulator import ReplayBuffer
from ccoa.contribution import MDP, causal, parallel, qnet, reinforce, trajcv
from ccoa.contribution.modules import hindsight_object, value, qvalue, contribution_coefficient
from ccoa.experiment import Event, Experiment
from ccoa.networks.models import (
    get_policy_model,
    get_value_model,
    get_qvalue_model,
    get_hindsight_model,
    get_feature_model,
)
from ccoa.envs import Tree

FLAGS = flags.FLAGS

# Import ml_collections flags defined in configs/
config_flags.DEFINE_config_file(
    name="config",
    default="configs/default.py",
    help_string="Training configuration.",
)

# Expose jax flags, allows for example to disable jit with --jax_disable_jit
jax.config.parse_flags_with_absl()

# Additional abseil flags
flags.DEFINE_bool("checkpoint", False, "Whether to save the checkpoint of trained models.")
flags.DEFINE_bool("synchronize", True, "Synchronize logs  to wandb.")
flags.DEFINE_integer("log_level", 2, "Logging level.")
flags.DEFINE_string("logdir", "logs", "Directory where logs are saved.")
flags.DEFINE_string("wandb_entity", "", "Entity for WANDB logging")
flags.DEFINE_string("wandb_project", "", "Project for WANDB logging")


def load_from_cache_or_compute(file_name, fct):
    import glob
    import pickle

    if len(glob.glob(file_name)) > 0:
        return pickle.load(open(file_name, "rb"))
    else:
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        result = fct()
        pickle.dump(result, open(file_name, "wb"))
        return result


def get_contribution(config, env, mdp):
    if config.contribution == "reinforce" or config.contribution == "reinforce_gt":
        if config.contribution == "reinforce_gt":
            value_module = value.ValueGT(
                mdp=mdp,
            )
        elif config.return_contribution == "advantage":
            value_module = value.ValueFunction(
                model=get_value_model(config, env),
                optimizer=getattr(optax, config.optimizer_value)(config.lr_contrib),
                steps=config.steps_value,
                td_lambda=config.lambda_value,
            )
        else:
            value_module = None

        contribution = reinforce.ReinforceContribution(
            num_actions=env.num_actions,
            obs_shape=env.observation_shape,
            return_type=config.return_contribution,
            value_module=value_module if config.return_contribution == "advantage" else None,
        )
    elif config.contribution == "qnet" or config.contribution == "qnet_gt":
        if config.contribution == "qnet_gt":
            qvalue_module = qvalue.QValueGT(
                mdp=mdp,
            )
        else:
            qvalue_module = qvalue.QValue(
                model=get_qvalue_model(config, env),
                optimizer=getattr(optax, config.optimizer_qnet)(config.lr_contrib),
                steps=config.steps_qnet,
                td_lambda=config.lambda_qnet,
            )
        contribution = qnet.QCriticContribution(
            num_actions=env.num_actions,
            obs_shape=env.observation_shape,
            return_type=config.return_contribution,
            qvalue_module=qvalue_module,
        )
    elif config.contribution == 'traj_cv' or config.contribution == 'traj_cv_gt':
        if config.contribution == 'traj_cv_gt':
            qvalue_module = qvalue.QValueGT(
                mdp=mdp,
            )
        else:
            qvalue_module = qvalue.QValue(
                model=get_qvalue_model(config, env),
                optimizer=getattr(optax, config.optimizer_qnet)(config.lr_contrib),
                steps=config.steps_qnet,
                td_lambda=config.lambda_qnet,
            )
        contribution = trajcv.TrajCVContribution(
            num_actions=env.num_actions,
            obs_shape=env.observation_shape,
            return_type=config.return_contribution,
            qvalue_module=qvalue_module,
        )
    elif config.contribution == "causal" or config.contribution == "causal_gt":
        if config.hindsight_feature_type == "feature":
            feature_module = hindsight_object.FeatureObject(
                num_actions=env.num_actions,
                model=get_feature_model(config, env),
                optimizer=getattr(optax, config.optimizer_features)(
                    config.lr_features, weight_decay=0
                ),
                steps=config.steps_features,
                reward_values=env.reward_values,
                per_action_readout = config.get("per_action_readout_feature", True),
                l1_reg_params=config.get("l1_reg_params_features", 0.0),
                l2_reg_readout=config.get("l2_reg_readout_feature", 0.0),
            )
        elif config.hindsight_feature_type == "state_based":
            feature_module = hindsight_object.StateObject()
        elif config.hindsight_feature_type == "reward_based":
            feature_module = hindsight_object.RewardObject(
                reward_values=env.reward_values,
            )
        elif config.hindsight_feature_type == "tree_grouping":
            assert isinstance(env, Tree)
            feature_module = hindsight_object.TreeGroupingObject(
                reward_values=env.reward_values,
                reward_modulo=env.reward_modulo,
                rewarding_outcome_nb_groups=config.rewarding_outcome_nb_groups,
                reward_offset=env.reward_offset,
                compute_state_idx=env.compute_state_idx,
                large_prime=env.large_prime,
            )
        else:
            assert False

        if config.contribution == "causal_gt":
            counterfactual_module = contribution_coefficient.CoefficientGT(
                mdp=mdp,
            )
        elif config.hindsight_loss_type == "contrastive":
            counterfactual_module = contribution_coefficient.ContrastiveCoefficient(
                model=get_hindsight_model(config, env),
                optimizer=getattr(optax, config.optimizer_hindsight)(config.lr_contrib),
                steps=config.steps_hindsight,
                mask_zero_reward_loss=config.get("mask_zero_reward_loss", False),
                max_grad_norm=config.get("hindsight_max_grad_norm", None),
                clip_contrastive=config.get("clip_contrastive", False),
            )
        elif config.hindsight_loss_type == "hindsight":
            counterfactual_module = contribution_coefficient.HindsightCoefficient(
                model=get_hindsight_model(config, env),
                optimizer=getattr(optax, config.optimizer_hindsight)(config.lr_contrib),
                steps=config.steps_hindsight,
                mask_zero_reward_loss=config.get("mask_zero_reward_loss", False),
                max_grad_norm=config.get("hindsight_max_grad_norm", None),
                modulate_with_policy=config.get("policy_modulation", False),
            )
        else:
            assert False

        contribution = causal.CausalContribution(
            num_actions=env.num_actions,
            obs_shape=env.observation_shape,
            return_type=config.return_contribution,
            counterfactual_module=counterfactual_module,
            feature_module=feature_module,
        )

    elif config.contribution == "parallel":
        contribution_dict = dict()
        for k in config.parallel_keys.split(","):
            contribution_dict[k] = get_contribution(getattr(config, k), env, mdp)

        contribution = parallel.ParallelContribution(
            contribution_dict, config.parallel_main_key, config.parallel_reset_before_update
        )
    else:
        raise ValueError('Contribution module "{}" undefined'.format(config.contribution))

    return contribution


def run(config, logger, logdir, log_level):
    config = mlc.ConfigDict(config)

    switch_env = config.get("env_switch_episode", 0) > 0
    env = envs.create(**config.environment)

    mdp = None
    if config.get("compute_mdp", True):
        mdp = load_from_cache_or_compute(
            os.path.join("cache", "mdp_" + config.env_id + ".pkl"),
            partial(MDP, env),
        )
    if switch_env:
        env_bis_config = copy.deepcopy(config.environment)
        env_bis_config["reward_treasure"] = [-r for r in env_bis_config["reward_treasure"]]
        env_bis = envs.create(**env_bis_config)
        mdp_bis = load_from_cache_or_compute(
            os.path.join("cache", "mdp_bis_" + config.env_id + ".pkl"),
            partial(MDP, env_bis),
        )
    else:
        env_bis = env
        mdp_bis = None

    policy_model = get_policy_model(config, env, mdp)

    agent = agents.PolicyGradient(
        num_actions=env.num_actions,
        obs_shape=env.observation_shape,
        policy=policy_model,
        optimizer=getattr(optax, config.optimizer_agent)(config.lr_agent),
        loss_type=config.pg_loss,
        num_sample=config.pg_num_sample,
        entropy_reg=config.entropy_reg,
        max_grad_norm=config.get("pg_norm", None),
        epsilon=config.get("epsilon_exploration", 0),
        epsilon_at_eval=config.get("epsilon_exploration_eval", False),
    )

    buffer = ReplayBuffer(config.buffer_size)
    offline_buffer = ReplayBuffer(config.offline_buffer_size)
    contribution = get_contribution(config, env, mdp)

    runner = Experiment(
        agent=agent,
        contribution=contribution,
        env=env,
        env_switched=env_bis,
        env_switch_episode=config.get("env_switch_episode", 0),
        buffer=buffer,
        offline_buffer=offline_buffer,
        num_episodes=config.num_episodes,
        max_trials=config.environment.length,
        batch_size=config.batch_size,
        offline_batch_size=config.offline_batch_size,
        burnin_episodes=config.get("burnin_episodes", 0),
        logger=logger,
        logdir=logdir,
        eval_interval_episodes=config.eval_interval_episodes,
        eval_batch_size=config.eval_batch_size,
        log_level=log_level,
    )
    if config.seed is None:
        config.seed = random.randint(0, 99999)
    runner_state = runner.reset(jax.random.PRNGKey(config.seed))

    runner.add_callback(Event.EVAL_EPISODE, env_info_callback, log_level=1)

    if config.environment.name == "interleaving_stochastic":
        runner.add_callback(Event.EVAL_EPISODE, get_task_disentangling_callback(env), log_level=3)

    if switch_env:
        runner.add_callback(
            Event.EVAL_EPISODE,
            get_policy_gradient_analyses_callback(
                mdp,
                first_state_only=config.get("policy_var_first_state_only", False),
                prefix="first",
            ),
            log_level=2,
        )
        runner.add_callback(
            Event.EVAL_EPISODE,
            get_policy_gradient_analyses_callback(
                mdp_bis,
                first_state_only=config.get("policy_var_first_state_only", False),
                prefix="second",
            ),
            log_level=2,
        )
    else:
        runner.add_callback(
            Event.EVAL_EPISODE,
            get_policy_gradient_analyses_callback(
                mdp, first_state_only=config.get("policy_var_first_state_only", False), prefix=""
            ),
            log_level=2,
        )

    runner.run(runner_state)


def main(argv):
    # Get config from flags
    config = flags.FLAGS.config

    # Setup logger
    if not FLAGS.synchronize:
        os.environ["WANDB_DISABLED"] = "true"
    wandb.init(
        config=config,
        tags=config.get("tags", None),
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
    )

    run(config, logger=wandb, logdir=FLAGS.logdir, log_level=FLAGS.log_level)


if __name__ == "__main__":
    app.run(main)
