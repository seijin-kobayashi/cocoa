"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ml_collections as mlc


def environment_config(num_contexts, max_parallel_contexts, num_objects, die_prob, seed):
    config = mlc.ConfigDict()
    config.num_contexts = num_contexts
    config.max_parallel_contexts = max_parallel_contexts
    config.num_objects = num_objects
    config.length = 100
    config.die_prob = die_prob
    config.name = f"interleaving_stochastic"
    config.seed = seed

    return config


def get_config(method):
    config = mlc.ConfigDict()
    num_contexts, max_parallel_contexts, num_objects, die_prob, method = method.split(";")
    config.method = method

    config.num_contexts = int(num_contexts)
    config.max_parallel_contexts = int(max_parallel_contexts)
    config.num_objects = int(num_objects)
    config.die_prob = float(die_prob)
    env_seed = 2020
    config.environment = environment_config(config.num_contexts, config.max_parallel_contexts, config.num_objects, config.die_prob, env_seed)
    config.env_id = f"interleaving_stochastic_{num_contexts}_{max_parallel_contexts}_{num_objects}_{die_prob}_{env_seed}"

    config.policy_var_first_state_only = False
    config.seed = 0
    config.batch_size = 8
    config.offline_batch_size = 720
    config.buffer_size = config.batch_size
    config.offline_buffer_size = config.offline_batch_size
    config.eval_batch_size = 512
    config.eval_interval_episodes = 100
    config.num_episodes = 10000
    config.pg_loss = "sum"
    config.pg_num_sample = 0
    config.epsilon_exploration = 0.05

    config.interleaved_episodes = 0

    config.hidden_dim_agent = (64, 64)
    config.optimizer_agent = "adamw"
    config.epsilon_exploration_eval = False
    config.compute_mdp = True

    config.hindsight_model_type = "hypernet"

    config.method = method
    config.entropy_reg = 0.01
    config.pg_norm = -1

    if method == "reinforce":
        config.contribution = "reinforce"
        config.return_contribution = "action_value"
        config.lr_agent = 0.001

    elif method == "advantage":
        config.lr_agent = 0.001
        config.lr_contrib = 0.001

        config.contribution = "reinforce"
        config.hidden_dim_value = (256,)
        config.lambda_value = 1.0
        config.optimizer_value = "adamw"
        config.return_contribution = "advantage"
        config.steps_value = 1

    elif method == "qnet":
        config.entropy_reg = 0.003

        config.lr_agent = 0.0003
        config.lr_contrib = 0.003
        config.contribution = "qnet"
        config.hidden_dim_qnet = (256,)
        config.lambda_qnet = 0.9
        config.optimizer_qnet = "adamw"
        config.return_contribution = "action_value"
        config.steps_qnet = 1

    elif method == "causal_state":
        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "state_based"
        config.lr_agent = 0.0001
        config.lr_contrib = 0.001
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.mask_zero_reward_loss = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None

    elif method == "causal_reward":
        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "reward_based"
        config.lr_contrib = 0.001
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None
        config.mask_zero_reward_loss = True
        config.lr_agent = 0.0003

    elif method == "causal_reward_feature":
        config.burnin_episodes = 90
        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "feature"
        config.lr_contrib = 0.001
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None
        config.mask_zero_reward_loss = True
        config.lr_agent = 0.0003

        config.hidden_dim_features = 128
        config.optimizer_features = "adamw"
        config.lr_features = 0.001
        config.steps_features = 30000

        config.l1_reg_params_features = 0.003
        config.l2_reg_readout_feature = 0.0003
        config.threshold_shift = 0.05
        config.per_action_readout_feature = False
        config.feature_model = "mask"
        config.discretization = "relu"

    elif method == "traj_cv":
        config.lr_agent = 0.001
        config.lr_contrib = 0.001
        config.contribution = "traj_cv"
        config.hidden_dim_qnet = (256,)
        config.lambda_qnet = 0.9
        config.optimizer_qnet = "adamw"
        config.return_contribution = "action_value"
        config.steps_qnet = 1

    elif method == "advantage_gt":
        config.lr_agent = 0.003
        config.contribution = "reinforce_gt"
        config.return_contribution = "advantage"

    elif method == "qnet_gt":
        config.lr_agent = 0.01
        config.contribution = "qnet_gt"
        config.return_contribution = "advantage"

    elif "causal_state_gt" in method:
        config.lr_agent = 0.003
        config.contribution = "causal_gt"
        config.hindsight_loss_type = "contrastive"
        config.return_contribution = "advantage"
        config.hindsight_feature_type = "state_based"

    elif "causal_reward_gt" in method:
        config.lr_agent = 0.01
        config.contribution = "causal_gt"
        config.hindsight_loss_type = "contrastive"
        config.return_contribution = "advantage"
        config.hindsight_feature_type = "reward_based"

    elif method == "traj_cv_gt":
        config.lr_agent = 0.003
        config.contribution = "traj_cv_gt"
        config.return_contribution = "action_value"

    else:
        raise ValueError
    return config
