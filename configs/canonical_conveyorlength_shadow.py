"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ml_collections as mlc


def contribution_config(method, global_config):
    config = mlc.ConfigDict(type_safe=False)
    config.environment = global_config.environment

    if method == "reinforce":
        config.hindsight_model_type = "hypernet"
        config.contribution = "reinforce"
        config.return_contribution = "action_value"

    elif method == "traj_cv":
        config.hindsight_model_type = "hypernet"
        config.lr_contrib = 0.003
        config.contribution = "traj_cv"
        config.hidden_dim_qnet = (256,)
        config.lambda_qnet = 0.9
        config.optimizer_qnet = "adamw"
        config.return_contribution = "action_value"
        config.steps_qnet = 1

    elif method == "advantage":
        config.hindsight_model_type = "hypernet"
        config.lr_contrib = 0.001
        config.contribution = "reinforce"
        config.hidden_dim_value = (256,)
        config.lambda_value = 1.0
        config.optimizer_value = "adamw"
        config.return_contribution = "advantage"
        config.steps_value = 1

    elif method == "qnet":
        config.hindsight_model_type = "hypernet"
        config.lr_contrib = 0.003
        config.contribution = "qnet"
        config.hidden_dim_qnet = (256,)
        config.lambda_qnet = 0.9
        config.optimizer_qnet = "adamw"
        config.return_contribution = "action_value"
        config.steps_qnet = 1

    elif method == "causal_state":
        config.hindsight_model_type = "hypernet"
        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "state_based"
        config.lr_contrib = 0.003
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None

    elif method == "causal_reward":
        config.hindsight_model_type = "hypernet"

        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "reward_based"
        config.lr_contrib = 0.003
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None
        config.mask_zero_reward_loss = True

    elif method == "causal_reward_feature":
        config.hindsight_model_type = "hypernet"

        config.contribution = "causal"
        config.hidden_dim_hindsight = (256,)
        config.hindsight_loss_type = "hindsight"
        config.hindsight_feature_type = "feature"
        config.lr_contrib = 0.003
        config.optimizer_hindsight = "adamw"
        config.return_contribution = "advantage"
        config.steps_hindsight = 1
        config.policy_modulation = True
        config.clip_contrastive = True
        config.hindsight_max_grad_norm = None
        config.mask_zero_reward_loss = True

        config.hidden_dim_features = 128
        config.optimizer_features = "adamw"
        config.lr_features = 0.003
        config.steps_features = 20000

        config.l1_reg_params_features = 0.001
        config.l2_reg_readout_feature = 0.03
        config.threshold_shift = 0.05

    elif method == "qnet_gt":
        config.hindsight_model_type = "hypernet"
        config.contribution = "qnet_gt"
        config.return_contribution = "advantage"

    elif method == "traj_cv_gt":
        config.hindsight_model_type = "hypernet"
        config.contribution = "traj_cv_gt"
        config.return_contribution = "action_value"

    elif "causal_state_gt" in method:
        config.hindsight_model_type = "hypernet"
        config.contribution = "causal_gt"
        config.hindsight_loss_type = "contrastive"
        config.return_contribution = "advantage"
        config.hindsight_feature_type = "state_based"

    elif "causal_reward_gt" in method:
        config.hindsight_model_type = "hypernet"
        config.contribution = "causal_gt"
        config.hindsight_loss_type = "contrastive"
        config.return_contribution = "advantage"
        config.hindsight_feature_type = "reward_based"

    elif method == "advantage_gt":
        config.hindsight_model_type = "hypernet"
        config.contribution = "reinforce_gt"
        config.return_contribution = "advantage"
        config.steps_value = 1

    else:
        raise ValueError

    return config


def environment_config(env_length: int):
    """
    Variable treasure probabilty variant of `treasure_conveyor_noisy_distractor_treasure_short`
    """
    config = mlc.ConfigDict()
    config.length = env_length
    config.num_keys = 1
    config.name = "treasure_conveyor"
    config.reward_distractor = [r * (23.0 / env_length) for r in [0.1, 0.9]]
    config.reward_distractor_logits = [0.0, 0.0]
    config.reward_treasure = 0.2 * (23.0 / env_length)
    config.reward_treasure_logits = [0.0]  # deterministic reward
    config.random_distractors = False
    config.seed = 2022
    config.distractor_prob = 0.8

    return config


def get_config(env_length):
    config = mlc.ConfigDict()
    config.env_length = int(env_length)
    config.environment = environment_config(config.env_length)
    config.env_id = "canonical_final_env_len_{}".format(config.env_length)

    config.lr_agent = 0.0001
    config.epsilon_exploration = 0.05
    config.entropy_reg = 0.003
    config.pg_norm = 0.003

    config.seed = 0
    config.batch_size = 8
    config.offline_batch_size = 128 * config.env_length
    config.buffer_size = config.batch_size
    config.offline_buffer_size = config.offline_batch_size
    config.eval_batch_size = 512
    config.eval_interval_episodes = 5
    config.burnin_episodes = 30
    config.num_episodes = 5000
    config.pg_loss = "sum"
    config.pg_num_sample = 0

    config.hidden_dim_agent = (64, 64)
    config.optimizer_agent = "adamw"
    config.epsilon_exploration_eval = True
    config.hindsight_model_type = "hypernet"

    config.contribution = "parallel"
    config.parallel_keys = (
        "qnet_gt,qnet,causal_state,causal_reward,advantage,reinforce,traj_cv"
    )
    config.parallel_main_key = "qnet_gt"
    config.parallel_reset_before_update = False

    config.compute_mdp = True
    config.policy_var_first_state_only = True

    for key in config.parallel_keys.split(","):
        setattr(config, key, contribution_config(method=key, global_config=config))

    return config
