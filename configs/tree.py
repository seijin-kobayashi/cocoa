import ml_collections as mlc


def contribution_config(method, global_config, grouping):
    config = mlc.ConfigDict(type_safe=False)
    config.environment = global_config.environment
    config.method = method
    config.burnin_episodes = 0
    config.num_episodes = 3

    config.hindsight_model_type = "hypernet"
    config.feature_model_type = "mlp"

    if "qnet_gt" in method:
        config.hindsight_model_type = "hypernet"
        config.contribution = "qnet_gt"
        config.return_contribution = "advantage"

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

    elif method == "causal_feature_gt":
        config.hindsight_model_type = "hypernet"
        config.contribution = "causal_gt"
        config.hindsight_loss_type = "contrastive"
        config.return_contribution = "advantage"
        config.hindsight_feature_type = "tree_grouping"
        config.rewarding_outcome_nb_groups = grouping

    elif method == "reinforce_gt":
        config.hindsight_model_type = "hypernet"
        config.contribution = "reinforce"
        config.return_contribution = "action_value"

    else:
        raise ValueError

    return config


def environment_config(state_overlap: int, seed: int):
    config = mlc.ConfigDict()
    config.name = "tree"
    config.length = 4
    config.branching = 6
    config.state_overlap = state_overlap
    config.reward_overlap_fraction = 0.0
    config.reward_sparsify_fraction = 0.0
    config.only_reward_last_timestep = False
    config.action_dependent_reward = True
    config.number_different_rewards = 5
    config.negative_rewards = True
    config.seed = seed
    return config


def get_config(state_overlap_seed):
    config = mlc.ConfigDict()
    state_overlap, env_seed = state_overlap_seed.split(";")
    config.state_overlap = int(state_overlap)
    config.env_seed = int(env_seed)
    assert config.state_overlap >= 0
    config.environment = environment_config(config.state_overlap, config.env_seed)
    config.env_id = "tree_{}_{}".format(config.state_overlap, config.env_seed)

    config.lr_agent = 0.0
    config.epsilon_exploration = 0.0
    config.entropy_reg = 0.00
    config.pg_norm = 0.003

    config.seed = 0
    config.batch_size = 4
    config.buffer_size = config.batch_size
    config.offline_buffer_size = 0
    config.offline_batch_size = 128
    config.burnin_episodes = 0
    config.eval_batch_size = 2048
    config.eval_interval_episodes = 1
    config.num_episodes = 3
    config.pg_loss = "sum"
    config.pg_num_sample = 0

    config.hidden_dim_agent = (64, 64)
    config.tabular_agent = True
    config.optimizer_agent = "adamw"
    config.epsilon_exploration_eval = False

    config.hindsight_model_type = "hypernet"
    config.feature_model_type = "mlp"
    config.contribution = "parallel"
    parallel_keys = (
        "qnet_gt,causal_state_gt,causal_reward_gt,causal_feature_gt,reinforce_gt,advantage_gt"
    )
    config.parallel_main_key = "qnet_gt"
    config.parallel_reset_before_update = False

    config.compute_mdp = True
    config.policy_var_first_state_only = False

    groupings = [1, 2, 4, 8, 16, 32, 64, 128]

    parallel_keys_new = ""

    for key in parallel_keys.split(","):
        if key == "causal_feature_gt":
            for grouping in groupings:
                new_key = "causal_feature_gt{}".format(grouping)
                parallel_keys_new += new_key + ","
                setattr(
                    config,
                    new_key,
                    contribution_config(method=key, global_config=config, grouping=grouping),
                )
        else:
            parallel_keys_new += key + ","
            setattr(config, key, contribution_config(method=key, global_config=config, grouping=1))

    config.parallel_keys = parallel_keys_new[:-1]

    return config
