import chex
import haiku as hk
import jax
import jax.numpy as jnp
from ccoa.networks.utils import relu_gate


def get_policy_model(config, env, mdp):
    def policy_model(x):
        chex.assert_equal(x.shape[1:], env.observation_shape)  # assumes a leading batch dim
        x = hk.Flatten(preserve_dims=1)(x)
        x = hk.nets.MLP(output_sizes=config.hidden_dim_agent + (env.num_actions,))(x)
        return x

    def tabular_policy_model(x):
        logit = hk.get_parameter("logit", [mdp.num_state, mdp.num_actions], init=jnp.zeros)
        state = jax.vmap(mdp.observation_to_state)(x)
        return state @ logit

    if config.get("tabular_agent", False):
        return tabular_policy_model

    return policy_model


def get_qvalue_model(config, env):
    def qvalue_model(x):
        x = hk.Flatten(preserve_dims=1)(x)
        x = hk.nets.MLP(output_sizes=config.hidden_dim_qnet + (env.num_actions,))(x)
        return x

    return qvalue_model


def get_value_model(config, env):
    def value_model(x):
        x = hk.Flatten(preserve_dims=1)(x)
        x = hk.nets.MLP(output_sizes=config.hidden_dim_value + (1,))(x)
        return x

    return value_model


def get_hindsight_model(config, env):
    def hindsight_model(observations, hindsight_objects, policy_logits):
        if config.hindsight_model_type == "mlp":
            x = jnp.concatenate(
                [observations.flatten(), hindsight_objects.flatten(), policy_logits.flatten()],
                axis=0,
            )
            x = hk.nets.MLP(output_sizes=config.hidden_dim_hindsight + (env.num_actions,))(x)
            return x
        elif config.hindsight_model_type == "hypernet":
            x = jnp.concatenate([observations.flatten(), hindsight_objects.flatten()], axis=0)
            z = jax.nn.relu(hk.Linear(256)(x))

            logits = relu_gate(hk.Linear(2 * env.num_actions)(z))
            gate = relu_gate(hk.Linear(2 * 2 * env.num_actions * env.num_actions)(z))

            policy_logit_features = jnp.concatenate(
                [policy_logits, jnp.log(1 - jax.nn.softmax(policy_logits, axis=-1))]
            )

            gate_proj = gate.reshape(env.num_actions, 2 * env.num_actions)
            gated_policy_logits = gate_proj @ policy_logit_features

            return logits + gated_policy_logits

    return hindsight_model


def get_feature_model(config, env):
    def feature_model(x, action):
        assert len(x.shape) == 1

        gate = hk.get_parameter(
            name="gate",
            shape=(env.num_actions, config.hidden_dim_features, *x.shape),
            init=hk.initializers.RandomNormal(mean=config.threshold_shift),
        )

        def hard_sigmoid_st(x):
            return x + jax.lax.stop_gradient((x > config.threshold_shift) * 1.0 - x)

        feature = hard_sigmoid_st(gate[action].reshape(config.hidden_dim_features, *x.shape)) @ x
        feature = hard_sigmoid_st(feature)
        return feature

    return feature_model
