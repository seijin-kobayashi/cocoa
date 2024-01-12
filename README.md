# COCOA

Code accompanying the paper [Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis](https://github.com/seijin-kobayashi/cocoa).

## Dependencies

Please install [jax](https://jax.readthedocs.io/en/latest/installation.html), as well as the packages in requirements.txt.

## Experiments

To reproduce the results of the paper, you can run the following `wandb` sweeps followed by the corresponding scripts to create the plots.

### Linear key-to-door, performance, learnt-models

```
wandb sweep sweeps/performance_asymptotic_learnt.yaml
```
The relevant figures can be generated using the following scripts:
```
python3 figures/performance_asymptotic_length_learned.py <WANDB_SWEEP_ID>
python3 figures/performance_time_envlen103_learned.py <WANDB_SWEEP_ID>
```

### Linear key-to-door, performance, groundtruth-models
```
wandb sweep sweeps/performance_asymptotic_gt.yaml 
```

```
python3 figures/performance_time_envlen103_gt.py <WANDB_SWEEP_ID>
```
### Linear key-to-door, shadow training
```
wandb sweep sweeps/shadow_asymptotic_learnt.yaml
```

```
python3 figures/bias-variance-snr_asymptotic_length_learned.py <WANDB_SWEEP_ID>
python3 figures/bias-variance_aggregate_env103_learned.py <WANDB_SWEEP_ID>
python3 figures/snr_aggregate_env103_learned.py <WANDB_SWEEP_ID>
```
### Reward switching
```
wandb sweep sweeps/reward_switch.yaml
```

```
python3 figures/reward-switch_performance_time_learned.py <WANDB_SWEEP_ID>
```
### Reward aliasing
```
wandb sweep sweeps/aliasing_exp.yaml
```

```
python3 figures/performance_time_envlen103_reward-aliasing.py <WANDB_SWEEP_ID>
```

### Tree environment
```
wandb sweep sweeps/tree_env.yaml
```

```
python3 figures/var_asymptotic_state-overlap_gt.py <WANDB_SWEEP_ID>
```

### Interleaving tasks
```
wandb sweep task_interleaving_stochastic.yaml

wandb sweep task_interleaving_stochastic_gt.yaml
```


## Acknowledgements
Research supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
