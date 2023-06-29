import os
from pathlib import Path

import pandas as pd
import wandb

USEFUL_COLUMNS = [
    "percentage_treasure_rewards",
    "state_overlap",
    "grouping",
    "seed",
    "config",
    "method",
    "env_length",
    "eval/episode",
    "policy_grad_var_dB",
    "policy_grad_snr_dB",
    "policy_grad_bias_dB",
]


def get_color_by_name(name):
    # plotly.colors.qualitative.Plotly
    return {
        "blue": "#636EFA",
        "dark blue": "#3B4296",
        "light blue": "#A1A8FC",
        "red": "#EF553B",
        "dark green": "#00CC96",
        "purple": "#AB63FA",
        "orange": "#FFA15A",
        "cyan": "#19D3F3",
        "pink": "#FF6692",
        "light green": "#B6E880",
        "rose": "#FF97FF",
        "yellow": "#FECB52",
        "blue2red1": "#774878",
        "blue2red2": "#b34f59",
    }[name]


def load_from_cache_or_compute(file_name, compute_fun):
    import glob

    if len(glob.glob(file_name)) > 0:
        return pd.read_csv(file_name, low_memory=False)
    else:
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        result = compute_fun(None)
        result.to_csv(file_name)
        return result


def parse_to_pd_frame(sweep_id, max_steps, filter_keys=None):
    api = wandb.Api()
    runs = api.sweep(sweep_id).runs
    df_list = []

    for i, run in enumerate(runs):
        print(run.name)
        if "_fields" not in run.config:
            continue
        config = {k: v for k, v in run.config["_fields"].items() if not k.startswith("_")}
        config.update({"config": run.config["config"]})

        df = run.history(
            samples=max_steps, keys=None, x_axis="_step", pandas=True, stream="default"
        )
        for k, v in config.items():
            df[k] = pd.Series([v for x in range(len(df.index))])

        if filter_keys is not None:
            filter_cols = []
            for key in filter_keys:
                filter_cols += [col for col in df.columns if key in col]

            df = df.filter(items=set(filter_cols))  # avoid duplicates

        df_list.append(df)

    return pd.concat(df_list)
