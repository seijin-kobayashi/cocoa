import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.colors import hex_to_rgb

from utils import get_color_by_name, load_from_cache_or_compute, parse_to_pd_frame, USEFUL_COLUMNS


import sys

SWEEP_IDS = [sys.argv[1]]

MAX_STEPS = 20000

# Define plotting keys and label mappings
metric_dict = dict(
    {
        "eval/policy_grad_var_dB": "Var dB",
        "eval/policy_grad_snr_dB": "SNR dB",
        "eval/policy_grad_bias_dB": "Bias dB",
    }
)

colors_dict = {
    "reinforce": get_color_by_name("light green"),
    "qnet": get_color_by_name("rose"),
    "causal_reward": get_color_by_name("dark blue"),
    "causal_reward_feature": get_color_by_name("light blue"),
    "causal_state": get_color_by_name("red"),
    "advantage": get_color_by_name("dark green"),
}
methods_dict = {
    "causal_reward": "COCOA<br>reward",
    "causal_reward_feature": "COCOA<br>feature",
    "causal_state": "HCA+",
    "qnet": "Q-critic",
    "advantage": "Advantage",
    "reinforce": "REIN-<br>FORCE",
}

# Construct the dataframe
pd_list = []
for id in SWEEP_IDS:
    df = load_from_cache_or_compute(
        file_name="cache/{}.csv".format(id.replace("/", "_")),
        compute_fun=lambda _: parse_to_pd_frame(id, MAX_STEPS, filter_keys=USEFUL_COLUMNS),
    )
    pd_list.append(df)

df = pd.concat(pd_list)
df = df[df["env_length"] == 103]
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Average in two stages: 1) bin by treasure rewards picked i.e. policy progress 2) env length
df_bin = df.groupby([pd.cut(df["eval/percentage_treasure_rewards"], 10), "seed"])
df_bin_mean = df_bin.mean(numeric_only=True).reset_index(level=1).reset_index(drop=True)

df_mean = df_bin_mean.mean(numeric_only=True)
df_sem = df_bin_mean.sem(numeric_only=True)
df_lower, df_upper = df_mean - df_sem, df_mean + df_sem

# Construct line plots with sem continuous error bands
fig = go.Figure()
for method in methods_dict:
    key_y = "eval/policy_grad_snr_dB" + "_" + method
    color = colors_dict[method]

    if key_y not in df_mean:
        print("warning: skipped ", key_y)
        continue

    group = "Ground truth" if "gt" in method else "Learned"
    side = "positive" if "gt" in method else "negative"

    fig.add_trace(
        go.Violin(
            x=len(np.array(df_bin_mean[key_y])) * [methods_dict[method]],
            y=np.array(df_bin_mean[key_y]),
            marker=dict(size=3.0),
            line=dict(width=0.5, color="rgba({},{},{},1.0)".format(*hex_to_rgb(color))),
        )
    )

fig.update_layout(violingap=0, violingroupgap=0)
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(
    font_family="Arial",
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=0),
    width=400,
    height=250,
    template="simple_white",
    yaxis_title="SNR dB",
)

import plotly.io as pio

pio.kaleido.scope.mathjax = None

# Save figure
fig.write_image("pdf/snr_aggregate_env103_learned.pdf")
