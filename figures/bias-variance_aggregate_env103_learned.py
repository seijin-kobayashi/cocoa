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
    "reinforce": "REINFORCE",
    "advantage": "Advantage",
    "causal_reward_feature": "COCOA-feature",
    "causal_reward": "COCOA-reward",
    "causal_state": "HCA+",
    "qnet": "Q-critic",
}

# Construct the dataframe
pd_list = []
for id in SWEEP_IDS:
    df = load_from_cache_or_compute(
        file_name="cache/{}.csv".format(id.replace("/", "_")),
        compute_fun=lambda _: parse_to_pd_frame(id, MAX_STEPS, filter_keys=USEFUL_COLUMNS),
    )
    pd_list.append(df)

MIN_VAL = -80
df = pd.concat(pd_list)

df = df[df["env_length"] == 103]
df.replace([np.inf], np.nan, inplace=True)
df.replace([-np.inf], MIN_VAL, inplace=True)

# Average in two stages: 1) bin by treasure rewards picked i.e. policy progress 2) env length
df_bin = df.groupby([pd.cut(df["eval/percentage_treasure_rewards"], 10), "seed"])
df_bin_mean = df_bin.mean(numeric_only=True).reset_index(level=1).reset_index(drop=True)

df_mean = df_bin_mean.mean(numeric_only=True)
df_sem = df_bin_mean.sem(numeric_only=True)
df_lower, df_upper = df_mean - df_sem, df_mean + df_sem

# Construct line plots with sem continuous error bands
fig = go.Figure()
for method in methods_dict:
    key_x = "eval/policy_grad_bias_dB" + "_" + method
    key_y = "eval/policy_grad_var_dB" + "_" + method
    color = colors_dict[method]

    if key_y not in df_mean:
        print("warning: skipped ", key_y)
        continue
    x = np.array(df_bin_mean[key_x])
    x = np.maximum(x, MIN_VAL)
    y = np.array(df_bin_mean[key_y])
    y = np.maximum(y, MIN_VAL)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            # error_x={"array": np.array(df_sem[key_x])},
            # error_y={"array": np.array(df_sem[key_y])},
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            line=dict(color="rgba({},{},{},1.0)".format(*hex_to_rgb(color))),
            name=methods_dict[method],
        )
    )

fig.update_layout(
    font_family="Arial",
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    margin=dict(l=0, r=0, b=0, t=0),
    width=400,
    height=250,
    template="simple_white",
    xaxis_title="Bias dB",
    yaxis_title="Variance dB",
)

print("WARNING: cutting y-axis range excluding some outliers for state-based")
fig.update_yaxes(range=[MIN_VAL - 5, 120])
fig.update_xaxes(range=[MIN_VAL - 5, 80])


import plotly.io as pio

pio.kaleido.scope.mathjax = None

# Save figure
fig.write_image("pdf/bias-variance_aggregate_env103_learned.pdf")
