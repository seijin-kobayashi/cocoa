import pandas as pd
import plotly.graph_objs as go
import numpy as np

from plotly.colors import hex_to_rgb

from utils import get_color_by_name, load_from_cache_or_compute, parse_to_pd_frame, USEFUL_COLUMNS

import sys

SWEEP_IDS = [sys.argv[1]]

MAX_STEPS = 20000

# Define plotting keys and label mappings
key_x = "state_overlap"
metric_dict = dict(
    {
        "eval/policy_grad_var_dB": "Var dB",
    }
)

methods_dict = {
    # "advantage_gt": "Advantage",
    "reinforce_gt": "REINFORCE",
    "causal_state_gt": "HCA+",
    "causal_reward_gt": "COCOA full grouping",
    "causal_feature_gt32": "COCOA more grouping",
    "causal_feature_gt4": "COCOA little grouping",
    # "qnet_gt": "Q-critic",
}
colors_dict = {
    "reinforce_gt": get_color_by_name("light green"),
    "qnet_gt": get_color_by_name("rose"),
    "causal_marginal_gt": get_color_by_name("purple"),
    "causal_reward_gt": get_color_by_name("dark blue"),
    "causal_feature_gt4": get_color_by_name("blue2red1"),
    "causal_feature_gt32": get_color_by_name("blue2red2"),
    "causal_state_gt": get_color_by_name("red"),
    "advantage_gt": get_color_by_name("dark green"),
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
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=[key_x])

# Average in two stages: 1) bin by treasure rewards picked i.e. policy progress 2) env length
# df_bin = df.groupby([pd.cut(df["eval/percentage_treasure_rewards"], 10), key_x, "seed"])
# df_bin_mean = df_bin.mean(numeric_only=True).reset_index(level=1).reset_index(drop=True)

df_mean = df.groupby(key_x).mean(numeric_only=True).reset_index()
df_sem = df.groupby(key_x).sem().reset_index()
df_lower, df_upper = df_mean - df_sem, df_mean + df_sem

# Construct line plots with sem continuous error bands
for metric in metric_dict.keys():
    fig = go.Figure()
    for method in methods_dict:
        key_y = metric + "_" + method
        color = colors_dict[method]

        if key_y not in df_mean.columns:
            print("warning: skipped ", key_y)
            continue

        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_upper[key_y],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_lower[key_y],
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba({},{},{},0.2)".format(*hex_to_rgb(color)),
                fill="tonexty",
                showlegend=False,
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_mean[key_y],
                mode="lines+markers",
                line=dict(color="rgba({},{},{},1.0)".format(*hex_to_rgb(color))),
                name=methods_dict[method],
            )
        )

    fig.update_layout(
        font_family="Arial",
        legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, b=0, t=0),
        width=400,
        height=250,
        template="simple_white",
        xaxis_title="State overlap",
        yaxis_title=metric_dict[metric],
    )
    fig.update_layout()

    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    # Save figure
    fig.write_image("pdf/var_asymptotic_state-overlap_gt.pdf")
