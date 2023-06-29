import pandas as pd
import plotly.graph_objs as go
from plotly.colors import hex_to_rgb

from utils import get_color_by_name, load_from_cache_or_compute, parse_to_pd_frame, USEFUL_COLUMNS

import sys

SWEEP_IDS = [sys.argv[1]]

MAX_STEPS = 20000

metric_dict = {
    "eval/percentage_treasure_rewards": "Treasure reward collected",
}
methods_dict = {
    "reinforce": "REINFORCE",
    "qnet": "Q-critic",
    "causal_reward": "COCOA-reward",
    "causal_reward_feature": "COCOA-feature",
    "causal_state": "HCA+",
    "advantage": "Advantage",
}

colors_dict = {
    "REINFORCE": get_color_by_name("light green"),
    "Advantage": get_color_by_name("dark green"),
    "Q-critic": get_color_by_name("rose"),
    "HCA+": get_color_by_name("red"),
    "COCOA-reward": get_color_by_name("dark blue"),
    "COCOA-feature": get_color_by_name("light blue"),
}

# Construct the dataframe
df_list = []
for id in SWEEP_IDS:
    df = load_from_cache_or_compute(
        file_name="cache/{}.csv".format(id.replace("/", "_")),
        compute_fun=lambda _: parse_to_pd_frame(id, MAX_STEPS, filter_keys=USEFUL_COLUMNS),
    )
    df_list.append(df)

# Define plotting keys and label mappings
key_x = "eval/episode"
df = pd.concat(df_list)
df = df.dropna(subset=[key_x])

df = df[df["env_length"] == 103]
# Construct line plots
for metric in metric_dict:
    fig = go.Figure()
    for config_key in methods_dict:
        color = colors_dict[methods_dict[config_key]]
        df_tmp = df[df["method"] == config_key]
        df_tmp.sort_values(by=key_x)

        df_mean = df_tmp.groupby(key_x).mean().reset_index()
        df_sem = df_tmp.groupby(key_x).sem().reset_index()

        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_mean[metric] + df_sem[metric],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_mean[metric] - df_sem[metric],
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
                y=df_mean[metric],
                mode="lines",
                line=dict(color="rgba({},{},{},1.0)".format(*hex_to_rgb(color))),
                name=methods_dict[config_key],
            )
        )
    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_layout(
        # showlegend=False,
        font_family="Arial",
        legend=dict(orientation="h", yanchor="top", y=1.3, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, b=0, t=0),
        template="simple_white",
        width=400,
        height=250,
        xaxis_title="Episode",
        yaxis_title=metric_dict[metric],
    )
    # fig.show()

    # Save figure
    import plotly.io as pio  # HACK: Prevent weird bug

    pio.kaleido.scope.mathjax = None
    fig.write_image("pdf/performance_time_envlen103_learned_{}.pdf".format(metric.split("/")[1]))
