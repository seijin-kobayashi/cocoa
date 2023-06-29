import pandas as pd
import plotly.graph_objs as go
from plotly.colors import hex_to_rgb

from utils import get_color_by_name, load_from_cache_or_compute, parse_to_pd_frame, USEFUL_COLUMNS
import sys

SWEEP_IDS = [sys.argv[1]]

MAX_STEPS = 10000
SWITCH_STEP = 60000

metric_dict = {
    "Treasure reward collected": (
        "eval/percentage_treasure_rewards",
        "eval/percentage_treasure_rewards",
        "eval/episode",
    ),
}
methods_dict = {
    "reinforce": "REINFORCE",
    "advantage": "Advantage",
    "qnet": "Q-critic",
    "causal_state": "HCA+",
    "causal_reward": "COCOA-reward",
    "causal_reward_feature": "COCOA-feature",
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

df = pd.concat(df_list)

# Construct line plots with sem continuous error bands
for metric in metric_dict:
    fig = go.Figure()
    first_key_y, second_key_y, key_x = metric_dict[metric]
    for config_key in methods_dict.keys():
        color = colors_dict[methods_dict[config_key]]
        df_tmp = df[df["method"] == config_key]
        df_tmp.loc[df_tmp[key_x] > SWITCH_STEP, first_key_y] = df_tmp[df_tmp[key_x] > SWITCH_STEP][
            second_key_y
        ]
        # df_tmp[first_key_y] = np.where(df_tmp[key_x]>SWITCH_STEP, df_tmp[second_key_y], df_tmp[first_key_y])
        df_mean = df_tmp.groupby(key_x).mean().reset_index()
        df_sem = df_tmp.groupby(key_x).sem().reset_index()
        df_lower, df_upper = df_mean - df_sem, df_mean + df_sem

        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_upper[first_key_y],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_mean[key_x],
                y=df_lower[first_key_y],
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
                y=df_mean[first_key_y],
                mode="lines",
                line=dict(color="rgba({},{},{},1.0)".format(*hex_to_rgb(color))),
                name=methods_dict[config_key],
            )
        )

    fig.update_layout(
        # showlegend=False,
        font_family="Arial",
        legend=dict(orientation="h", yanchor="top", y=1.6, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, b=0, t=0),
        template="simple_white",
        width=400,
        height=300,
        xaxis_title="Episode",
        yaxis_title=metric,
    )
    fig.update_xaxes(range=[55000, 65000])
    # fig.show()

    # Save figure
    import plotly.io as pio  # HACK: Prevent weird bug

    pio.kaleido.scope.mathjax = None
    fig.write_image(
        "pdf/reward-switch_performance_time_learned_{}.pdf".format(metric.replace(" ", "_"))
    )
