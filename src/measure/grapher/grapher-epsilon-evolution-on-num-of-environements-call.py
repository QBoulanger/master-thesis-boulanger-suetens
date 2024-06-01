"""
Grapher Program to observe epsilon evolution based on num of environment calls
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
import os
import numpy as np



env = "taxi"  # "taxi", "frozen-lake" or "wumpus"
env_config = "classic" # "classic" or "all" (for wumpus)
algos_to_be_tested_and_compared = [
    {
        "algo": "constant-epsilon",
        "constant_x": 0.2,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-basic",

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    # {
    #     "algo": "ngu",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #
    #     "actor_epsilon_greedy_epsilon": 0,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "ngu",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #
    #     "actor_epsilon_greedy_epsilon": 0.3,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "ngu",
    #     "N": 1,
    #     "k": 6,
    #     "beta": 0.3,
    #
    #     "actor_epsilon_greedy_epsilon": 0,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    {
        "algo": "ngu",
        "N": 1,
        "k": 6,
        "beta": 0.3,

        "actor_epsilon_greedy_epsilon": 0.3,
        "actor_epsilon_greedy_alpha": 7,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    # {
    #     "algo": "evo-ngu",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    {
        "algo": "evo-ngu",
        "N": 1,
        "k": 6,
        "beta": 0.3,
        "actor_epsilon_greedy_alpha": 7,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },

    # {
    #     "algo": "ngu-with-prioritized-replay-buffer",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #     "batch_size": 500,
    #     "replay_buffer_capacity": 5000,
    #     "replay_buffer_alpha": 0.6,
    #     "replay_buffer_beta": 0.4,
    #
    #     "actor_epsilon_greedy_epsilon": 0.3,
    #     "actor_epsilon_greedy_alpha": 8,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "ngu-with-prioritized-replay-buffer",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #     "batch_size": 500,
    #     "replay_buffer_capacity": 5000,
    #     "replay_buffer_alpha": 0.6,
    #     "replay_buffer_beta": 0.4,
    #
    #     "actor_epsilon_greedy_epsilon": 0,
    #     "actor_epsilon_greedy_alpha": 8,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    {
        "algo": "ngu-with-prioritized-replay-buffer",
        "N": 1,
        "k": 6,
        "beta": 0.3,
        "batch_size": 500,
        "replay_buffer_capacity": 5000,
        "replay_buffer_alpha": 0.6,
        "replay_buffer_beta": 0.4,

        "actor_epsilon_greedy_epsilon": 0.3,
        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    # {
    #     "algo": "ngu-with-prioritized-replay-buffer",
    #     "N": 1,
    #     "k": 6,
    #     "beta": 0.3,
    #     "batch_size": 500,
    #     "replay_buffer_capacity": 5000,
    #     "replay_buffer_alpha": 0.6,
    #     "replay_buffer_beta": 0.4,
    #
    #     "actor_epsilon_greedy_epsilon": 0,
    #     "actor_epsilon_greedy_alpha": 8,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "evo-ngu-with-prioritized-replay-buffer",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #     "batch_size": 500,
    #     "replay_buffer_capacity": 5000,
    #     "replay_buffer_alpha": 0.6,
    #     "replay_buffer_beta": 0.4,
    #
    #     "actor_epsilon_greedy_alpha": 8,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    {
        "algo": "evo-ngu-with-prioritized-replay-buffer",
        "N": 1,
        "k": 6,
        "beta": 0.3,
        "batch_size": 500,
        "replay_buffer_capacity": 5000,
        "replay_buffer_alpha": 0.6,
        "replay_buffer_beta": 0.4,

        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    # {
    #     "algo": "ngu",
    #     "N": 2,
    #     "k": 6,
    #     "beta": 0.3,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    #
    # {
    #     "algo": "constant-epsilon",
    #     "constant_x": 0.1,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "linear-epsilon",
    #     "start": 0.9,
    #     "end": 0.1,
    #     "in_x_iterations": 2000,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    #
    # },
    # {
    #     "algo": "logarithmic-epsilon",
    #     "start": 0.9,
    #     "end": 0.1,
    #     "in_x_iterations": 2000,
    #
    #     "learning_rate": 0.9,
    #     "discount_factor": 0.99,
    #
    # },
]
max_num_of_episodes = 2000
stats_every_x_steps = 10
num_of_runs = 3
validation_mean_over_x_episodes = 1

graph_title = "Epsilon Evolution of Algos with N=1 on Taxi"

df_global_all_runs_register = pd.DataFrame(
    columns=['id', 'try', 'validation_mean_over_x_episodes', 'env', 'env_config', 'algo_legend', 'number_of_episodes',
             'average_reward', 'num_of_environment_calls', 'mu_t_expl', 'num_of_different_states_visited'])
if os.path.exists(f'../../../data/csv_files/global_storage_all_runs.csv'):
    df_global_all_runs_register = pd.read_csv(f'../../../data/csv_files/global_storage_all_runs.csv')


df_for_graph_generator = pd.DataFrame(
    columns=['id', 'try',  'legend','num_of_environment_calls',
             'mu_epsilon'])
legendsMatching = {}
color_dict = {}
# Environment Generator
environment_configs = None
if (env == "wumpus"):
    import src.environments.WumpusEnv.wumpus_env as WumpusEnv

    if env_config == "all":
        from src.environments.WumpusEnv.wumpus_configs import WumpusConfigs

        environment_configs = [
            f"all-{index}"

            for index, wumpusConfig in enumerate(WumpusConfigs)
        ]
    else:
        environment_configs = [
            env_config
        ]
elif (env == "taxi"):
    environment_configs = [
        env_config
    ]

elif (env == "frozen-lake"):
    environment_configs = [
        env_config
    ]

for algo_config in algos_to_be_tested_and_compared:
    print("New Algo")
    for current_env_config in environment_configs:
        for j in range(num_of_runs):
            legend = ""
            niceLegend = ""
            color = ""
            if algo_config["algo"] == "ngu":
                legend = f"ngu-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['actor_epsilon_greedy_epsilon']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
                niceLegend = 'NGU'
                color = 'brown'
            elif algo_config["algo"] == "evo-ngu":
                legend = f"evo-ngu-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
                niceLegend = "EVO-NGU"
                color = 'red'
            elif algo_config["algo"] == "ngu-with-prioritized-replay-buffer":
                niceLegend = "PNGU"
                color = 'orange'
                legend = f"ngu-with-prioritized-replay-buffer-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['batch_size']}-{algo_config['replay_buffer_capacity']}-{algo_config['replay_buffer_alpha']}-{algo_config['replay_buffer_beta']}-{algo_config['actor_epsilon_greedy_epsilon']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "evo-ngu-with-prioritized-replay-buffer":
                legend = f"evo-ngu-with-prioritized-replay-buffer-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['batch_size']}-{algo_config['replay_buffer_capacity']}-{algo_config['replay_buffer_alpha']}-{algo_config['replay_buffer_beta']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
                niceLegend = "EVO-PNGU"
                color = 'green'
            elif algo_config["algo"] == "constant-epsilon":
                color = 'blueviolet'
                niceLegend = f"Baseline ({algo_config['constant_x']})"
                legend = f"constant-epsilon-{algo_config['constant_x']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "linear-epsilon":
                legend = f"linear-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "logarithmic-epsilon":
                legend = f"logarithmic-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "evo-basic":
                legend = f"evo-basic-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
                niceLegend = "EVO-Basic"
                color = 'dodgerblue'
            print(legend)
            legendsMatching[legend] = niceLegend
            color_dict[legend] = color
            for i in range(0, max_num_of_episodes, stats_every_x_steps):
                already_existing_df = df_global_all_runs_register[(df_global_all_runs_register["try"] == j) & (
                        df_global_all_runs_register['number_of_episodes'] == i) & (df_global_all_runs_register[
                                                                                       "algo_legend"] == legend) & (
                                                                          df_global_all_runs_register[
                                                                              "env"] == env) & (
                                                                          df_global_all_runs_register[
                                                                              "env_config"] ==
                                                                          current_env_config) & (
                                                                          df_global_all_runs_register[
                                                                              "validation_mean_over_x_episodes"] == validation_mean_over_x_episodes)]

                if len(already_existing_df) == 0:
                    raise ValueError("Missing data")
                else:
                    df_for_graph_generator.loc[len(df_for_graph_generator)] = [len(df_for_graph_generator), already_existing_df.iloc[0]["try"], legend, already_existing_df.iloc[0]["num_of_environment_calls"], already_existing_df.iloc[0]["mu_t_expl"] ]



print("Making Graph")
plt.figure(figsize=(10, 6))

list_of_current = []
legends = []
for i in df_for_graph_generator['legend'].unique():
    current_df = df_for_graph_generator[df_for_graph_generator['legend'] == i]

    all_trys = []
    for group in current_df.groupby('try'):
        all_trys.append(group[1])

    merged_df = pd.DataFrame()

    for df in all_trys:
        df = df.reset_index(drop=True)
        if merged_df.empty:
            merged_df = df
            merged_df['max'] = np.full(len(merged_df), -float('inf'))
            merged_df['min'] = np.full(len(merged_df), float('inf'))
        else:
            for i, row in df.iterrows():
                merged_df.at[i, 'num_of_environment_calls'] += row['num_of_environment_calls']
                merged_df.at[i, 'mu_epsilon'] += row['mu_epsilon']
                merged_df.at[i, 'max'] = max(merged_df.at[i, 'max'], row['mu_epsilon'])
                merged_df.at[i, 'min'] = min(merged_df.at[i, 'min'], row['mu_epsilon'])

    merged_df['num_of_environment_calls'] = merged_df['num_of_environment_calls'].div(len(all_trys))
    merged_df['mu_epsilon'] = merged_df['mu_epsilon'].div(len(all_trys))

    merged_df = merged_df.rename(columns={'mu_epsilon': 'mean'})

    legends.append(current_df['legend'].iloc[0])
    merged_df = merged_df.drop(['id', 'legend'], axis=1)

    merged_df['num_of_environment_calls'] = merged_df['num_of_environment_calls'].apply(
        lambda x: round(x / 100) * 100)

    list_of_current.append(merged_df)

print("Making graph ...")

all_A_values = np.concatenate([df['num_of_environment_calls'].values for df in list_of_current])
all_A_values.sort()
common_X = np.linspace(all_A_values.min(), all_A_values.max(), 1000)


def my_interpolate(df, col):
    for i, row in df.iterrows():
        if pd.isna(row[col]):
            valid_indices = df[df[col].notna()].index
            differences = np.abs(df.loc[valid_indices, 'num_of_environment_calls'] - row['num_of_environment_calls'])
            nearest_index = valid_indices[differences.argmin()]
            df.at[i, col] = df.at[nearest_index, col]
    return df


for i, df in enumerate(list_of_current):
    df_complete = pd.DataFrame({'num_of_environment_calls': all_A_values}).merge(df, on='num_of_environment_calls',
                                                                                 how='left')

    df_complete = my_interpolate(df_complete, 'mean')
    df_complete = my_interpolate(df_complete, 'min')
    df_complete = my_interpolate(df_complete, 'max')

    sns.lineplot(x='num_of_environment_calls', y='mean', data=df_complete, label=f'{legendsMatching[i]}', color=color_dict[i])

    plt.fill_between(df_complete['num_of_environment_calls'], df_complete['min'], df_complete['max'], alpha=0.2, color=color_dict[i])

plt.title(graph_title)
plt.xlabel('Number of Environment calls')
plt.ylabel('Mu Epsilon')
plt.legend(title='Algorithm used', loc='upper right')

print("Saving Graph")
plt.savefig(f'../../../data/graphs/{graph_title}-env-calls.png')
