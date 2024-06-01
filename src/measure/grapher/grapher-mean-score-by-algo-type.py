"""
Grapher Program to observe mean score based on Algo Type (Not used anymore)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
import os

env = "taxi"  # "taxi", "frozen-lake" or "wumpus"
env_config = "classic"  # "classic" or "all" (for wumpus)
algos_to_be_tested_and_compared = [
    {
        "algo": "constant-epsilon",
        "constant_x": 0.1,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-basic",

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    {
        "algo": "ngu",
        "N": 2,
        "k": 6,
        "beta": 0.3,

        "actor_epsilon_greedy_epsilon": 0,
        "actor_epsilon_greedy_alpha": 7,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-ngu",
        "N": 2,
        "k": 6,
        "beta": 0.3,

        "actor_epsilon_greedy_alpha": 7,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-ngu-with-prioritized-replay-buffer",
        "N": 2,
        "k": 6,
        "beta": 0.4,
        "batch_size": 500,
        "replay_buffer_capacity": 5000,
        "replay_buffer_alpha": 0.6,
        "replay_buffer_beta": 0.4,

        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.9,
        "discount_factor": 0.99,
    },

    {
        "algo": "ngu-with-prioritized-replay-buffer",
        "N": 2,
        "k": 6,
        "beta": 0.4,
        "batch_size": 500,
        "replay_buffer_capacity": 5000,
        "replay_buffer_alpha": 0.6,
        "replay_buffer_beta": 0.4,

        "actor_epsilon_greedy_epsilon": 0.4,
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

num_of_episodes_legends = [100, 500, 1000]
num_of_runs = 3
validation_mean_over_x_episodes = 3

graph_title = "On algos"

df_global_all_runs_register = pd.DataFrame(
    columns=['id', 'try', 'validation_mean_over_x_episodes', 'env', 'env_config', 'algo_legend', 'number_of_episodes',
             'average_reward', 'num_of_environment_calls'])
if os.path.exists(f'../../../data/csv_files/global_storage_all_runs.csv'):
    df_global_all_runs_register = pd.read_csv(f'../../../data/csv_files/global_storage_all_runs.csv')


df_for_graph_generator = pd.DataFrame(
    columns=['id', 'legend', 'number_of_episodes',
             'average_reward'])

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
            f"classic"
        ]
elif (env == "taxi"):
    environment_configs = [
        f"classic"
    ]

elif (env == "frozen-lake"):
    environment_configs = [
        f"classic"
    ]

for algo_config in algos_to_be_tested_and_compared:
    print("New Algo")
    for current_env_config in environment_configs:
        for j in range(num_of_runs):
            legend = ""
            if algo_config["algo"] == "ngu":
                legend = f"ngu-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "constant-epsilon":
                legend = f"constant-epsilon-{algo_config['constant_x']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "linear-epsilon":
                legend = f"linear-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "logarithmic-epsilon":
                legend = f"logarithmic-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"

            for i in num_of_episodes_legends:
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
                    df_for_graph_generator.loc[len(df_for_graph_generator)] = [len(df_for_graph_generator), algo_config['legend'], already_existing_df.iloc[0]["number_of_episodes"], already_existing_df.iloc[0]["average_reward"] ]


print("Making Graph")
plt.figure(figsize=(10, 6))

for i in df_for_graph_generator['number_of_episodes'].unique():
    print(i)
    current_df = df_for_graph_generator[df_for_graph_generator['number_of_episodes'] == i]
    legend = current_df['number_of_episodes'].iloc[0]
    current_df = current_df.drop(['id', 'number_of_episodes'], axis=1)

    grouped_df = current_df.groupby('legend')['average_reward'].agg(['mean', 'min', 'max']).reset_index()

    sns.lineplot(x='legend', y='mean', data=grouped_df, label=f'{legend}')
    print("Making graph ...")

plt.title(graph_title)
plt.xlabel('Algos')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

plt.ylabel('Average Score')
plt.legend(title='Nulber of episodes', loc='upper left')

print("Saving Graph")
plt.savefig(f'../../../data/graphs/{graph_title}.png')