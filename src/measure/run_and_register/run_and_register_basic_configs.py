"""
Program that run all code based on configs and retrieve stats into csv files (it allow to then generate graphs)
"""

import pandas as pd
import os

### Parameters

env = "frozen-lake"  # "taxi", "frozen-lake", "grid-world" or "wumpus"
env_config = "classic"  # "classic" or "all" (for wumpus)
algos_to_be_tested_and_compared = [
    {
        "algo": "constant-epsilon",
        "constant_x": 0.8,

        "learning_rate": 0.1,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-basic",

        "learning_rate": 0.1,
        "discount_factor": 0.99,
    },
    {
        "algo": "ngu",
        "N": 2,
        "k": 6,
        "beta": 0.3,

        "actor_epsilon_greedy_epsilon": 0.8,
        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.1,
        "discount_factor": 0.99,
    },
    {
        "algo": "evo-ngu",
        "N": 2,
        "k": 6,
        "beta": 0.3,
        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.1,
        "discount_factor": 0.99,
    },
    {
        "algo": "ngu-with-prioritized-replay-buffer",
        "N": 1,
        "k": 6,
        "beta": 0.3,
        "batch_size": 500,
        "replay_buffer_capacity": 5000,
        "replay_buffer_alpha": 0.6,
        "replay_buffer_beta": 0.4,

        "actor_epsilon_greedy_epsilon": 0.8,
        "actor_epsilon_greedy_alpha": 8,

        "learning_rate": 0.1,
        "discount_factor": 0.99,
    },
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

        "learning_rate": 0.1,
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
    # {
    #     "algo": "ngu",
    #     "N": 1,
    #     "k": 6,
    #     "beta": 0.3,
    #
    #     "actor_epsilon_greedy_epsilon": 0.8,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.1,
    #     "discount_factor": 0.99,
    # },
    # {
    #     "algo": "evo-ngu",
    #     "N": 1,
    #     "k": 6,
    #     "beta": 0.3,
    #     "actor_epsilon_greedy_alpha": 7,
    #
    #     "learning_rate": 0.1,
    #     "discount_factor": 0.99,
    # },
    #
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
    #     "actor_epsilon_greedy_epsilon": 0.8,
    #     "actor_epsilon_greedy_alpha": 8,
    #
    #     "learning_rate": 0.1,
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
    #     "learning_rate": 0.1,
    #     "discount_factor": 0.99,
    # },
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

stop_when_min_score_obtained = False
use_existing_data_if_possible = True

max_num_of_episodes = None
max_num_of_environment_calls = 2500000
stats_every_x_steps = 10
min_score = 0.5
num_of_runs = 5
validation_mean_over_x_episodes = 20

# dataframes to register data in and from
df_global_all_runs_register = pd.DataFrame(
    columns=['id', 'try', 'validation_mean_over_x_episodes', 'env', 'env_config', 'algo_legend', 'number_of_episodes',
             'average_reward', 'num_of_environment_calls', 'mu_t_expl', 'num_of_different_states_visited'])
if os.path.exists(f'../../../data/csv_files/global_storage_all_runs.csv'):
    df_global_all_runs_register = pd.read_csv(f'../../../data/csv_files/global_storage_all_runs.csv')

df_global_min_score_achieved_register = pd.DataFrame(
    columns=['id', 'try', 'validation_mean_over_x_episodes', 'env', 'env_config', 'algo_legend',
             'mean_average_reward_to_be_achieved', 'achieved', 'average_reward_achieved', 'num_of_environment_calls',
             'achieved_after_num_of_episodes'])
if os.path.exists(f'../../../data/csv_files/global_storage_min_score_achieved.csv'):
    df_global_min_score_achieved_register = pd.read_csv(
        f'../../../data/csv_files/global_storage_min_score_achieved.csv')

# Environment Generator
environment_generators = None
if (env == "wumpus"):
    import src.environments.WumpusEnv.wumpus_env as WumpusEnv

    if env_config == "all":
        from src.environments.WumpusEnv.wumpus_configs import WumpusConfigs

        environment_generators = []

        for index, wumpusConfig in enumerate(WumpusConfigs):
            environment_generators.append({
                "environment_generator": lambda wumpusConfig=wumpusConfig: WumpusEnv.WumpusEnv(
                    size=wumpusConfig['size'],
                    entrance=wumpusConfig['entrance'],
                    heading=wumpusConfig['heading'],
                    wumpus=wumpusConfig['wumpus'],
                    gold=wumpusConfig['gold'],
                    pits=wumpusConfig['pits'],
                ),
                "env_config": f"all-{index}"
            })


    else:
        from src.environments.WumpusEnv.wumpus_configs import WumpusConfigs

        wumpusConfig = WumpusConfigs[0]
        environment_generators = [
            {
                "environment_generator": lambda: WumpusEnv.WumpusEnv(
                    size=wumpusConfig['size'],
                    entrance=wumpusConfig['entrance'],
                    heading=wumpusConfig['heading'],
                    wumpus=wumpusConfig['wumpus'],
                    gold=wumpusConfig['gold'],
                    pits=wumpusConfig['pits'],
                ),
                "env_config": f"classic"
            }
        ]
elif (env == "grid-world"):
    import src.environments.GridWorldEnv.grid_world_env as ge

    environment_generators = [
        {
            "environment_generator": lambda: ge.GridWorldEnv(),
            "env_config": f"classic"
        }
    ]


elif (env == "taxi"):
    import src.environments.TaxiEnv.taxi_env as TaxiEnv

    environment_generators = [
        {
            "environment_generator": lambda: TaxiEnv.TaxiEnv(),
            "env_config": f"classic"
        }
    ]

elif (env == "frozen-lake"):
    import src.environments.FrozenLakeEnv.frozen_lake_env as FrozenLakeEnv

    environment_generators = [
        {
            "environment_generator": lambda: FrozenLakeEnv.FrozenLakeEnv(),
            "env_config": f"classic"
        }
    ]

# Run
for environment_generator_config in environment_generators:
    print(environment_generator_config["env_config"])
    for algo_config in algos_to_be_tested_and_compared:

        environment_generator = environment_generator_config["environment_generator"]
        for j in range(num_of_runs):
            results = None
            legend = ""
            if algo_config["algo"] == "ngu":
                legend = f"ngu-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['actor_epsilon_greedy_epsilon']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "evo-ngu":
                legend = f"evo-ngu-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"

            elif algo_config["algo"] == "ngu-with-prioritized-replay-buffer":
                legend = f"ngu-with-prioritized-replay-buffer-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['batch_size']}-{algo_config['replay_buffer_capacity']}-{algo_config['replay_buffer_alpha']}-{algo_config['replay_buffer_beta']}-{algo_config['actor_epsilon_greedy_epsilon']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "evo-ngu-with-prioritized-replay-buffer":
                legend = f"evo-ngu-with-prioritized-replay-buffer-{algo_config['N']}-{algo_config['k']}-{algo_config['beta']}-{algo_config['batch_size']}-{algo_config['replay_buffer_capacity']}-{algo_config['replay_buffer_alpha']}-{algo_config['replay_buffer_beta']}-{algo_config['actor_epsilon_greedy_alpha']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"

            elif algo_config["algo"] == "constant-epsilon":
                legend = f"constant-epsilon-{algo_config['constant_x']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "linear-epsilon":
                legend = f"linear-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "logarithmic-epsilon":
                legend = f"logarithmic-epsilon-{algo_config['start']}-{algo_config['end']}-{algo_config['in_x_iterations']}-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            elif algo_config["algo"] == "evo-basic":
                legend = f"evo-basic-{algo_config['learning_rate']}-{algo_config['discount_factor']}"
            print(legend)

            must_run_code = True
            if use_existing_data_if_possible:
                must_run_code = False
                i = 0
                num_of_environment_calls = 0
                while True:
                    if max_num_of_episodes != None:
                        if i >= max_num_of_episodes:
                            break
                    if max_num_of_environment_calls != None:
                        if num_of_environment_calls >= max_num_of_environment_calls:
                            break
                    already_existing_df = df_global_all_runs_register[(df_global_all_runs_register["try"] == j) & (
                            df_global_all_runs_register['number_of_episodes'] == i) & (df_global_all_runs_register[
                                                                                           "algo_legend"] == legend) & (
                                                                              df_global_all_runs_register[
                                                                                  "env"] == env) & (
                                                                              df_global_all_runs_register[
                                                                                  "env_config"] ==
                                                                              environment_generator_config[
                                                                                  "env_config"]) & (
                                                                              df_global_all_runs_register[
                                                                                  "validation_mean_over_x_episodes"] == validation_mean_over_x_episodes)]

                    if len(already_existing_df) == 0:
                        must_run_code = True
                        break
                    num_of_environment_calls = already_existing_df.iloc[-1]["num_of_environment_calls"]

                    i += stats_every_x_steps
                # if not must_run_code:
                     # already_existing_df = df_global_min_score_achieved_register[
                    #     (df_global_min_score_achieved_register["try"] == j) & (
                    #             df_global_min_score_achieved_register["algo_legend"] == legend) & (
                    #             df_global_min_score_achieved_register["env"] == env) & (
                    #             df_global_all_runs_register["env_config"] == environment_generator_config[
                    #         "env_config"]) & (
                    #             df_global_min_score_achieved_register[
                    #                 'mean_average_reward_to_be_achieved'] == min_score) & (
                    #             df_global_min_score_achieved_register[
                    #                 "validation_mean_over_x_episodes"] == validation_mean_over_x_episodes)]
                    # if len(already_existing_df) == 0:
                    #     print("yo1")
                    #     must_run_code = True

            if must_run_code:
                print("Running Algo ...")

                if algo_config["algo"] == "ngu":
                    import src.algos.ngu as ngu

                    algo = ngu.NGU(environment_generator, N=algo_config['N'], k=algo_config['k'],
                                   beta=algo_config['beta'],
                                   actor_epsilon_greedy_epsilon=algo_config["actor_epsilon_greedy_epsilon"],
                                   actor_epsilon_greedy_alpha=algo_config["actor_epsilon_greedy_alpha"])

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                if algo_config["algo"] == "evo-ngu":
                    import src.algos.evo_ngu as ngu

                    algo = ngu.EVO_NGU(environment_generator, N=algo_config['N'], k=algo_config['k'],
                                       beta=algo_config['beta'],
                                       actor_epsilon_greedy_alpha=algo_config["actor_epsilon_greedy_alpha"])

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                if algo_config["algo"] == "ngu-with-prioritized-replay-buffer":
                    import src.algos.ngu_with_prioritized_replay_buffer as ngu_with_prioritized_replay_buffer

                    algo = ngu_with_prioritized_replay_buffer.NGUWithPrioritizedReplayBuffer(environment_generator,
                                                                                             N=algo_config['N'],
                                                                                             k=algo_config['k'],
                                                                                             beta=algo_config['beta'],
                                                                                             batch_size=algo_config[
                                                                                                 'batch_size'],
                                                                                             replay_buffer_capacity=
                                                                                             algo_config[
                                                                                                 'replay_buffer_capacity'],
                                                                                             replay_buffer_alpha=
                                                                                             algo_config[
                                                                                                 'replay_buffer_alpha'],
                                                                                             replay_buffer_beta=
                                                                                             algo_config[
                                                                                                 'replay_buffer_beta'],

                                                                                             actor_epsilon_greedy_epsilon=
                                                                                             algo_config[
                                                                                                 "actor_epsilon_greedy_epsilon"],
                                                                                             actor_epsilon_greedy_alpha=
                                                                                             algo_config[
                                                                                                 "actor_epsilon_greedy_alpha"])

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                elif algo_config["algo"] == "evo-ngu-with-prioritized-replay-buffer":
                    import src.algos.evo_ngu_with_prioritized_replay_buffer as evo_ngu_with_prioritized_replay_buffer

                    algo = evo_ngu_with_prioritized_replay_buffer.EvoNGUWithPrioritizedReplayBuffer(
                        environment_generator,
                        N=algo_config['N'],
                        k=algo_config['k'],
                        beta=algo_config['beta'],
                        batch_size=algo_config[
                            'batch_size'],
                        replay_buffer_capacity=
                        algo_config[
                            'replay_buffer_capacity'],
                        replay_buffer_alpha=
                        algo_config[
                            'replay_buffer_alpha'],
                        replay_buffer_beta=
                        algo_config[
                            'replay_buffer_beta'],
                        actor_epsilon_greedy_alpha=
                        algo_config[
                            "actor_epsilon_greedy_alpha"])

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                elif algo_config["algo"] == "evo-basic":
                    import src.algos.evo_default_q_learning as evo_default_q_learning

                    algo = evo_default_q_learning.EvoDefaultQLearning(environment_generator)

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                elif algo_config["algo"] == "constant-epsilon":
                    import src.algos.default_q_learning as defaultQLearning
                    import src.utils.epsilon_basic_functions as ebf

                    algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                             ebf.EpsilonBasicFuncs().constant_x(
                                                                 algo_config['constant_x']))

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)
                elif algo_config["algo"] == "linear-epsilon":
                    import src.algos.default_q_learning as defaultQLearning
                    import src.utils.epsilon_basic_functions as ebf

                    algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                             ebf.EpsilonBasicFuncs().linear(algo_config['start'],
                                                                                            algo_config['end'],
                                                                                            algo_config[
                                                                                                'in_x_iterations']))

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                elif algo_config["algo"] == "logarithmic-epsilon":
                    import src.algos.default_q_learning as defaultQLearning
                    import src.utils.epsilon_basic_functions as ebf

                    algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                             ebf.EpsilonBasicFuncs().linear(algo_config['start'],
                                                                                            algo_config['end'],
                                                                                            algo_config[
                                                                                                'in_x_iterations']))

                    results = algo.run(algo_config['learning_rate'], algo_config['discount_factor'],
                                       max_num_of_episodes=max_num_of_episodes,max_num_of_environment_calls=max_num_of_environment_calls, min_score=min_score,
                                       retrieve_stats=True, retrieve_stats_every_x_steps=stats_every_x_steps,
                                       validation_mean_over_x_episodes=validation_mean_over_x_episodes,
                                       stop_training_when_min_score_obtained=stop_when_min_score_obtained)

                for score_store in results["all_scores_store"]:
                    df_global_all_runs_register.loc[len(df_global_all_runs_register)] = [
                        len(df_global_all_runs_register), j, validation_mean_over_x_episodes, env,
                        environment_generator_config["env_config"], legend, score_store["episode_num"],
                        score_store["average_reward"], score_store["num_of_environment_calls"],
                        score_store["stats"]["mu_t_expl"], score_store["stats"]["num_of_different_states_visited"]]

                df_global_min_score_achieved_register.loc[len(df_global_min_score_achieved_register)] = [
                    len(df_global_min_score_achieved_register), j, validation_mean_over_x_episodes, env,
                    environment_generator_config["env_config"], legend, min_score,
                    results["achieved_min_score"]["achieved_min_score"],
                    results["achieved_min_score"]["average_reward"],
                    results["achieved_min_score"]["after_num_of_environment_calls"],
                    results["achieved_min_score"]["after_num_of_episodes"]]

                df_global_all_runs_register['try'] = df_global_all_runs_register['try'].astype(int)
                df_global_all_runs_register['validation_mean_over_x_episodes'] = df_global_all_runs_register[
                    'validation_mean_over_x_episodes'].astype(int)
                df_global_all_runs_register['number_of_episodes'] = df_global_all_runs_register[
                    'number_of_episodes'].astype(int)
                df_global_all_runs_register['average_reward'] = df_global_all_runs_register['average_reward'].astype(
                    float)
                df_global_all_runs_register['num_of_environment_calls'] = df_global_all_runs_register[
                    'num_of_environment_calls'].astype(int)

                df_global_min_score_achieved_register['try'] = df_global_min_score_achieved_register['try'].astype(int)
                df_global_min_score_achieved_register['validation_mean_over_x_episodes'] = \
                    df_global_min_score_achieved_register['validation_mean_over_x_episodes'].astype(int)
                df_global_min_score_achieved_register['mean_average_reward_to_be_achieved'] = \
                    df_global_min_score_achieved_register['mean_average_reward_to_be_achieved'].astype(float)
                df_global_min_score_achieved_register['average_reward_achieved'] = \
                    df_global_min_score_achieved_register['average_reward_achieved'].astype(float)
                df_global_min_score_achieved_register['num_of_environment_calls'] = \
                    df_global_min_score_achieved_register['num_of_environment_calls'].astype(int)
                df_global_min_score_achieved_register['achieved_after_num_of_episodes'] = \
                    df_global_min_score_achieved_register['achieved_after_num_of_episodes'].astype(int)

                df_global_all_runs_register.to_csv(f'../../../data/csv_files/global_storage_all_runs.csv', index=False)
                df_global_min_score_achieved_register.to_csv(
                    f'../../../data/csv_files/global_storage_min_score_achieved.csv', index=False)
