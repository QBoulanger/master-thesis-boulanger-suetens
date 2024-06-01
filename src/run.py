"""
Program used to run specified algo on specified environment with given parameters
"""

import argparse

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Frozen Lake argument parser')

    # Add arguments
    parser.add_argument('--env', type=str,
                        help='Algo use. Must be one of: ["wumpus", "taxi", "frozen-lake"]',
                        default="wumpus")
    parser.add_argument('--algo', type=str,
                        help='Algo use. Must be one of: ["ngu", "custom", "constant-epsilon", "linear-epsilon", '
                             '"logarithmic-epsilon"]',
                        default="custom")
    parser.add_argument('--constant_x', type=float,
                        help='Exploration ratio func value when algo="constant"',
                        default=0.5)
    parser.add_argument('--start', type=float,
                        help='Exploration ratio func value at start when algo="linear" or '
                             '"logarithmic"',
                        default=0.8)
    parser.add_argument('--end', type=float,
                        help='Exploration ratio func value at end when algo="linear" or "logarithmic"',
                        default=0.3)
    parser.add_argument('--in_x_iteration', type=int,
                        help='Number of iterations to go from start to end when algo="linear" '
                             '"logarithmic"',
                        default=2000)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.1)
    parser.add_argument('--discount_factor', type=float, help='Discount factor', default=0.99)
    parser.add_argument('--num_of_episodes', type=int, help='Num of Episodes', default=1000)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=2)

    # Parse the command line arguments
    args = parser.parse_args()

    env = args.env
    algo = args.algo
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episodes = args.num_of_episodes
    constant_x = args.constant_x
    start = args.start
    end = args.end
    in_x_iteration = args.in_x_iteration

    total_res = 0

    environment_generator = None
    if (env == "wumpus"):
        import src.environments.WumpusEnv.wumpus_env as WumpusEnv

        environment_generator = lambda: WumpusEnv.WumpusEnv()
    elif (env == "taxi"):
        import src.environments.TaxiEnv.taxi_env as TaxiEnv

        environment_generator = lambda: TaxiEnv.TaxiEnv()

    elif (env == "frozen-lake"):
        import src.environments.FrozenLakeEnv.frozen_lake_env as FrozenLakeEnv

        environment_generator = lambda: FrozenLakeEnv.FrozenLakeEnv()

    if args.algo == "custom":
        # Custom part: We can change that as we want for easy testing
        import pprint

        import src.algos.ngu as ngu

        print("Running custom algo")

        for j in range(args.num_of_runs):
            algo = ngu.NGU(environment_generator)

            results = algo.run(learning_rate, discount_factor, num_of_episodes, min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=1)

            pprint.pprint(results)
            total_res += results["at_end"]["average_reward"]

    elif args.algo == "ngu":
        import pprint
        import src.algos.ngu as ngu

        print("Running custom algo")

        for j in range(args.num_of_runs):
            algo = ngu.NGU(environment_generator)

            results = algo.run(learning_rate, discount_factor, num_of_episodes,  min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=1)

            pprint.pprint(results)
            total_res += results["at_end"]["average_reward"]


    elif args.algo == "constant-epsilon":

        print("Running constant func with following params:\n",
              "- Constant X: " + str(constant_x) + "\n",
              "- Exploration-ratio: " + str(args.constant_x) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episodes: " + str(num_of_episodes) + "\n"
                                                             "- Num of runs: " + str(args.num_of_runs) + "\n")

        import pprint
        import src.algos.default_q_learning as defaultQLearning
        import src.utils.epsilon_basic_functions as ebf

        for j in range(args.num_of_runs):
            algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                     ebf.EpsilonBasicFuncs().constant_x(constant_x))

            results = algo.run(learning_rate, discount_factor, num_of_episodes, min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=3)

            pprint.pprint(results)
            total_res += results["at_end"]["average_reward"]

    elif args.algo == "linear-epsilon":

        print("Running linear func with following params:\n",
              "- start: " + str(args.start) + "\n",
              "- end: " + str(args.end) + "\n",
              "- in_x_iteration: " + str(args.in_x_iteration) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episodes: " + str(num_of_episodes) + "\n",
              "- Num of runs: " + str(args.num_of_runs) + "\n")
        import pprint
        import src.algos.default_q_learning as defaultQLearning
        import src.utils.epsilon_basic_functions as ebf

        for j in range(args.num_of_runs):
            algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                     ebf.EpsilonBasicFuncs().linear(start=start, end=end,
                                                                                    in_x_iteration=in_x_iteration))

            results = algo.run(learning_rate, discount_factor, num_of_episodes,  min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=3)

            pprint.pprint(results)
            total_res += results["at_end"]["average_reward"]
    elif args.algo == "logarithmic-epsilon":

        print("Running logarithmic func with following params:\n",
              "- start: " + str(args.start) + "\n",
              "- end: " + str(args.end) + "\n",
              "- in_x_iteration: " + str(args.in_x_iteration) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episodes: " + str(num_of_episodes) + "\n",
              "- Num of runs: " + str(args.num_of_runs) + "\n")

        import pprint
        import src.algos.default_q_learning as defaultQLearning
        import src.utils.epsilon_basic_functions as ebf

        for j in range(args.num_of_runs):
            algo = defaultQLearning.DefaultQLearning(environment_generator,
                                                     ebf.EpsilonBasicFuncs().logarithmic(start=start, end=end,
                                                                                         in_x_iteration=in_x_iteration))

            results = algo.run(learning_rate, discount_factor, num_of_episodes,  min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=3)

            pprint.pprint(results)
            total_res += results["at_end"]["average_reward"]
    else:
        print("Bad Arguments")
        exit(1)

    print("total_average_reward : " + str(total_res / args.num_of_runs))
    exit(0)
