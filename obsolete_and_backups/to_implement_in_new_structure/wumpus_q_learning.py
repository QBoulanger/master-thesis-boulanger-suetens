import sys
import gym_wumpus
import gym as gym
import numpy as np
import argparse
import pickle
import warnings

sys.path.append('../../')
import functions.exploration_ratio_funcs as exploration_ratio_funcs


def compare_2_arrays(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def custom_add_array_for_data(list_of_states, state):
    state = np.array(state)
    if not np.any(np.array([compare_2_arrays(s, state) for s in list_of_states])):
        list_of_states.append(tuple(state))


def run_wumpus(exploration_ratio_func, learning_rate, discount_factor, num_episodes, train_visual, val_visual):
    """
    Main function that initiates the training of the Wumpus-world benchmark and evaluates the agent's performance
    after training.

    Parameters:
    - exploration_ratio_func (function): The ratio corresponding to the probability of taking a random action during training, based on state, iteration num.
    - learning_rate (float): The learning rate.
    - discount_factor (float): The discount factor.
    - num_episodes (int): Number of episodes during training.
    - train_visual (bool): Boolean indicating whether visual representation of the agent is desired on train.
    - val_visual (bool): Boolean indicating whether visual representation of the agent is desired on val.

    Returns:
    - Agent's performance.
    """
    size = (12, 12)
    entrance = (1, 7)
    heading = 'south'
    wumpus = (5, 3)
    gold = (12, 4)
    pits = [(4, 8), (2, 3), (8, 7), (9, 1), (9, 3), (8, 10), (10, 1), (4, 5), (9, 4), (10, 3), (9, 6), (4, 9)]

    env = gym.make('wumpus-v0', width=size[0], height=size[1], entrance=entrance, heading=heading, wumpus=wumpus,
                   pits=pits, gold=gold)

    num_actions = env.action_space.high[0] + 1
    Q = {}
    list_of_states_for_data = []

    def epsilon_greedy(state, expl):
        if np.random.rand() < expl:
            return np.random.choice(num_actions)
        else:
            return np.argmax(Q[state])

    action_list = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        truncated = False
        action_list.append(-1)

        while not done and not truncated:
            tuple_state = tuple(state)

            if tuple_state not in Q:
                Q[tuple_state] = [0, 0, 0, 0, 0, 0, 0]

            custom_add_array_for_data(list_of_states_for_data, tuple_state)  # Only for testing

            action = epsilon_greedy(tuple_state, exploration_ratio_func(state, episode))
            action_list.append(action)

            next_state, reward, done, truncated = env.step(action)

            if episode % num_episodes // 10 == 0 and train_visual:
                env.render()

            tuple_next_state = tuple(next_state)
            if tuple_next_state not in Q:
                Q[tuple_next_state] = [0, 0, 0, 0, 0, 0, 0]

            Q[tuple_state][action] = Q[tuple_state][action] + learning_rate * (
                    float(reward) + discount_factor * np.max(Q[tuple_next_state]) - Q[tuple_state][action])
            state = next_state

    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        tuple_state = tuple(state)

        if tuple_state not in Q:
            Q[tuple_state] = [0, 0, 0, 0, 0, 0, 0]

        action = np.argmax(Q[tuple_state])
        next_state, reward, done, info = env.step(action)

        if val_visual:
            env.render()

        total_reward += reward
        state = next_state

    print(f"Average reward over {num_episodes} evaluation episodes: {total_reward}")
    print(f"number of different states : {len(list_of_states_for_data)}")
    env.close()

    data = {
        'number_of_states': list_of_states_for_data,
        'actions': action_list,
        'wumpus': wumpus,
        'size': size,
        'entrance': entrance,
        'heading': heading,
        'gold': gold,
        'pits': pits,
    }

    with open('donnees.pkl', 'wb') as f:
        print("test")
        pickle.dump(data, f)

    return total_reward


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Create the parser
    parser = argparse.ArgumentParser(description='Wumpus argument parser')

    # Add arguments
    parser.add_argument('--exploration_ratio_func', type=str,
                        help='Exploration Ratio func name. Must be one of: ["custom", "constant", "linear", '
                             '"logarithmic"]',
                        default="custom")
    parser.add_argument('--constant_x', type=float,
                        help='Exploration ratio func value when exploration_ratio_func="constant"',
                        default=0.5)
    parser.add_argument('--start', type=float,
                        help='Exploration ratio func value at start when exploration_ratio_func="linear" or '
                             '"logarithmic"',
                        default=0.8)
    parser.add_argument('--end', type=float,
                        help='Exploration ratio func value at end when exploration_ratio_func="linear" or "logarithmic"',
                        default=0.3)
    parser.add_argument('--in_x_iteration', type=int,
                        help='Number of iterations to go from start to end when exploration_ratio_func="linear" '
                             '"logarithmic"',
                        default=2000)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.9)
    parser.add_argument('--discount_factor', type=float, help='Discount factor', default=0.99)
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=400)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=1)

    # Parse the command line arguments
    args = parser.parse_args()

    exploration_ratio_func = None  # initiate variable
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episods = args.num_of_episods

    if args.exploration_ratio_func == "custom":
        # Custom part: We can change that as we want for easy testing
        exploration_ratio_func = exploration_ratio_funcs.constant_x(0.1)

        print("Running custom Exploration-ratio-func (supposedly used only for testing purposes)")

    elif args.exploration_ratio_func == "constant":
        exploration_ratio_func = exploration_ratio_funcs.constant_x(args.constant_x)

        print("Running constant func with following params:\n",
              "- Exploration-ratio: " + str(args.constant_x) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episods: " + str(num_of_episods) + "\n"
                                                           "- Num of runs: " + str(args.num_of_runs) + "\n")

    elif args.exploration_ratio_func == "linear":
        exploration_ratio_func = exploration_ratio_funcs.logarithmic(args.start, args.end,
                                                                     args.in_x_iteration)

        print("Running linear func with following params:\n",
              "- start: " + str(args.start) + "\n",
              "- end: " + str(args.end) + "\n",
              "- in_x_iteration: " + str(args.in_x_iteration) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episods: " + str(num_of_episods) + "\n",
              "- Num of runs: " + str(args.num_of_runs) + "\n")
    elif args.exploration_ratio_func == "logarithmic":
        exploration_ratio_func = exploration_ratio_funcs.logarithmic(args.start, args.end,
                                                                     args.in_x_iteration)

        print("Running linear func with following params:\n",
              "- start: " + str(args.start) + "\n",
              "- end: " + str(args.end) + "\n",
              "- in_x_iteration: " + str(args.in_x_iteration) + "\n",
              "- Learning rate: " + str(learning_rate) + "\n",
              "- Discount factor: " + str(discount_factor) + "\n",
              "- Num of episods: " + str(num_of_episods) + "\n",
              "- Num of runs: " + str(args.num_of_runs) + "\n")
    else:
        print("Bad Arguments")
        exit(1)

    total_res = 0
    for j in range(args.num_of_runs):
        total_res += run_wumpus(exploration_ratio_func, learning_rate, discount_factor, num_of_episods, False,
                                True)

    print("total_average_reward : " + str(total_res / args.num_of_runs))
    exit(0)
