import math
import sys
import gym_wumpus
import gym as gym
import numpy as np
import argparse
import pickle
import warnings
import random
import copy

sys.path.append('../../')
import functions.exploration_ratio_funcs as exploration_ratio_funcs


def compare_2_arrays(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def custom_add_replay(replay_buffer, transition, max_size):
    replay_buffer.append(transition)
    if len(replay_buffer) > max_size:
        del replay_buffer[0]


def custom_add_array_for_data(list_of_states, state):
    state = np.array(state)
    if not np.any(np.array([compare_2_arrays(s, state) for s in list_of_states])):
        list_of_states.append(tuple(state))

def sample_n(mu_t, sigma, n):
    samples = [np.random.normal(mu_t, sigma) for _ in range(n)]
    return np.clip(samples, 0, 1).tolist()

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

    mu_t = 0.5
    sigma = 0.2
    n = 10
    beta = 0.0001

    envs = [gym.make('wumpus-v0', width=size[0], height=size[1], entrance=entrance, heading=heading, wumpus=wumpus,
                   pits=pits, gold=gold) for _ in range(n) ]

    num_actions = envs[0].action_space.high[0] + 1

    Qs = [{} for _ in range(n)]

    list_of_states_for_data = []

    replay_buffer = []


    def epsilon_greedy(state, Q, expl):
        if np.random.rand() < expl:
            return np.random.choice(num_actions)
        else:
            return np.argmax(Q[state])

    action_list = [[] for _ in range(n) ]

    for episode in range(num_episodes):
        states = [envs[i].reset() for i in range(n) ]

        print(mu_t)

        for i in range(n):
            action_list[i].append(-1)

        n_hyperparameters = sample_n(mu_t, sigma, n)

        # print(n_hyperparameters)

        Ltjs = [0 for _ in range(n)]

        for i in range(n):
            done = False
            truncated = False

            envs[i].reset()

            action_list[i].append(-1)

            while not done and not truncated:

                tuple_state = tuple(states[i])

                if tuple_state not in Qs[i]:
                    Qs[i][tuple_state] = [0, 0, 0, 0, 0, 0, 0]

                # custom_add_array_for_data(list_of_states_for_data, tuple_state)  # Only for testing

                action = epsilon_greedy(tuple_state, Qs[i], n_hyperparameters[i])
                action_list[i].append(action)

                next_state, reward, done, truncated = envs[i].step(action)

                if episode % num_episodes // 10 == 0 and i == 0 and train_visual:
                    envs[0].render()

                tuple_next_state = tuple(next_state)
                if tuple_next_state not in Qs[i]:
                    Qs[i][tuple_next_state] = [0, 0, 0, 0, 0, 0, 0]

                transition = (tuple_state, action, reward, tuple_next_state)
                custom_add_replay(replay_buffer, transition, 100)

                if len(replay_buffer) >= 10:
                    transitions = random.sample(replay_buffer, 10)
                else:
                    transitions = [transition]

                for current_transition in transitions:
                    current_state = current_transition[0]
                    current_action = current_transition[1]
                    current_reward = current_transition[2]
                    current_next_state = current_transition[3]

                    if tuple(current_state) not in Qs[i]:
                        Qs[i][tuple(current_state)] = [0, 0, 0, 0, 0, 0, 0]
                    if tuple(current_next_state) not in Qs[i]:
                        Qs[i][tuple(current_next_state)] = [0, 0, 0, 0, 0, 0, 0]

                    Qs[i][current_state][current_action] = Qs[i][current_state][current_action] + learning_rate * (
                            float(current_reward) + discount_factor * np.max(Qs[i][current_next_state]) - Qs[i][current_state][current_action])

                states[i] = next_state

            Ltjs[i] = 0
            state = envs[i].reset()
            done = False
            t = 0
            while not done:

                tuple_state = tuple(state)

                if tuple_state not in Qs[i]:
                    Qs[i][tuple_state] = [0, 0, 0, 0, 0, 0, 0]

                action = np.argmax(Qs[i][tuple_state])
                next_state, reward, done, info = envs[i].step(action)

                if(reward == 500):
                    print("YEEES: " + str(episode))

                Ltjs[i] += reward * (math.pow(discount_factor, t))
                state = next_state
                t += 1

        cummulative_part = 0
        best_arg = 0
        best_Ltjs = Ltjs[0]
        for i in range(n):
            cummulative_part += Ltjs[i] * (n_hyperparameters[i] - mu_t)
            if(Ltjs[i] > best_Ltjs):
                best_Ltjs = Ltjs[i]
                best_arg = i


        mu_t = mu_t + (beta /(sigma * n)) * cummulative_part
        mu_t = np.clip(mu_t, 0, 1)

        for i in range(0, n):
            Qs[i] = copy.deepcopy(Qs[best_arg])

    total_reward = 0
    state = envs[0].reset()
    done = False
    while not done:
        tuple_state = tuple(state)

        if tuple_state not in Qs[0]:
            Qs[0][tuple_state] = [0, 0, 0, 0, 0, 0, 0]

        action = np.argmax(Qs[0][tuple_state])
        next_state, reward, done, info = envs[0].step(action)

        if val_visual:
            envs[0].render()

        total_reward += reward
        state = next_state

    print(f"Average reward over {num_episodes} evaluation episodes: {total_reward}")
    print(f"number of different states : {len(list_of_states_for_data)}")
    envs[0].close()

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
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=500)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=1)

    # Parse the command line arguments
    args = parser.parse_args()

    exploration_ratio_func = None  # initiate variable
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episods = args.num_of_episods

    if args.exploration_ratio_func == "custom":
        # Custom part: We can change that as we want for easy testing
        exploration_ratio_func = exploration_ratio_funcs.constant_x(0.2)

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
