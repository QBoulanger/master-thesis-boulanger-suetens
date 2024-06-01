import gym_wumpus
import gym as gym
import numpy as np
import sys
import argparse
import pickle
import warnings
import pickle
import math
import heapq


# Environement PARAMETERS
size = (12, 12)
entrance = (1, 7)
heading = 'south'
wumpus = (5, 3)
gold = (12, 4)
pits = [(4, 8), (2, 3), (8, 7), (9, 1), (9, 3), (8, 10), (10, 1), (4, 5), (9, 4), (10, 3), (9, 6), (4, 9)]

# NGU PARAMETERS
beta = 0.3
N = 2
k = 6
epsilon = 0.001
c = 0.001


def squared_euclidean_distance(p, q):
    return np.sum((p - q) ** 2)


def compare_2_arrays(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def add_to_memory(M_memory, state):
    state = np.array(state)
    # if not np.any(np.array([compare_2_arrays(s, state) for s in M_memory])):
    M_memory.append(state)

    # if len(M_memory) > a:
    #    del M_memory[0]


def custom_add_array_for_data(list_of_states, state):
    state = np.array(state)
    if not np.any(np.array([compare_2_arrays(s, state) for s in list_of_states])):
        list_of_states.append(state)


def kernel_function(squared_distance, running_average):
    return epsilon / ((squared_distance / running_average) + epsilon)


accumulator = 0
count = 0


def get_intrinsic_reward(M_matrix, state):
    distances = [squared_euclidean_distance(point, state) for point in M_matrix]

    if len(distances) >= k:
        kth_nearest_distances = heapq.nsmallest(k, distances)
        kth_nearest_distance = kth_nearest_distances[-1]
    else:
        kth_nearest_distances = distances
        if len(kth_nearest_distances) == 0:
            kth_nearest_distance = 0
        else:
            kth_nearest_distance = max(kth_nearest_distances)

    global accumulator, count

    if len(distances) != 0:
        accumulator += kth_nearest_distance
        count += 1
    running_average = accumulator / count

    sum_of_K = 0
    for dist in kth_nearest_distances:
        sum_of_K += kernel_function(dist, running_average)

    reward = 1 / math.sqrt(sum_of_K + c)
    return reward


def run_wumpus(learning_rate, discount_factor, num_episodes, train_visual, val_visual, a):
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
    global accumulator, count
    global size, entrance, heading, wumpus, gold, pits
    global beta, N, k, epsilon, c

    betas_i = []
    if N > 2:
        betas_i.append(0)
        for i in range(1, N - 1):
            betas_i.append(beta * (1 / (1 + np.exp(-(10 * (2 * i - (N - 2)) / (N - 2))))))
        betas_i.append(beta)
    else:
        betas_i = [0, beta]

    # Environments
    envs = [gym.make('wumpus-v0', width=size[0], height=size[1], entrance=entrance, heading=heading, wumpus=wumpus,
                     pits=pits, gold=gold) for _ in range(N)]

    # Models Tables and variables
    UVFA_Q_TABLE = {beta: {} for beta in betas_i}

    M_Memory = []

    # Debugging variables
    action_list = []
    list_of_states_for_data = []

    for episode in range(num_episodes):

        M_Memory = []
        # accumulator = 0
        # count = 0

        for n in range(N):
            beta_i = betas_i[len(betas_i) - N + n]
            state = envs[n].reset()
            done = False
            truncated = False
            # action_list.append(-1)
            action_list = []

            while not done and not truncated:
                tuple_state = tuple(state)

                for beta_i_prime in betas_i:
                    if tuple_state not in UVFA_Q_TABLE[beta_i_prime]:
                        UVFA_Q_TABLE[beta_i_prime][tuple_state] = [0, 0, 0, 0, 0, 0, 0]

                add_to_memory(M_Memory, tuple_state)
                # custom_add_array_for_data(list_of_states_for_data, tuple_state)  # Only for testing

                action = np.argmax(UVFA_Q_TABLE[beta_i][tuple_state])
                action_list.append(action)

                next_state, reward, done, truncated = envs[n].step(action)

                if n == N - 1 and episode % num_episodes // 10 == 0 and train_visual:
                    envs[n].render()

                tuple_next_state = tuple(next_state)

                for beta_i_prime in betas_i:
                    if tuple_next_state not in UVFA_Q_TABLE[beta_i_prime]:
                        UVFA_Q_TABLE[beta_i_prime][tuple_next_state] = [0, 0, 0, 0, 0, 0, 0]

                intrinsic_reward = get_intrinsic_reward(M_Memory, tuple_next_state)
                # print(intrinsic_reward)
                for beta_i_prime in betas_i:
                    augmented_reward = reward + beta_i_prime * intrinsic_reward

                    UVFA_Q_TABLE[beta_i_prime][tuple_state][action] = UVFA_Q_TABLE[beta_i_prime][tuple_state][
                                                                          action] + learning_rate * (
                                                                              float(
                                                                                  augmented_reward) + discount_factor * np.max(
                                                                          UVFA_Q_TABLE[beta_i_prime][
                                                                              tuple_next_state]) -
                                                                              UVFA_Q_TABLE[beta_i_prime][tuple_state][
                                                                                  action])
                state = next_state

        print(M_Memory)

    total_reward = 0
    state = envs[0].reset()
    done = False
    while not done:
        tuple_state = tuple(state)

        if tuple_state not in UVFA_Q_TABLE[0]:
            UVFA_Q_TABLE[0][tuple_state] = [0, 0, 0, 0, 0, 0, 0]
        action = np.argmax(UVFA_Q_TABLE[0][tuple_state])
        next_state, reward, done, info = envs[0].step(action)

        if val_visual:
            envs[0].render()

        total_reward += reward
        state = next_state

    print(f"Average reward over {num_episodes} evaluation episodes: {total_reward}")
    print(f"number of different states : {len(list_of_states_for_data)}")

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

    for env in envs:
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
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.9)
    parser.add_argument('--discount_factor', type=float, help='Discount factor', default=0.99)
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=66)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=1)
    parser.add_argument('--remember_deep', type=int, help='Num of Runs (the average is taken from them)', default=14)

    # Parse the command line arguments
    args = parser.parse_args()

    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episods = args.num_of_episods
    remember_deep = args.remember_deep

    print("Running constant func with following params:\n",
          "- Learning rate: " + str(learning_rate) + "\n",
          "- Discount factor: " + str(discount_factor) + "\n",
          "- Num of episods: " + str(num_of_episods) + "\n",
          "- Num of runs: " + str(args.num_of_runs) + "\n",
          "- remember_deep: " + str(remember_deep) + "\n"
          )

    total_res = 0
    for j in range(args.num_of_runs):
        total_res += run_wumpus(learning_rate, discount_factor, num_of_episods, False, True, remember_deep)

    print("total_average_reward : " + str(total_res / args.num_of_runs))

    exit(0)
