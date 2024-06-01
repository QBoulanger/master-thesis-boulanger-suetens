import gym_wumpus
import gym as gym
import numpy as np
import sys
import argparse
import pickle
import warnings
import pickle

sys.path.append('../../')
import functions.exploration_ratio_funcs as exploration_ratio_funcs


def custom_distance(a, b):
    distance = 0
    for i in range(3):
        if a[i] >= b[i]:
            distance += a[i] - b[i]
        else:
            distance += b[i] - a[i]
    return distance


def compare_2_arrays(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def custom_add_array(list_of_states, state, a):
    state = np.array(state)
    if not np.any(np.array([compare_2_arrays(s, state) for s in list_of_states])):
        list_of_states.append(tuple(state))
    if len(list_of_states) > a:
        del list_of_states[0]


def custom_add_array_for_data(list_of_states, state):
    state = np.array(state)
    if not np.any(np.array([compare_2_arrays(s, state) for s in list_of_states])):
        list_of_states.append(tuple(state))


def greedy_function(list_of_states, state):
    min_distance = 0
    for i in list_of_states:
        min_distance += custom_distance(state, i)

    reward = min_distance
    return -1 / reward


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
    size = (12, 12)
    entrance = (1, 7)
    heading = 'south'
    wumpus = (5, 3)

    gold = (12, 4)
    pits = [(4, 8), (2, 3), (8, 7), (9, 1), (9, 3), (8, 10), (10, 1), (4, 5), (9, 4), (10, 3), (9, 6), (4, 9)]

    # def __init__(self, width=4, height=4, entrance=(1, 1), heading='north',
    #              wumpus=(1, 3), pits=((3, 3), (3, 1)), gold=(2, 3),
    #              modify_reward=True, stochastic_action_prob=1.0):
    # env = gym.make('wumpus-v0', width=10, height=10, entrance=(1, 10), heading='south', wumpus=(7, 5),
    #                pits=((3, 9), (6, 9), (8, 8), (5, 7), (2, 6), (9, 6), (4, 5), (2, 3), (9, 3), (4, 2)), gold=(7, 2))

    env = gym.make('wumpus-v0', width=size[0], height=size[1], entrance=entrance, heading=heading, wumpus=wumpus,
                   pits=pits, gold=gold)

    Q = {}

    list_of_states = []
    list_of_states_for_data = []
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

            custom_add_array(list_of_states, tuple_state, a)
            custom_add_array_for_data(list_of_states_for_data, tuple_state)  # Only for testing

            action = np.argmax(Q[tuple_state])
            action_list.append(action)

            next_state, reward, done, truncated = env.step(action)

            if episode % num_episodes // 10 == 0 and train_visual:
                env.render()

            tuple_next_state = tuple(next_state)
            if tuple_next_state not in Q:
                Q[tuple_next_state] = [0, 0, 0, 0, 0, 0, 0]

            intrinsic_reward = greedy_function(list_of_states, tuple_next_state)
            # print(intrinsic_reward)
            reward += intrinsic_reward
            Q[tuple_state][action] = Q[tuple_state][action] + learning_rate * (
                    float(reward) + discount_factor * np.max(Q[tuple_next_state]) - Q[tuple_state][action])
            state = next_state

    total_reward = 0
    state = env.reset()
    done = False
    print(Q)
    while not done:
        tuple_state = tuple(state)

        if tuple_state not in Q:
            Q[tuple_state] = [0, 0, 0, 0, 0, 0, 0]

        print(state)
        action = np.argmax(Q[tuple_state])
        next_state, reward, done, info = env.step(action)

        if val_visual:
            env.render()

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
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=400)
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
