import sys
import gymnasium as gym
import numpy as np
import cv2
import argparse

sys.path.append('../../')
import functions.exploration_ratio_funcs as exploration_ratio_funcs


def custom_distance(a, b):
    return abs(b - a)


def custom_add_array(list_of_states, state, a):
    if state not in list_of_states:
        list_of_states.append(state)
    if len(list_of_states) > a:
        del list_of_states[0]


def custom_add_array_for_data(list_of_states, state):
    if state not in list_of_states:
        list_of_states.append(state)


def greedy_function(list_of_states, state, b):
    min_distance = 0
    for i in list_of_states:
        distance = custom_distance(state, i)
        min_distance = min(min_distance, distance)

    if min_distance == 0:
        min_distance = -0.1

    reward = min_distance / b
    return reward


def show_visual(env):
    """
    This function is used to visualize the current environment using Pygame.

    Parameters:
    - env : The environment to be visualized.
    """
    img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    cv2.imshow("test", img)
    cv2.waitKey(150)


def run_frozen_lake(learning_rate, discount_factor, num_episodes, train_visual, val_visual, a, b):
    """
    Main function that initiates the training of the FrozenLake benchmark and evaluates the agent's performance
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
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", render_mode='rgb_array', is_slippery=True)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    list_of_states = []
    list_of_states_for_data = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False

        while not done and not truncated:
            custom_add_array(list_of_states, state, a)
            custom_add_array_for_data(list_of_states_for_data, state)  # Only for testing

            action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)

            if episode % num_episodes // 10 == 0 and val_visual:
                show_visual(env)

            reward += greedy_function(list_of_states, next_state, b)

            Q[state][action] = Q[state][action] + learning_rate * (
                    float(reward) + discount_factor * np.max(Q[next_state]) - Q[state][action])
            state = next_state

    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        while not done and not truncated:
            action = np.argmax(Q[state, :])
            next_state, reward, done, truncated, _ = env.step(action)

            if val_visual:
                show_visual(env)

            total_reward += reward
            state = next_state

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} evaluation episodes: {average_reward}")
    print(f"number of different states : {len(list_of_states_for_data)}")
    env.close()
    return average_reward


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Taxi argument parser')

    # Add arguments
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.1)
    parser.add_argument('--discount_factor', type=float, help='Discount factor', default=0.99)
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=5000)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=2)
    parser.add_argument('--remember_deep', type=int, help='Num of Runs (the average is taken from them)', default=2)
    parser.add_argument('--reward_division', type=int, help='Num of Runs (the average is taken from them)', default=9)

    # Parse the command line arguments
    args = parser.parse_args()

    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episods = args.num_of_episods
    remember_deep = args.remember_deep
    reward_division = args.reward_division

    print("Running constant func with following params:\n",
          "- Learning rate: " + str(learning_rate) + "\n",
          "- Discount factor: " + str(discount_factor) + "\n",
          "- Num of episods: " + str(num_of_episods) + "\n",
          "- Num of runs: " + str(args.num_of_runs) + "\n",
          "- remember_deep: " + str(remember_deep) + "\n",
          "- reward_division: " + str(reward_division) + "\n"
          )

    total_res = 0
    for j in range(args.num_of_runs):
        total_res += run_frozen_lake(learning_rate, discount_factor, num_of_episods, False, False, remember_deep,
                                     reward_division)

    print("total_average_reward : " + str(total_res / args.num_of_runs))
    exit(0)
