import gymnasium as gym
import numpy as np
import cv2
import sys
import argparse

sys.path.append('../../')
import functions.exploration_ratio_funcs as exploration_ratio_funcs


def custom_add_array_for_data(list_of_states, state):
    if state not in list_of_states:
        list_of_states.append(state)


def show_visual(env):
    """
    This function is used to visualize the current environment using Pygame.

    Parameters:
    - env : The environment to be visualized.
    """
    img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    cv2.imshow("test", img)
    cv2.waitKey(150)


def run_taxi(exploration_ratio_func, learning_rate, discount_factor, num_episodes, train_visual, val_visual):
    """
    Main function that initiates the training of the Taxi benchmark and evaluates the agent's performance
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
    env = gym.make("Taxi-v3", render_mode='rgb_array', )

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    list_of_states_for_data = []

    def epsilon_greedy(state, expl):
        if np.random.rand() < expl:
            # print(np.random.choice(num_actions))
            return np.random.choice(num_actions)
        else:
            #   print(np.argmax(Q[state, :]))
            return np.argmax(Q[state, :])

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False

        # print("episode : ", episode)
        while not done and not truncated:

            custom_add_array_for_data(list_of_states_for_data, state)  # Only for testing

            action = epsilon_greedy(state, exploration_ratio_func(state, episode))

            next_state, reward, done, truncated, _ = env.step(action)

            if episode % num_episodes // 10 == 0 and val_visual:
                show_visual(env)

            Q[state][action] = Q[state][action] + learning_rate * (
                    float(reward) + discount_factor * np.max(Q[next_state]) - Q[state][action])
            state = next_state

    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        while not done and not truncated:
            action = np.argmax(Q[state, :])
            next_state, reward, done, truncated, _ = env.step(action)

            if episode % num_episodes // 10 == 0 and train_visual:
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
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.1)
    parser.add_argument('--discount_factor', type=float, help='Discount factor', default=0.99)
    parser.add_argument('--num_of_episods', type=int, help='Num of Episods', default=1000)
    parser.add_argument('--num_of_runs', type=int, help='Num of Runs (the average is taken from them)', default=2)

    # Parse the command line arguments
    args = parser.parse_args()

    exploration_ratio_func = None  # initiate variable
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    num_of_episods = args.num_of_episods

    if args.exploration_ratio_func == "custom":
        # Custom part: We can change that as we want for easy testing
        exploration_ratio_func = exploration_ratio_funcs.constant_x(5)

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
        total_res += run_taxi(exploration_ratio_func, learning_rate, discount_factor, num_of_episods, False,
                              False)

    print("total_average_reward : " + str(total_res / args.num_of_runs))
    exit(0)
