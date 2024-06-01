import gymnasium as gym
import numpy as np
import cv2


def show_visual(env):
    """
    This function is used to visualize the current environment using Pygame.

    Parameters:
    - env : The environment to be visualized.
    """
    img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    cv2.imshow("test", img)
    cv2.waitKey(150)


def run_blackjack(exploration_ratio, learning_rate, discount_factor, num_episodes, visual):
    """
    Main function that initiates the training of the FrozenLake benchmark and evaluates the agent's performance
    after training.

    Parameters:
    - exploration_ratio (float): The ratio corresponding to the probability of taking a random action during training.
    - learning_rate (float): The learning rate.
    - discount_factor (float): The discount factor.
    - num_episodes (int): Number of episodes during training.
    - visual (bool): Boolean indicating whether visual representation of the agent is desired.

    Returns:
    - Agent's performance.
    """
    env = gym.make("Blackjack-v1", render_mode='rgb_array')

    num_actions = env.action_space.n

    Q = {}

    def epsilon_greedy(state, expl):
        if np.random.rand() < expl:
            return np.random.choice(num_actions)
        else:
            return np.argmax(Q[state])

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        exploration_prob = 1

        # print("episode : ", episode)
        while not done and not truncated:
            if state not in Q:
                Q[state] = [0, 0]

            action = epsilon_greedy(state, exploration_prob * exploration_ratio)

            next_state, reward, done, truncated, _ = env.step(action)

            if episode % num_episodes // 10 == 0 and visual:
                show_visual(env)

            if next_state not in Q:
                Q[next_state] = [0, 0]

            # env.render()
            Q[state][action] = Q[state][action] + learning_rate * (
                    float(reward) + discount_factor * np.max(Q[next_state]) - Q[state][action])
            state = next_state

    total_reward = 0
    for _ in range(10000):
        state = env.reset()[0]
        done = False
        truncated = False
        while not done and not truncated:
            if state not in Q:
                Q[state] = [0, 0]

            action = np.argmax(Q[state])
            next_state, reward, done, truncated, _ = env.step(action)

            # show_visual(env)

            total_reward += reward

            state = next_state

    average_reward = total_reward / 10000
    print(f"Average reward over {10000} evaluation episodes: {average_reward}")
    env.close()
    return average_reward


if __name__ == '__main__':
    run_blackjack(0.7, 0.1, 0.99, 100000, False)
    exit(0)
