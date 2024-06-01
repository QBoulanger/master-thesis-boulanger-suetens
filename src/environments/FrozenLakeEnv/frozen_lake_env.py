import sys
import gymnasium as gym
import numpy as np
import cv2
import argparse

class FrozenLakeEnv:
    def __init__(self, desc=None, map_name="8x8", render_mode='rgb_array',is_slippery=True):
        self.env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, render_mode=render_mode, is_slippery=is_slippery)
        return

    def get_num_of_states(self):
        return self.env.observation_space.n

    def get_num_of_actions(self):
        return self.env.action_space.n

    def get_comparable_representation_of_state(self, state):
        return state

    def get_squared_euclidean_distance(self, state_a, state_b):
        x_a, y_a = state_a // 8, state_a % 8
        x_b, y_b = state_b // 8, state_b % 8

        return (x_a - x_b) ** 2 + (y_a - y_b) ** 2

    def make_step(self, action):
        next_state, reward, done, truncated, _ = self.env.step(action)
        return next_state, reward, done, truncated

    def render(self):
        img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow("test", img)
        cv2.waitKey(150)

    def reset(self):
        reset = self.env.reset()
        return reset[0]

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = FrozenLakeEnv()

    print("Num of states:", env.get_num_of_states())
    print("Num of actions:", env.get_num_of_actions())
    print("state return after reset: ", env.reset())

    next_state, reward, done, truncated, _ = env.make_step(0)

    print("Next State", next_state)

    print("Comparable Representation Of Next State", env.get_comparable_representation_of_state(next_state))

    print("reward", reward)



