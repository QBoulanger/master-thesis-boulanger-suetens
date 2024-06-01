import sys
import gymnasium as gym
import numpy as np
import cv2
import argparse


class TaxiEnv:
    def __init__(self):
        self.env = gym.make('Taxi-v3', render_mode='rgb_array')
        return

    def get_num_of_states(self):
        return self.env.observation_space.n

    def get_num_of_actions(self):
        return self.env.action_space.n

    def get_comparable_representation_of_state(self, state):
        return state

    def get_squared_euclidean_distance(self, state_a, state_b):
        return abs(state_a - state_b)

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
    env = TaxiEnv()

    print("Num of states:", env.get_num_of_states())
    print("Num of actions:", env.get_num_of_actions())
    print("state return after reset: ", env.reset())

    next_state, reward, done, truncated, _ = env.make_step(0)

    print("Next State", next_state)

    print("Comparable Representation Of Next State", env.get_comparable_representation_of_state(next_state))

    print("reward", reward)
