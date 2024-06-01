import sys
import gym_wumpus
import gym as gym
import numpy as np
import cv2
import argparse


class WumpusEnv:
    def __init__(self,
                 size=(12, 12),
                 entrance=(1, 7),
                 heading='south',
                 wumpus=(5, 3),
                 gold=(12, 4),
                 pits=[(4, 8), (2, 3), (8, 7), (9, 1), (9, 3), (8, 10), (10, 1), (4, 5), (9, 4), (10, 3), (9, 6),
                       (4, 9)]
                 ):
        self.env = gym.make('wumpus-v0', width=size[0], height=size[1], entrance=entrance, heading=heading,
                            wumpus=wumpus,
                            pits=pits, gold=gold)
        return

    def get_num_of_states(self):
        return -1

    def get_num_of_actions(self):
        return self.env.action_space.high[0] + 1

    def get_comparable_representation_of_state(self, state):
        return tuple(state)

    def get_squared_euclidean_distance(self, state_a, state_b):
        return np.sum((np.array(state_a) - np.array(state_b)) ** 2)

    def make_step(self, action):
        next_state, reward, done, truncated = self.env.step(action)
        return next_state, reward, done, truncated

    def render(self):
        self.env.render()

    def reset(self):
        reset = self.env.reset()
        return reset

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = WumpusEnv()

    print("Num of states:", env.get_num_of_states())
    print("Num of actions:", env.get_num_of_actions())
    print("state return after reset: ", env.reset())

    next_state, reward, done, truncated = env.make_step(0)

    print("Next State", next_state)

    print("Comparable Representation Of Next State", env.get_comparable_representation_of_state(next_state))

    print("reward", reward)
