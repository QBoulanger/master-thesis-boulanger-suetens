import numpy as np


class GridWorldEnv:
    def __init__(self, grid_size=10, start=(0, 0), goal=(9, 9), treasures=None, traps=None):
        if traps is None:
            traps = [(2, 3), (2, 4), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        if treasures is None:
            treasures = [(1, 2), (3, 1), (4, 7), (7, 4)]
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.grid_treasures_indices = np.zeros((grid_size, grid_size))
        self.start = start
        self.goal = goal
        self.treasures = treasures
        self.traps = traps
        self.set_rewards()
        self.time_out = 100
        self.current_state = start[0], start[1], [1 for _ in range(len(treasures))]

    def get_num_of_states(self):
        return self.grid_size ** 2 + 2 ** (len(self.treasures))

    def get_num_of_actions(self):
        return 4

    def get_comparable_representation_of_state(self, state):
        return tuple([state[0], state[1], tuple(state[2])])

    def get_squared_euclidean_distance(self, state_a, state_b):
        return (state_a[0] - state_b[0]) ** 2 + (state_a[1] - state_b[1]) ** 2

    def set_rewards(self):

        for i, treasure in enumerate(self.treasures):
            self.grid[treasure] = 10
            self.grid_treasures_indices[treasure] = i
        for trap in self.traps:
            self.grid[trap] = -10
        self.grid[self.goal] = 20

    def make_step(self, action):
        self.time_out = self.time_out -1
        x, y, treasure_visited = self.current_state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)

        reward = self.grid[(x, y)]
        if reward == 10:
            self.grid[(x, y)] = 0
            treasure_visited[int(self.grid_treasures_indices[(x, y)])] = 0
        self.current_state = (x, y, treasure_visited)
        return self.current_state, reward - 1, (x, y) == self.goal, self.time_out <= 0

    def render(self):
        gridCopy = np.copy(self.grid)
        gridCopy[(self.current_state[0], self.current_state[1])] = 1
        gridCopy[self.goal] = 2
        print(gridCopy)

    def reset(self):
        self.time_out = 100
        self.set_rewards()

        x, y = self.start
        self.current_state = x, y, [1 for _ in range(len(self.treasures))]

        return self.current_state

    def close(self):
        return None


if __name__ == "__main__":
    env = GridWorldEnv()

    print("Num of states:", env.get_num_of_states())
    print("Num of actions:", env.get_num_of_actions())
    print("state return after reset: ", env.reset())

    next_state, reward, done, truncated = env.make_step(1)
    next_state, reward, done, truncated = env.make_step(1)

    print("Next State", next_state)

    print("Comparable Representation Of Next State", env.get_comparable_representation_of_state(next_state))

    print("reward", reward)



