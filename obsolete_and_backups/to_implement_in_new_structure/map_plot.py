import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

headings = {
    "south": 270,
    "north": 90,
    "east": 0,
    "west": 180
}

with open('donnees.pkl', 'rb') as f:
    data = pickle.load(f)

number_of_states = data['number_of_states']
actions = data['actions']
heading = data['heading']

X, Y = data['size'][0] + 2, data['size'][1] + 2

start_pos = list((data['entrance'][0], data['entrance'][1]))

walls = set()
for i in range(X):
    walls.add((i, 0))
    walls.add((i, Y - 1))
for j in range(Y):
    walls.add((0, j))
    walls.add((X - 1, j))

grid = np.zeros((X, Y))


def run_simulation(actions, heading):
    current_direction = headings[heading]
    current_frame = 0
    running = True
    grid[start_pos[0]][start_pos[1]] += 1
    while running:
        if current_frame < len(actions):
            action = actions[current_frame]

            dx, dy = 0, 0

            if action == 0:
                current_direction = (current_direction - 90) % 360
            elif action == 1:
                current_direction = (current_direction + 90) % 360
            elif action == 2:
                if current_direction == 90:
                    dy = -1
                elif current_direction == 270:
                    dy = 1
                elif current_direction == 0:
                    dx = 1
                elif current_direction == 180:
                    dx = -1
            elif action == -1:
                start_pos[0] = data['entrance'][0]
                start_pos[1] = data['entrance'][1]

            new_x = max(0, min(start_pos[0] + dx, X - 1))
            new_y = max(0, min(start_pos[1] + dy, Y - 1))
            if (new_x, new_y) not in walls:
                start_pos[0] = new_x
                start_pos[1] = new_y
            grid[start_pos[0]][start_pos[1]] += 1
            current_frame += 1

        else:
            running = False


run_simulation(actions, heading)
# grid = np.array([row[::-1] for row in grid])
grid = np.transpose(grid)
grid = grid[::-1]
grid = grid[1:-1, 1:-1]

start_pos = list((- 1 + data['entrance'][0], Y - 2 - data['entrance'][1]))
wumpus_pos = (-1 + data['wumpus'][0], Y - 2 - data['wumpus'][1])
gold_pos = (-1 + data['gold'][0], Y - 2 - data['gold'][1])
pit_pos = data['pits']
print(start_pos)
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()

plt.scatter(wumpus_pos[0], wumpus_pos[1], color='red')
plt.scatter(gold_pos[0], gold_pos[1], color='orange')
plt.scatter(start_pos[0], start_pos[1], color='green')
for pos in pit_pos:
    plt.scatter(-1 + pos[0], Y - 2 - pos[1], color='blue')

plt.show()
