import pygame
import numpy as np
import pickle

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

start_pos = list((data['entrance'][0], Y - 1 - data['entrance'][1]))
wumpus_pos = (data['wumpus'][0], Y - 1 - data['wumpus'][1])
gold_pos = (data['gold'][0], Y - 1 - data['gold'][1])
pit_pos = data['pits']

WINDOW_SIZE = (1000, 1000)
GRID_COLOR = (0, 0, 0)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

walls = set()
for i in range(X):
    walls.add((i, 0))
    walls.add((i, Y - 1))
for j in range(Y):
    walls.add((0, j))
    walls.add((X - 1, j))

new_pit_pos = []
for coord in pit_pos:
    new_pit_pos.append((coord[0], 13 - coord[1]))

pit_pos = new_pit_pos

grid = np.zeros((X, Y))

pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
clock = pygame.time.Clock()

cell_width = WINDOW_SIZE[0] // X
cell_height = WINDOW_SIZE[1] // Y
font_size = min(cell_width, cell_height)
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)
point_radius = min(cell_width, cell_height) // 5


def draw_grid():
    for i in range(X):
        for j in range(Y):
            brightness = 255 - min(255, grid[i][j])
            color = (brightness, brightness, brightness)
            pygame.draw.rect(screen, color, (i * cell_width, j * cell_height, cell_width, cell_height), 0)

            if (i, j) == wumpus_pos:
                text_surface = bold_font.render('W', True, RED)
                text_rect = text_surface.get_rect(center=(i * cell_width + cell_width // 2,
                                                          j * cell_height + cell_height // 2))
                screen.blit(text_surface, text_rect)
            elif (i, j) == gold_pos:
                text_surface = bold_font.render('G', True, ORANGE)
                text_rect = text_surface.get_rect(center=(i * cell_width + cell_width // 2,
                                                          j * cell_height + cell_height // 2))
                screen.blit(text_surface, text_rect)
            elif (i, j) in pit_pos:
                text_surface = bold_font.render('P', True, GREEN)
                text_rect = text_surface.get_rect(center=(i * cell_width + cell_width // 2,
                                                          j * cell_height + cell_height // 2))
                screen.blit(text_surface, text_rect)
            if (i, j) in walls:
                pygame.draw.rect(screen, BLUE, (i * cell_width, j * cell_height, cell_width, cell_height), 0)

            if i == start_pos[0] and j == start_pos[1]:
                pygame.draw.circle(screen, RED,
                                   (i * cell_width + cell_width // 2, j * cell_height + cell_height // 2), point_radius)

    for i in range(X + 1):
        pygame.draw.line(screen, GRID_COLOR, (i * cell_width, 0), (i * cell_width, WINDOW_SIZE[1]))

    for j in range(Y + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, j * cell_height), (WINDOW_SIZE[0], j * cell_height))


def run_simulation(actions, speed, heading):
    current_direction = headings[heading]
    current_frame = 0
    running = True
    grid[start_pos[0]][start_pos[1]] += 1
    while running:
        screen.fill(WHITE)
        draw_grid()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

        clock.tick(speed)

    pygame.quit()


run_simulation(actions, 0, heading)
