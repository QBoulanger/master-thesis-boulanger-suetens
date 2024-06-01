from mazepy import mazepy

grid=mazepy.Grid(3,3)
grid=mazepy.getRandomMaze(grid)
print("%s Maze:" % grid.algorithm)
print(grid)