import matplotlib
matplotlib.use('tkAgg')
from pylab import *


class Environment():
    ProbabilityOfBlockedMaze = 0.4
    DimensionOfMaze = 10

    def __init__(self):
        self.maze = None
        self.maze_copy = None

    def generate_maze(self, n = DimensionOfMaze, p = ProbabilityOfBlockedMaze):
        self.n = n
        self.p = p

        self.maze = np.array([list(np.random.binomial(1, 1 - p, n)) for i in range(n)])
        self.maze[0, 0] = 1
        self.maze[n-1, n-1] = 1

        # Create a copy of maze to render and update
        self.maze_copy = self.maze.copy()

    def display_maze(self):
        plt.pcolormesh(self.maze, edgecolor = 'k', linewidth = 0.5, antialiased = False)
        plt.axes().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render_maze(self):
        plt.pcolormesh(self.maze_copy, edgecolor = 'k', linewidth = 0.5, antialiased = False)
        plt.axes().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.show()
        plt.pause(0.1)

    def update_color_of_cell(self, row, column):
        self.maze_copy[self.n - 1 - row, column] = 10

    def reset_color_of_cell(self, row, column):
        self.maze_copy[self.n - 1 - row, column] = self.maze[row, column]

