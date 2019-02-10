import matplotlib
matplotlib.use('tkAgg')
from pylab import *
from matplotlib import colors


class Environment():
    ProbabilityOfBlockedMaze = 0.4
    DimensionOfMaze = 10

    def __init__(self):
        self.maze = None
        self.maze_copy = None

        # The default colormap of our maze - 0: Black, 1: White
        self.cmap = colors.ListedColormap(['black', 'white'])

    def generate_maze(self, n = DimensionOfMaze, p = ProbabilityOfBlockedMaze):
        self.n = n
        self.p = p

        self.maze = np.array([list(np.random.binomial(1, 1 - p, n)) for _ in range(n)])
        self.maze[0, 0] = 1
        self.maze[n-1, n-1] = 1

        # Create a copy of maze to render and update
        self.maze_copy = self.maze.copy()

    def display_maze(self):
        plt.pcolormesh(self.maze, cmap = self.cmap, edgecolor = 'k', linewidth = 0.5, antialiased = False)
        plt.axes().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render_maze(self):
        # Create a mask for the particular cell and change its color to green
        masked_maze_copy = np.ma.masked_where(self.maze_copy == -1, self.maze_copy)
        self.cmap.set_bad(color = 'green')

        # Plot the new maze
        plt.pcolormesh(masked_maze_copy, cmap = self.cmap, edgecolor = 'k', linewidth = 0.5, antialiased = False)
        plt.axes().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.show()
        plt.pause(0.000000002)

    def update_color_of_cell(self, row, column):
        self.maze_copy[row, column] = -1

    def reset_color_of_cell(self, row, column):
        self.maze_copy[row, column] = self.maze[row, column]

