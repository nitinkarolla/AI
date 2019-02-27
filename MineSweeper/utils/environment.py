import matplotlib
matplotlib.use('tkAgg')
from pylab import *
from utils.graph import Graph
from scipy.signal import convolve2d
from matplotlib.patches import RegularPolygon


class Environment():
    DimensionOfMaze = 8
    NumberOfMines = 10
    CoveredColor = '#DDDDDD'
    EdgeColor = '#888888'
    UncoveredColor = '#AAAAAA'
    CountColors = ['none', 'blue', 'green', 'red', 'darkblue', 'darkred', 'darkgreen', 'black', 'black']
    FlagVertices = np.array([[0.25, 0.2], [0.25, 0.8],
                            [0.75, 0.65], [0.25, 0.5]])

    def __init__(self,
                 n = DimensionOfMaze,
                 number_of_mines = NumberOfMines):
        self.n = n
        self.number_of_mines = number_of_mines
        self.mines = np.zeros((self.n, self.n), dtype = bool)
        self.clicked = np.zeros((self.n, self.n), dtype = bool)
        self.flags = np.zeros((self.n, self.n), dtype = object)

    def _place_mines(self, row, column):

        # randomly place mines on a grid, but not on space (i, j)
        idx = np.concatenate([np.arange(row * self.n + column), np.arange(row * self.n + column + 1, self.n * self.n)])
        np.random.shuffle(idx)
        self.mines.flat[idx[:self.number_of_mines]] = 1

        # count the number of mines bordering each square
        self.mine_env = convolve2d(self.mines.astype(complex), np.ones((3, 3)), mode = 'same').real.astype(int)

    def _reveal_unmarked_mines(self):
        for (row, column) in zip(*np.where(self.mines & ~self.flags.astype(bool))):
            self._draw_mine(row, column)

    def _draw_mine(self, row, column):
        self.ax.add_patch(plt.Circle((row + 0.5, column + 0.5), radius = 0.25, ec = 'black', fc = 'black'))

    def _draw_red_X(self, row, column):
        self.ax.text(x = row + 0.5, y = column + 0.5, s = 'X', color = 'r', fontsize = 20, ha = 'center', va = 'center')

    def _cross_out_wrong_flags(self):
        for (row, column) in zip(*np.where(~self.mines & self.flags.astype(bool))):
            self._draw_red_X(row, column)

    def _mark_remaining_mines(self):
        for (row, column) in zip(*np.where(self.mines & ~self.flags.astype(bool))):
            self.add_mine_flag(row, column)

    def generate_environment(self):

        # Create the figure and axes
        self.fig = plt.figure(figsize = ((self.n + 2) / 3., (self.n + 2) / 3.))
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
                                    aspect = 'equal',
                                    frameon = False,
                                    xlim = (-0.05, self.n + 0.05),
                                    ylim = (-0.05, self.n + 0.05))
        for axis in (self.ax.xaxis, self.ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())

        # Create the grid of squares
        self.squares = np.array([[RegularPolygon((i + 0.5, j + 0.5),
                                                 numVertices = 4,
                                                 radius = 0.5 * np.sqrt(2),
                                                 orientation = np.pi / 4.002,
                                                 ec = self.EdgeColor,
                                                 fc = self.CoveredColor)
                                  for j in range(self.n)]
                                 for i in range(self.n)])
        [self.ax.add_patch(sq) for sq in self.squares.flat]

    def create_graph_from_env(self):
        self.graph = Graph(mine_env = self.mine_env)
        self.graph.create_graph_from_env()

    def add_mine_flag(self, row, column):
        if self.clicked[row, column]:
            pass
        elif self.flags[row, column]:
            self.ax.patches.remove(self.flags[row, column])
            self.flags[row, column] = None
        else:
            self.flags[row, column] = plt.Polygon(self.FlagVertices + [row, column], fc = 'red', ec = 'black', lw = 2)
            self.ax.add_patch(self.flags[row, column])

    def click_square(self, row, column):
        if self.mines is None:
            self._place_mines(row, column)

        # if there is a flag or square is already clicked, do nothing
        if self.flags[row, column] or self.clicked[row, column]:
            return

        # Set clicked to True for this square
        self.clicked[row, column] = True

        # hit a mine: game over
        if self.mines[row, column]:
            self.game_over = True
            self._reveal_unmarked_mines()
            self._draw_red_X(row, column)
            self._cross_out_wrong_flags()

        # square with no surrounding mines: clear out all adjacent squares
        elif self.mine_env[row, column] == 0:
            self.squares[row, column].set_facecolor(self.UncoveredColor)
            for ii in range(max(0, row - 1), min(self.n, row + 2)):
                for jj in range(max(0, column - 1), min(self.n, column + 2)):
                    self.click_square(ii, jj)
        else:
            self.squares[row, column].set_facecolor(self.UncoveredColor)
            self.ax.text(x = row + 0.5,
                         y = column + 0.5,
                         s = str(self.mine_env[row, column]),
                         color = self.CountColors[self.mine_env[row, column]],
                         ha = 'center',
                         va = 'center',
                         fontsize = 18,
                         fontweight = 'bold')

        # if all remaining squares are mines, mark them and end game
        if self.mines.sum() == (~self.clicked).sum():
            self.game_over = True
            self._mark_remaining_mines()

    def render_env(self, timer = 2):
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.show()
        plt.pause(timer)