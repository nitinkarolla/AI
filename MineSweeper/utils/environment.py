import matplotlib
matplotlib.use('tkAgg')
from pylab import *
from utils.variable import Variable
from scipy.signal import convolve2d
from matplotlib.patches import RegularPolygon
import matplotlib.gridspec as gridspec


class Environment():
    DimensionOfGround = 8
    NumberOfMines = 10
    CoveredColor = '#DDDDDD'
    EdgeColor = '#888888'
    UncoveredColor = '#AAAAAA'
    CountColors = ['none', 'blue', 'green', 'red', 'darkblue', 'darkred', 'darkgreen', 'black', 'black']
    FlagVertices = np.array([[0.25, 0.2], [0.25, 0.8],
                            [0.75, 0.65], [0.25, 0.5]])

    def __init__(self,
                 n = DimensionOfGround,
                 number_of_mines = NumberOfMines):
        self.n = n
        self.number_of_mines = number_of_mines
        self.mines = None
        self.clicked = np.zeros((self.n, self.n), dtype = bool)
        self.flags = np.zeros((self.n, self.n), dtype = object)
        self.number_of_flags_left = number_of_mines
        self.mine_ground_copy = np.empty((self.n, self.n))*np.nan
        self.variable_mine_ground_copy = np.empty((self.n, self.n), dtype = object)
        self.grid = gridspec.GridSpec(1, 2, wspace = 0.2, hspace = 0.7)

    def _place_mines(self, row, column):

        # randomly place mines on a grid, but not on space (i, j)
        idx = np.concatenate([np.arange(row * self.n + column), np.arange(row * self.n + column + 1, self.n * self.n)])
        np.random.shuffle(idx)
        self.mines = np.zeros((self.n, self.n), dtype = bool)
        self.mines.flat[idx[:self.number_of_mines]] = 1

        # count the number of mines bordering each square
        self.mine_ground = convolve2d(self.mines.astype(complex), np.ones((3, 3)), mode = 'same').real.astype(int)

        for row in range(self.n):
            for column in range(self.n):
                self.squares[row, column].set_facecolor(self.UncoveredColor)
                if self.mines[row, column]:
                    self._reveal_unmarked_mines()
                else:
                    self.ax.text(x = row + 0.5,
                                 y = column + 0.5,
                                 s = str(self.mine_ground[row, column]),
                                 color = self.CountColors[self.mine_ground[row, column]],
                                 ha = 'center',
                                 va = 'center',
                                 fontsize = 18,
                                 fontweight = 'bold')

    def _reveal_unmarked_mines(self):
        for (row, column) in zip(*np.where(self.mines & ~self.flags.astype(bool))):
            self._draw_mine(row, column)

    def _draw_mine(self, row, column):
        self.ax.add_patch(plt.Circle((row + 0.5, column + 0.5), radius = 0.25, ec = 'black', fc = 'black'))

    def _draw_red_X(self, row, column):
        self.ax_copy.text(x = row + 0.5, y = column + 0.5, s = 'X', color = 'r', fontsize = 20, ha = 'center', va = 'center')

    def _cross_out_wrong_flags(self):
        for (row, column) in zip(*np.where(~self.mines & self.flags.astype(bool))):
            self._draw_red_X(row, column)

    def _mark_remaining_mines(self):
        for (row, column) in zip(*np.where(self.mines & ~self.flags.astype(bool))):
            self.add_mine_flag(row, column)

    def _create_graph_from_env(self):
        self.graph = Graph(mine_maze = self.mine_ground)
        self.graph.create_graph_from_env()

    def generate_environment(self):

        # Create the figure and axes
        self.fig = plt.figure(figsize = ((self.n + 2) / 3., (self.n + 2) / 3.))
        self.ax = self.fig.add_axes((0.1, 0.1, 0.4, 0.8),
                                    aspect = 'equal',
                                    frameon = False,
                                    xlim = (-0.05, self.n + 0.05),
                                    ylim = (-0.05, self.n + 0.05))
        self.ax.set_title("Actual", fontsize = 20)
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

        # Create a duplicate figure and axes
        self.ax_copy = self.fig.add_axes((0.5, 0.1, 0.4, 0.8),
                                      aspect = 'equal',
                                      frameon = False,
                                      xlim = (-0.05, self.n + 0.05),
                                      ylim = (-0.05, self.n + 0.05))
        self.ax_copy.set_title("Agent", fontsize = 20)
        for axis in (self.ax_copy.xaxis, self.ax_copy.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())

        self.squares_copy = np.array([[RegularPolygon((i + 0.5, j + 0.5),
                                                 numVertices = 4,
                                                 radius = 0.5 * np.sqrt(2),
                                                 orientation = np.pi / 4.002,
                                                 ec = self.EdgeColor,
                                                 fc = self.CoveredColor)
                                  for j in range(self.n)]
                                 for i in range(self.n)])
        [self.ax_copy.add_patch(sq) for sq in self.squares_copy.flat]

    def add_mine_flag(self, row, column):
        if self.clicked[row, column]  :
            pass
        elif self.flags[row, column]:
            self.ax_copy.patches.remove(self.flags[row, column])
            self.flags[row, column] = None
            self.number_of_flags_left += 1
        else:
            self.number_of_flags_left -= 1
            self.flags[row, column] = plt.Polygon(self.FlagVertices + [row, column], fc = 'red', ec = 'black', lw = 2)
            self.ax_copy.add_patch(self.flags[row, column])

    def click_square(self, row, column):
        if self.mines is None:
            self._place_mines(row, column)

        # if there is a flag or square is already clicked, do nothing
        if self.flags.astype(bool)[row, column] ==  True or self.clicked[row, column]:
            return

        # Set clicked to True for this square
        self.clicked[row, column] = True

        # Open up cells in the mine ground copy
        self.mine_ground_copy[row, column] = self.mine_ground[row, column]
        self.variable_mine_ground_copy[row, column] = Variable(value = self.mine_ground[row, column],
                                                               row = row,
                                                               column = column,
                                                               has_mine = 0)

        # hit a mine: game over
        if self.mines[row, column]:
            self.game_over = True
            self._reveal_unmarked_mines()
            self._draw_red_X(row, column)
            self._cross_out_wrong_flags()
        else:
            self.squares_copy[row, column].set_facecolor(self.UncoveredColor)
            self.ax_copy.text(x = row + 0.5,
                         y = column + 0.5,
                         s = str(self.mine_ground[row, column]),
                         color = self.CountColors[self.mine_ground[row, column]],
                         ha = 'center',
                         va = 'center',
                         fontsize = 18,
                         fontweight = 'bold')

        # if all remaining squares are mines, mark them and end game
        if self.mines.sum() == (~self.clicked).sum():
            self.game_over = True
            self._mark_remaining_mines()

    def render_env(self, timer = 0.01):
        self.ax.plot()
        self.ax_copy.plot()
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.show()
        plt.pause(timer)
