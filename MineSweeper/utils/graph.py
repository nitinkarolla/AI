import numpy as np
from utils.node import Node


class Graph():
    def __init__(self, mine_env = None):
        self.mine_env = mine_env
        self.graph_maze = np.empty(shape = self.mine_env.shape, dtype = object)

    def create_graph_from_maze(self):
        for row in range(len(self.mine_env)):
            for column in range(len(self.mine_env)):
                self.graph_maze[row, column] = Node(value = self.mine_env[row, column],
                                                    row = row,
                                                    column = column)

        # Left
        for row in range(len(self.mine_env)):
            for column in range(len(self.mine_env)):
                try:
                    if column - 1 >= 0:
                        self.graph_maze[row, column].left = self.graph_maze[row, column - 1]
                except Exception:
                    continue

        # Right
        for row in range(len(self.mine_env)):
            for column in range(len(self.mine_env)):
                try:
                    self.graph_maze[row, column].right = self.graph_maze[row, column + 1]
                except Exception:
                    continue

        # Up
        for row in range(len(self.mine_env)):
            for column in range(len(self.mine_env)):
                try:
                    if row - 1 >= 0:
                        self.graph_maze[row, column].up = self.graph_maze[row - 1, column]
                except Exception:
                    continue

        # Down
        for row in range(len(self.mine_env)):
            for column in range(len(self.mine_env)):
                try:
                    self.graph_maze[row, column].down = self.graph_maze[row + 1, column]
                except Exception:
                    continue