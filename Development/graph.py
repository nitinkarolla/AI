import numpy as np


class Node():
    def __init__(self,
                 value = None,
                 previous = None,
                 left = None,
                 right = None,
                 up = None,
                 down = None,
                 pre_visit = None,
                 post_visit = None,
                 distance_from_root = None,
                 num_nodes_before_this_node = None):
        self.value = value
        self.previous = previous
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.pre_visit = pre_visit
        self.post_visit = post_visit
        self.distance_from_root = distance_from_root
        self.num_nodes_before_this_node = num_nodes_before_this_node

class Graph():
    def __init__(self, maze):
        self.maze = maze
        self.graph_maze = np.empty(shape = self.maze.shape, dtype = object)

    def create_graph_from_maze(self):
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                if self.maze[row, column] == 0:
                    continue
                self.graph_maze[row, column] = Node(value = self.maze[row, column])

        # Left
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    if column - 1 >= 0:
                        self.graph_maze[row, column].left = self.graph_maze[row, column - 1]
                except Exception:
                    continue

        # Right
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    self.graph_maze[row, column].right = self.graph_maze[row, column + 1]
                except Exception:
                    continue

        # Up
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    if row - 1 >= 0:
                        self.graph_maze[row, column].up = self.graph_maze[row - 1, column]
                except Exception:
                    continue

        # Down
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    self.graph_maze[row, column].down = self.graph_maze[row + 1, column]
                except Exception:
                    continue