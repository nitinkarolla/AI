class PathFinderAlgorithm():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "Astar"

    def __init__(self, graph = None, algorithm = None):
        self.graph = graph
        self.algorithm = algorithm
        self.path = []

    def _get_unvisited_children(self, node_children):
        unvisited_children = []
        for child in node_children:
            if child is None:
                continue

            if child not in self.path:
                unvisited_children.append(child)
        return unvisited_children

    def _run_dfs(self):

        root = self.graph.graph_maze[0, 0]
        dest = self.graph.graph_maze[self.graph.environment.n - 1, self.graph.environment.n - 1]

        self.fringe = [root]
        self.path = [root]
        while self.fringe:
            node = self.fringe.pop()

            # update color of the cell and render the maze
            self.graph.environment.update_color_of_cell(node.row, node.column)
            self.graph.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            if node not in self.path:
                self.path.append(node)

            # If there is no further path, then reset the color of the cell. Also, subsequently reset
            # the color of all parent cells along the path who have no other children to explore.
            flag = True
            while(flag):
                node_children = node.get_children(node = node)
                unvisited_children = self._get_unvisited_children(node_children)

                # If no unvisited children found, then reset the color of this cell in the current path.
                if len(unvisited_children) == 0:
                    self.graph.environment.reset_color_of_cell(node.row, node.column)
                    self.graph.environment.render_maze()
                else:
                    for child in unvisited_children:
                        child.parent = node
                        self.fringe.append(child)
                    flag = False

                node = node.parent
                if node is None:
                    flag = False

    def _run_bfs(self):

        root = self.graph.graph_maze[0, 0]
        dest = self.graph.graph_maze[self.graph.environment.n - 1, self.graph.environment.n - 1]

        self.fringe = [root]
        self.path = [root]
        while self.fringe:
            temp_path = []
            node = self.fringe.pop(0)

            node_children = node.get_children(node = node)
            unvisited_children = self._get_unvisited_children(node_children)

            for child in unvisited_children:
                child.parent = node
                self.fringe.append(child)
                self.path.append(child)

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while(flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above
            while(len(temp_path) != 0):
                temp_node = temp_path.pop()
                self.graph.environment.update_color_of_cell(temp_node.row, temp_node.column)
                self.graph.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)
                self.graph.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                self.graph.environment.render_maze()

    def _run_a_star(self):
        return

    def run_path_finder_algorithm(self):
        if self.algorithm == self.DfsString:
            self._run_dfs()
        elif self.algorithm == self.BfsString:
            self._run_bfs()
        else:
            self._run_a_star()

        # Display the final highlighted path
        self.graph.environment.render_maze(timer = 10)