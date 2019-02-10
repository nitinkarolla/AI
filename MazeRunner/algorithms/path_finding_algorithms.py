class PathFinderAlgorithm():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "Astar"

    def __init__(self, graph = None, algorithm = None):
        self.graph = graph
        self.algorithm = algorithm

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
        self.path = []
        try:
            while self.fringe:
                node = self.fringe.pop()

                # update color of the cell and render the maze
                self.graph.environment.update_color_of_cell(node.row, node.column)
                self.graph.environment.render_maze()

                if (node == dest):
                    exit()

                if node not in self.path:
                    self.path.append(node)

                flag = True
                while(flag):
                    node_children = node.get_children(node = node)
                    unvisited_children = self._get_unvisited_children(node_children)

                    if len(unvisited_children) == 0:
                        self.graph.environment.reset_color_of_cell(node.row, node.column)
                        self.graph.environment.render_maze()
                    else:
                        for child in unvisited_children:
                            child.parent = node
                            self.fringe.append(child)
                        flag = False

                    node = node.parent
        except Exception:
            self.graph.environment.display_maze()

    def _run_bfs(self):
        return

    def _run_a_star(self):
        return

    def run_path_finder_algorithm(self):
        if self.algorithm == self.DfsString:
            self._run_dfs()
        elif self.algorithm == self.BfsString:
            self._run_bfs()
        else:
            self._run_a_star()