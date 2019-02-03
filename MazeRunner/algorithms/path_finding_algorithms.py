class PathFinderAlgorithm():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "Astar"

    def __init__(self, graph = None, algorithm = None):
        self.graph = graph
        self.algorithm = algorithm

    def _run_dfs(self):
        return

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