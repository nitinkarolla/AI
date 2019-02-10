from MazeRunner.utils.graph import Graph
from MazeRunner.utils.environment import Environment
from MazeRunner.algorithms.path_finding_algorithms import PathFinderAlgorithm


class MazeRunner():

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def run(self):
        # Create the maze
        env = Environment()
        env.generate_maze(n = 20, p = 0.4)

        # Generate graph from the maze
        graph = Graph(environment = env)
        graph.create_graph_from_maze()

        # Run the path finding algorithm
        path_finder = PathFinderAlgorithm(graph = graph, algorithm = self.algorithm)
        path_finder.run_path_finder_algorithm()

if __name__ == "__main__":
    maze_runner = MazeRunner(algorithm = "dfs")
    maze_runner.run()