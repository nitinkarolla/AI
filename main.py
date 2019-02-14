import argparse
import sys
from MazeRunner.utils.graph import Graph
from MazeRunner.utils.environment import Environment
from MazeRunner.algorithms.path_finding_algorithms import PathFinderAlgorithm


class MazeRunner():

    def __init__(self, maze_dimension, probability_of_obstacles, algorithm):
        self.algorithm = algorithm
        self.maze_dimension = maze_dimension
        self.probability_of_obstacles = probability_of_obstacles

    def run(self):
        # Create the maze
        env = Environment()
        env.generate_maze(n = self.maze_dimension, p = self.probability_of_obstacles)

        # Generate graph from the maze
        graph = Graph(environment = env)
        graph.create_graph_from_maze()

        # Run the path finding algorithm on the graph
        path_finder = PathFinderAlgorithm(graph = graph, algorithm = self.algorithm)
        path_finder.run_path_finder_algorithm()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.3)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    args = parser.parse_args(sys.argv[1:])

    maze_runner = MazeRunner(maze_dimension = int(args.maze_dimension),
                             probability_of_obstacles = float(args.probability_of_obstacles),
                             algorithm = args.path_finding_algorithm)
    maze_runner.run()