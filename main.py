import argparse
import sys
from MazeRunner.utils.graph import Graph
from MazeRunner.utils.environment import Environment
from MazeRunner.algorithms.path_finding_algorithms import PathFinderAlgorithm


class MazeRunner():

    def __init__(self, maze_dimension, probability_of_obstacles, algorithm, visual, heuristic):
        self.algorithm = algorithm
        self.maze_dimension = maze_dimension
        self.probability_of_obstacles = probability_of_obstacles
        self.visual = visual
        self.heuristic = heuristic

    def run(self):
        # Create the maze
        env = Environment()
        env.generate_maze(n = self.maze_dimension, p = self.probability_of_obstacles)

        # Generate graph from the maze
        graph = Graph(environment = env)
        graph.create_graph_from_maze()

        # Run the path finding algorithm on the graph
        path_finder = PathFinderAlgorithm(graph = graph,
                                          algorithm = self.algorithm,
                                          visual = self.visual,
                                          heuristic = self.heuristic)
        path_finder.run_path_finder_algorithm()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.3)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-he', "--heuristic", default = "edit")
    args = parser.parse_args(sys.argv[1:])

    maze_runner = MazeRunner(maze_dimension = int(args.maze_dimension),
                             probability_of_obstacles = float(args.probability_of_obstacles),
                             algorithm = args.path_finding_algorithm,
                             visual = bool(args.visual),
                             heuristic = args.heuristic)
    maze_runner.run()