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

    def create_environment(self):

        # Create the maze
        self.env = Environment()
        self.env.generate_maze(n = self.maze_dimension, p = self.probability_of_obstacles)

        # Generate graph from the maze
        self.graph = Graph(environment = self.env)
        self.graph.create_graph_from_maze()

    def modify_environment(self):

        # Modify the current maze environment
        self.env.modify_environment()

        # Generate graph from the new maze
        self.graph = Graph(environment = self.env)
        self.graph.create_graph_from_maze()

    def run(self):

        # Run the path finding algorithm on the graph
        path_finder = PathFinderAlgorithm(graph = self.graph,
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

    maze_runner.create_environment()
    maze_runner.run()

    maze_runner.modify_environment()
    maze_runner.run()