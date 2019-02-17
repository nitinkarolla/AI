import argparse
import sys
from MazeRunner.utils.graph import Graph
from MazeRunner.utils.environment import Environment
from MazeRunner.algorithms.path_finding_algorithms import PathFinderAlgorithm


class MazeRunner():

    def __init__(self, algorithm, visual, heuristic, graph):
        self.algorithm = algorithm
        self.visual = visual
        self.heuristic = heuristic
        self.graph = graph

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

    # Create the maze environment
    env = Environment()
    env.generate_maze(n = int(args.maze_dimension),
                      p = float(args.probability_of_obstacles))

    # Generate graph from the maze
    graph = Graph(environment = env)
    graph.create_graph_from_maze()

    maze_runner = MazeRunner(algorithm = args.path_finding_algorithm,
                             graph = graph,
                             visual = bool(args.visual),
                             heuristic = args.heuristic)
    maze_runner.run()