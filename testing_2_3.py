import argparse
import sys
from time import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from MazeRunner.utils.environment import Environment
from MazeRunner.algorithms.path_finding_algorithms import PathFinderAlgorithm
from tqdm import tqdm

class MazeRunner():

    def __init__(self, maze_dimension, probability_of_obstacles, algorithm, visual, heuristic, fire):
        self.algorithm = algorithm
        self.maze_dimension = maze_dimension
        self.probability_of_obstacles = probability_of_obstacles
        self.visual = visual
        self.heuristic = heuristic
        self.fire = fire

    def create_environment(self, new_maze = None):

        # Create the maze
        self.env = Environment(algorithm = self.algorithm, n = self.maze_dimension, p = self.probability_of_obstacles, fire = self.fire)
        self.env.generate_maze(new_maze = new_maze)

        # Generate graph from the maze
        self.env.create_graph_from_maze()

    def run(self):

        # Run the path finding algorithm on the graph
        self.path_finder = PathFinderAlgorithm(environment = self.env,
                                               algorithm = self.algorithm,
                                               visual = self.visual,
                                               heuristic = self.heuristic)
        self.path_finder.run_path_finder_algorithm()

    def find_solvable_map_size(self):
        dim_list = range(10, 250, 10)
        runtimes = dict()
        for dim in dim_list:
            print("Dim = " + str(dim))

            self.maze_dimension = dim
            self.create_environment()

            start = time()
            self.run()
            end = time() - start
            print(end)
            runtimes[dim] = end
            del self.path_finder

        plt.plot(dim_list, runtimes.values(), marker = "o")
        plt.figure()
        plt.show()

    def plot_performance_metrics(self, prob):
        algorithms = {"dfs": 0, "bfs": 0, "astar_euclid": 0, "astar_manhattan": 0}

        for i in range(10):
            env = Environment(n = self.maze_dimension,
                              p = prob,
                              fire = self.fire)

            env.generate_maze()
            env.create_graph_from_maze()

            for algo in algorithms:
                if algo == 'dfs' or algo == 'bfs':
                    algo_temp = algo
                    heuristic = self.heuristic
                else:
                    algo_temp = algo.split("_")[0]
                    heuristic = algo.split("_")[1]

                env.algorithm = algo_temp
                env.create_graph_from_maze()
                path_finder = PathFinderAlgorithm(environment = env,
                                                algorithm = algo_temp,
                                                visual = self.visual,
                                                heuristic = heuristic)
                path_finder.run_path_finder_algorithm()
                performances = path_finder.performance_dict

                if performances['path_length'] != 1:
                    algorithms[algo] += 1

        for key in algorithms.keys():
            algorithms[key] /= 10

        return algo_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.2)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-he', "--heuristic", default = "euclid")
    parser.add_argument('-f', "--fire", default = False)
    args = parser.parse_args(sys.argv[1:])

    maze_runner = MazeRunner(maze_dimension = int(args.maze_dimension),
                             probability_of_obstacles = float(args.probability_of_obstacles),
                             algorithm = args.path_finding_algorithm,
                             visual = bool(args.visual),
                             heuristic = args.heuristic,
                             fire = args.fire)

    # maze_runner.create_environment()
    # maze_runner.run()
    probability = {'dfs': 0, 'bfs': 0, 'astar_euclid': 0, 'astar': 0}
    for prob in tqdm(np.arange(0.0, 1.1, 0.1)):
        algo_performance = maze_runner.plot_performance_metrics(prob)

    plt.figure()
    # max_path = max(max(path_bfs), max(path_dfs)) +
    plt.plot(np.arange(0.0, 1.1, 0.1), algo_performance['dfs'], label="#DFS Solutions")
    plt.plot(np.arange(0.0, 1.1, 0.1), algo_performance['bfs'], label="#BFS Solutions")
    plt.plot(np.arange(0.0, 1.1, 0.1), algo_performance['astar_euclid'], label="#Astar Euclidean Solutions")
    plt.plot(np.arange(0.0, 1.1, 0.1), algo_performance['astar_manhattan'], label="#AStar Manhattan Solutions")
    plt.legend()
    plt.xlabel("Probability")
    plt.ylabel("#Solvable mazes")
    plt.title("Solvable mazes w.r.t. probability")
    plt.savefig('Images/solvable_mazes_wrt_prob.png')
    
