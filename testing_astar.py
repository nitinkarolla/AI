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

    def plot_performance_metrics(self, dim):
        algo_performance = dict()

        algo = "astar"

        env = Environment(n = dim,
                          p = self.probability_of_obstacles,
                          fire = self.fire)

        env.generate_maze()
        env.create_graph_from_maze()

        # for q in np.arange(0, 1.1, 0.1):
        #     print(q)

        for heuristic in ["euclid", "manhattan"]:
            # print(algo)

            if heuristic not in algo_performance:
                algo_performance[heuristic] = dict()
            
            env.algorithm = algo
            env.create_graph_from_maze()

            path_finder = PathFinderAlgorithm(environment = env,
                                              algorithm = "astar",
                                              visual = self.visual,
                                              heuristic = heuristic)
            path_finder.run_path_finder_algorithm()
            performances = path_finder.performance_dict

            if "path_length" not in algo_performance[heuristic]:
                algo_performance[heuristic]["path_length"] = []

            if "maximum_fringe_size" not in algo_performance[heuristic]:
                algo_performance[heuristic]["maximum_fringe_size"] = []

            if "number_of_nodes_expanded" not in algo_performance[heuristic]:
                algo_performance[heuristic]["number_of_nodes_expanded"] = []

            algo_performance[heuristic]['path_length'].append(performances['path_length'])
            algo_performance[heuristic]['maximum_fringe_size'].append(performances['maximum_fringe_size'])
            algo_performance[heuristic]['number_of_nodes_expanded'].append(performances['number_of_nodes_expanded'])

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
    path_euc = []
    path_man = []
    for dim in tqdm(range(10, 111, 10)):
        algo_performance = maze_runner.plot_performance_metrics(dim)
        path_euc.append(algo_performance["euclid"]["path_length"])
        path_man.append(algo_performance["manhattan"]["path_length"])

    plt.figure()
    # max_path = max(max(path_bfs), max(path_dfs)) +
    plt.plot(range(10, 111, 10), path_euc, label="Euclidean Path Length")
    plt.plot(range(10, 111, 10), path_man, label="Manhattan Path Length")
    plt.legend()
    plt.xlabel("Dimensions")
    plt.ylabel("Path Length")
    plt.title("Path Length vs Dimensions for Euclidean and Manhattan")
    plt.savefig('Images/path_length_comparison_astar.png')



    # algo_performance = maze_runner.plot_performance_metrics()
    # plt.figure()
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_euclid"]["path_length"], label="AStar with Euclid")
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_manhattan"]["path_length"], label="AStar with Manhattan")
    # plt.legend()
    # plt.xlabel("Maze Size")
    # plt.ylabel("Path Length")
    # plt.title("Path Length vs Maze Size")
    # plt.savefig('path_length_comparison_astar.png')
    #
    # plt.figure()
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_euclid"]["maximum_fringe_size"], label = "AStar with Euclid")
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_manhattan"]["maximum_fringe_size"], label = "AStar with Manhattan")
    # plt.legend()
    # plt.xlabel("Maze Size")
    # plt.ylabel("Maximum Fringe Size")
    # plt.title("Maximum Fringe Size vs Maze Size")
    # plt.savefig('maximum_fringe_size_comparison_astar.png')
    #
    # plt.figure()
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_euclid"]["number_of_nodes_expanded"], label="AStar with Euclid")
    # plt.plot(np.arange(10, 100, 10), algo_performance["astar_manhattan"]["number_of_nodes_expanded"], label="AStar with Manhattan")
    # plt.legend()
    # plt.xlabel("Maze Size")
    # plt.ylabel("Number of Nodes Expanded")
    # plt.title("Number of Nodes Expanded vs Maze Size")
    # plt.savefig('num_nodes_expanded_comparison_astar.png')
    #
    # print(algo_performance)
