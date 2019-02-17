import argparse
import sys
from main import MazeRunner


class HardMazeGenerator():

    def __init__(self,
                 maze_dimension = 5,
                 probability_of_obstacles = 0.2,
                 algorithm = 'dfs',
                 metric = "path",
                 heuristic = None,
                 max_iterations = 100,
                 visual = None):
        self.maze_dimension = maze_dimension
        self.probability_of_obstacles = probability_of_obstacles
        self.algorithm = algorithm
        self.metric = metric
        self.visual = visual
        self.heuristic = heuristic
        self.max_iterations = max_iterations

    def run(self):
        maze_runner = MazeRunner(maze_dimension = self.maze_dimension,
                                 probability_of_obstacles = self.probability_of_obstacles,
                                 algorithm = self.algorithm,
                                 visual = self.visual,
                                 heuristic = self.heuristic)


        self.global_difficult_maze = None
        self.global_difficult_maze_metric = 0
        iteration_count = 0

        while iteration_count < self.max_iterations:
            maze_runner.create_environment()

            local_difficult_maze = None
            local_difficult_maze_metric = 0

            i = 0

            # Inside Terminate Condition
            while i < (self.maze_dimension)**2 :

                # Store the values of Maximum Difficult metric and the maze
                maze_runner.run()
                if maze_runner.path_finder.get_final_path_length() == 1 :
                    break
                if self.metric == "path":
                    if maze_runner.path_finder.get_final_path_length() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_final_path_length()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.reset_environment()
                elif self.metric == "memory":
                    if maze_runner.path_finder.get_maximum_fringe_length() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_maximum_fringe_length()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.reset_environment()
                elif self.metric == "nodes":
                    if maze_runner.path_finder.get_number_of_nodes_expanded() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_number_of_nodes_expanded()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.reset_environment()

                # Modify the maze environment - randomly change the value of a cell
                maze_runner.modify_environment()

                i = i + 1

            if self.global_difficult_maze_metric < local_difficult_maze_metric:
                self.global_difficult_maze_metric = local_difficult_maze_metric
                self.global_difficult_maze = local_difficult_maze

            # Stopping criteria design
            iteration_count = iteration_count + 1
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate hard mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.2)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-i', "--max_iterations", default = False)
    parser.add_argument('-he', "--heuristic", default = "edit")
    parser.add_argument('-m', "--metric", default = "path")
    args = parser.parse_args(sys.argv[1:])

    hard_maze = HardMazeGenerator(maze_dimension = int(args.maze_dimension),
                                  probability_of_obstacles = float(args.probability_of_obstacles),
                                  algorithm = args.path_finding_algorithm,
                                  visual = bool(args.visual),
                                  max_iterations = int(args.max_iterations),
                                  metric = args.metric,
                                  heuristic = args.heuristic)
    hard_maze.run()













