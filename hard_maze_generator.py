import numpy as np
from main import MazeRunner


class HardMazeGenerator():

    def __init__(self, n = 5, p = 0.2, algorithm = 'dfs', metric = "path", heuristic = None, max_iterations = 100, load_maze = None):
        self.n = n
        self.p = p
        self.algorithm = algorithm
        self.metric = metric
        self.load_maze = load_maze
        self.heuristic = heuristic
        self.max_iterations = max_iterations

    def _run(self):
        maze_runner = MazeRunner(maze_dimension = self.n, 
                                 probability_of_obstacles = self.p,
                                 algorithm = self.algorithm,
                                 visual = True,
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
            while i < (self.n)**2 :

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
    hard_maze = HardMazeGenerator()
    hard_maze._run()
    print(hard_maze.global_difficult_maze_metric)
    print(hard_maze.global_difficult_maze)












