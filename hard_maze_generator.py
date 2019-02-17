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
        j = 0

        while j < self.max_iterations: 
            #print(j)
            maze_runner.create_environment()
            #print(maze_runner.env.maze)

            local_difficult_maze = None
            local_difficult_maze_metric = 0

            i = 0
            #Inside Terminate Condition
            while i < (self.n)**2 :
                #print(i)
                # Store the values of Maximum Difficult metric and the maze
                maze_runner.run()
                if maze_runner.path_finder.get_final_path_length() == 1 :
                    break
                if self.metric == "path":
                    if maze_runner.path_finder.get_final_path_length() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_final_path_length()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.env.reset_environment()
                elif self.metric == "memory":
                    if maze_runner.path_finder.get_maximum_fringe_length() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_maximum_fringe_length()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.env.reset_environment()
                elif self.metric == "nodes":
                    if maze_runner.path_finder.get_number_of_nodes_expanded() > local_difficult_maze_metric and maze_runner.path_finder.get_final_path_length() != 1:
                        local_difficult_maze_metric = maze_runner.path_finder.get_number_of_nodes_expanded()
                        local_difficult_maze = maze_runner.env.maze
                    else :
                        maze_runner.env.reset_environment()

                #print("local" , local_difficult_maze_metric)
                #Modify Maze Function
                maze_runner.env.modify_environment()

                i = i + 1

            if self.global_difficult_maze_metric < local_difficult_maze_metric:
                self.global_difficult_maze_metric = local_difficult_maze_metric
                self.global_difficult_maze = local_difficult_maze
            #print("global" , global_difficult_maze_metric)
            #stoping criteria design 
            j = j + 1
         
if __name__ == "__main__":
    hard_maze = HardMazeGenerator()
    hard_maze._run()
    print(hard_maze.global_difficult_maze_metric)
    print(hard_maze.global_difficult_maze)












