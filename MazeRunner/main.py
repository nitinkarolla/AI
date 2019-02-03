from AI.MazeRunner.graph import Graph
from AI.MazeRunner.environment import Environment
from AI.MazeRunner.dfs import DFS
from AI.MazeRunner.bfs import BFS


class MazeRunner():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "Astar"

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def run(self):
        # Create the maze
        env = Environment()
        env.generate_maze(n = 16, p = 0.3)

        # Generate graph from the maze
        graph = Graph(environment = env)
        graph.create_graph_from_maze()

        # Run the algorithm on the graph
        if self.algorithm == self.DfsString:
            path_finder = DFS(graph = graph)
        elif self.algorithm == self.BfsString:
            path_finder = BFS(graph = graph)

if __name__ == "__main__":
    maze_runner = MazeRunner(algorithm = "dfs")
    maze_runner.run()