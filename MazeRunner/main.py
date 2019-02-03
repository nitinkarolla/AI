from graph import Graph
from environment import Environment


if __name__ == "__main__":

    # Create the maze
    env = Environment()
    env.generate_maze(n = 16, p = 0.3)

    # Generate graph from the maze
    graph = Graph(maze = env.maze)
    graph.create_graph_from_maze()