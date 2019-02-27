import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
from utils.environment import Environment
from agent import MineSweeperAgent


class MineSweeper():

    def __init__(self, maze_dimension, number_of_mines, visual):
        self.maze_dimension = maze_dimension
        self.number_of_mines = number_of_mines
        self.visual = visual

    def create_environment(self):

        # Create the maze
        self.env = Environment(n = self.maze_dimension, number_of_mines = self.number_of_mines)
        self.env.generate_environment()

        # Generate graph from the maze
        # self.env.create_graph_from_env()

    def run(self):

        # Use the agent to find mines in our mine-sweeper environment
        self.mine_sweeper_agent = MineSweeperAgent(environment = self.env, visual = self.visual)
        self.mine_sweeper_agent.find_mines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--maze_dimension", default = 8)
    parser.add_argument("-nmines", "--number_of_mines", default = 10)
    parser.add_argument('-v', "--visual", default = False)
    args = parser.parse_args(sys.argv[1:])

    mine_sweeper = MineSweeper(maze_dimension = int(args.maze_dimension),
                               number_of_mines = int(args.number_of_mines),
                               visual = bool(args.visual))

    mine_sweeper.create_environment()
    mine_sweeper.env.render_env()
    mine_sweeper.env.click_square(0, 0)
    mine_sweeper.env.render_env()
    mine_sweeper.env.click_square(mine_sweeper.maze_dimension - 1, mine_sweeper.maze_dimension - 1)
    mine_sweeper.env.render_env()