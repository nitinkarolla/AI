import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
from utils.environment import Environment
from agent import MineSweeperAgent


class MineSweeper():

    def __init__(self, ground_dimension, number_of_mines, visual):
        self.ground_dimension = ground_dimension
        self.number_of_mines = number_of_mines
        self.visual = visual

    def create_environment(self):

        # Create the maze
        self.env = Environment(n = self.ground_dimension, number_of_mines = self.number_of_mines)
        self.env.generate_environment()

    def run(self):

        # Use the agent to find mines in our mine-sweeper environment
        self.mine_sweeper_agent = MineSweeperAgent(environment = self.env, visual = self.visual)
        self.mine_sweeper_agent.play()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--ground_dimension", default = 8)
    parser.add_argument("-nmines", "--number_of_mines", default = 1)
    parser.add_argument('-v', "--visual", default = False)
    args = parser.parse_args(sys.argv[1:])

    mine_sweeper = MineSweeper(ground_dimension = int(args.ground_dimension),
                               number_of_mines = int(args.number_of_mines),
                               visual = bool(args.visual))

    mine_sweeper.create_environment()
    mine_sweeper.run()