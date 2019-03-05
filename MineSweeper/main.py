import argparse
import sys
from utils.environment import Environment
from agents.base_agent import BaseAgent
from agents.csp_agent import CSPAgent


class MineSweeper():
    BasicAgent = "base_agent"
    CSPAgent = "csp_agent"

    def __init__(self, ground_dimension = None, number_of_mines = None, agent_name = None, visual = None):
        self.ground_dimension = ground_dimension
        self.number_of_mines = number_of_mines
        self.agent_name = agent_name
        self.visual = visual

    def create_environment(self):

        # Create the maze
        self.env = Environment(n = self.ground_dimension, number_of_mines = self.number_of_mines)
        self.env.generate_environment()

    def run(self):

        # Use the agent to find mines in our mine-sweeper environment
        if self.agent_name == self.BasicAgent:
            self.mine_sweeper_agent = BaseAgent(env = self.env)
        else:
            self.mine_sweeper_agent = CSPAgent(env = self.env)
        self.mine_sweeper_agent.play()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'create AI agents to play mine-sweeper')
    parser.add_argument("-n", "--ground_dimension", default = 10)
    parser.add_argument("-a", "--agent_name", default = "base_agent")
    parser.add_argument("-nmines", "--number_of_mines", default = 4)
    parser.add_argument('-v', "--visual", default = False)
    args = parser.parse_args(sys.argv[1:])

    mine_sweeper = MineSweeper(ground_dimension = int(args.ground_dimension),
                               number_of_mines = int(args.number_of_mines),
                               agent_name = args.agent_name,
                               visual = bool(args.visual))

    mine_sweeper.create_environment()
    mine_sweeper.run()
    mine_sweeper.env.render_env(10)
