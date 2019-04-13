import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict
import sys
import argparse
import random
import csv

class SearchAndDestroy():
    def __init__(self, dimensions, visual, rule, target_type=None):
        self.dim = dimensions
        self.target = ()
        self.original_map = self.generate_map()
        self.num_trials = 0
        # Make the target
        self.target_type = target_type
        self.target = self.create_target()
        self.visual = visual
        self.rule = rule

        if self.visual:
            grid = gridspec.GridSpec(ncols=2, nrows=2)

            # Make a grid of 4 equal parts
            self.fig = plt.figure(figsize=(15,15))
            self.f_ax1 = self.fig.add_subplot(grid[0, 0])
            self.f_ax2 = self.fig.add_subplot(grid[0, 1])
            self.f_ax3 = self.fig.add_subplot(grid[1, 0])
            self.f_ax4 = self.fig.add_subplot(grid[1, 1])

    # MAPPING
    # 0 ---> "Flat"
    # 1 ---> "Hilly"
    # 2 ---> "Forest"
    # 3 ---> "Caves"
    def generate_map(self):
        mat = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                p = np.random.rand()
                if p <= 0.2:
                    mat[i][j] = 0
                elif p > 0.2 and p <= 0.5:
                    mat[i][j] = 1
                elif p > 0.5 and p <= 0.8:
                    mat[i][j] = 2
                else: 
                    mat[i][j] = 3
        
        return mat

    def create_target(self):
        if self.target_type is None:
            x = np.random.randint(self.dim)
            y = np.random.randint(self.dim)
            return (x, y)

        elif self.target_type == "flat":
            indices = np.where(self.original_map == 0)
           
        elif self.target_type == "hill":
            indices = np.where(self.original_map == 1)

        elif self.target_type == "forest":
            indices = np.where(self.original_map == 2)

        else:
            indices = np.where(self.original_map == 3)

        coordinates = list(zip(indices[0], indices[1]))
        if len(coordinates) > 1:
            choose = random.randint(1, len(coordinates)) - 1
            return coordinates[choose]
        else:
            return coordinates[0]

    def get_manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1-x2) + abs(y1-y2)

    def get_distance(self, x1, y1):
        distance_matrix = np.zeros([self.dim, self.dim])
        for x2 in range(self.dim):
            for y2 in range(self.dim):
                distance_matrix[x2][y2] = self.get_manhattan_distance(x1, y1, x2, y2)
        return distance_matrix

    def generate_layout(self, belief, confidence, heat_map, iterations):
        # Display original matrix in console
        # print("\nOriginal Map: \n", self.original_map)
        self.fig.suptitle("Number of iterations: {}".format(iterations))
        self.f_ax1.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
        self.f_ax1.set_title("Actual")
        self.f_ax2.matshow(belief, cmap=cm.get_cmap('Greys_r'))
        self.f_ax2.set_title("Belief Matrix")
        self.f_ax3.matshow(confidence, cmap=cm.get_cmap('Greys_r'))
        self.f_ax3.set_title("Confidence Matrix")
        self.f_ax4.matshow(heat_map, cmap=cm.get_cmap('Greys'))
        self.f_ax4.set_title("Agent")

        self.f_ax1.scatter(self.target[0], self.target[1], s=100, c='red', marker='x')

class Agent():
    def __init__(self, game):
        self.original_map = game.original_map
        self.dim = len(self.original_map)
        self.belief = np.full((self.dim, self.dim), 1/(self.dim**2))
        self.target_cell = game.create_target()
        self.confidence = np.full((self.dim, self.dim), 1/(self.dim**2))
        self.visual = game.visual
        self.heat_map = np.zeros((self.dim, self.dim))
    
        for i in range(self.dim):
            for j in range(self.dim):
                self.confidence[i][j] *= (1 - self.false_neg_rate(i, j)[0])
        # print("Initial Confidence Matrix: \n", self.confidence)

    def false_neg_rate(self, x, y):
        if self.original_map[x][y] == 0:
            fnr = (0.1, "flat")
        elif self.original_map[x][y] == 1:
            fnr = (0.3, "hill")
        elif self.original_map[x][y] == 2:
            fnr = (0.7, "forest")
        else:
            fnr = (0.9, "caves")

        return fnr

    def max_prob_cell(self, rule, matrix=None):
        if rule == "belief":
            mat = self.belief
        elif rule == "confidence":
            mat = self.confidence
        elif rule == "belief with distance" or rule == "confidence with distance":
            mat = matrix

        max_val = np.argmax(mat)
        first_index = int(max_val/self.dim)
        second_index = max_val%self.dim
        max_values = []
        for i in range(self.dim):
            for j in range(self.dim):
                if mat[i][j] == mat[first_index][second_index]:
                    max_values.append((i,j))
        random_from_max = random.randint(1, len(max_values)) - 1
        return max_values[random_from_max]

    def run_game(self, rule_type):
        iterations = 1
        if rule_type == "normal":
            while True:
                current_cell = self.max_prob_cell(game.rule)
                self.heat_map[current_cell[0], current_cell[1]] += 1
                
                # print("Current cell: {}, Target cell: {}".format(current_cell, self.target_cell))
                
                if self.visual:
                        plt.ion()
                        plt.show()
                        plt.pause(1e-15)
                        game.generate_layout(self.belief, self.confidence, self.heat_map, iterations)

                if current_cell == self.target_cell:
                    terrain_prob = self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    p = random.uniform(0, 1)
                    # print("Terrain FNR: ", terrain_prob, " Probability: ", p)
                    if p > terrain_prob:
                        return iterations
                        # break
                else:
                    # Update iterations
                    iterations += 1
                    
                    # Calculate new belief of current cell
                    self.belief[current_cell[0]][current_cell[1]] *= self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    # print("New Belief Matrix: \n", self.belief)
                    
                    # Sum of the belief matrix
                    belief_sum = np.sum(self.belief)
                    # Normalize the belief matrix
                    self.belief = self.belief/belief_sum
                    # print("Normalized Belief Matrix: \n", self.belief)

                    # Calculate new confidence based on new belief
                    for i in range(self.dim):
                        for j in range(self.dim):
                            self.confidence[i][j] = self.belief[i][j]*(1 - self.false_neg_rate(i, j)[0])
                    
                    # Sum of the confidence matrix
                    conf_sum = np.sum(self.confidence)
                    # Normalize the confidence matrix
                    self.confidence = self.confidence/conf_sum
        
        if rule_type == "dist":
            if "belief" in game.rule:
                current_cell = self.max_prob_cell("belief")
            elif "confidence" in game.rule:
                current_cell = self.max_prob_cell("confidence")

            distance_matrix = np.zeros_like(self.belief)

            while True:
                self.heat_map[current_cell[0], current_cell[1]] += 1
                
                # print("Current cell: {}, Target cell: {}".format(current_cell, self.target_cell))
                
                if self.visual:
                        plt.ion()
                        plt.show()
                        plt.pause(1e-15)
                        game.generate_layout(self.belief, self.confidence, self.heat_map, iterations)

                if current_cell == self.target_cell:
                    terrain_prob = self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    p = random.uniform(0, 1)
                    # print("Terrain FNR: ", terrain_prob, " Probability: ", p)
                    if p > terrain_prob:
                        return iterations
                        # break
                else:
                    # Update iterations
                    iterations += 1
                    
                    # Calculate new belief of current cell
                    self.belief[current_cell[0]][current_cell[1]] *= self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    # print("New Belief Matrix: \n", self.belief)
                    
                    # Sum of the belief matrix
                    belief_sum = np.sum(self.belief)
                    # Normalize the belief matrix
                    self.belief = self.belief/belief_sum
                    # print("Normalized Belief Matrix: \n", self.belief)

                    # Calculate new confidence based on new belief
                    for i in range(self.dim):
                        for j in range(self.dim):
                            self.confidence[i][j] = self.belief[i][j]*(1 - self.false_neg_rate(i, j)[0])
                    
                    # Sum of the confidence matrix
                    conf_sum = np.sum(self.confidence)
                    # Normalize the confidence matrix
                    self.confidence = self.confidence/conf_sum

                    distance_matrix = game.get_distance(current_cell[0], current_cell[1])
                    log_matrix = 1 + np.log(1 + distance_matrix)
                    if game.rule == "belief":
                        current_cell = self.max_prob_cell(rule=game.rule, matrix=self.belief/log_matrix)
                    else:
                        current_cell = self.max_prob_cell(rule=game.rule, matrix=self.confidence/log_matrix)

    def target_moves(self, x, y):
        type1 = self.false_neg_rate(x, y)[1]
        possible_moves = [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (-1,1), (1,-1), (1,1)]
        valid = []
        for moves in possible_moves:
            new_cell = (x + moves[0], y + moves[1])
            if new_cell[0] > -1 and new_cell[0] < self.dim and new_cell[1] > -1 and new_cell[1] < self.dim:
                type2 = self.false_neg_rate(new_cell[0], new_cell[1])[1] 
                if type2 != type1:
                    valid.append(new_cell)

        choose = random.randint(1, len(valid)) - 1
        new_target = valid[choose]
        type2 = self.false_neg_rate(new_target[0], new_target[1])[1]
        return new_target, type1, type2

    def valid_neighbors(self, x, y, terrain_type):
        possible_moves = [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (-1,1), (1,-1), (1,1)]
        check = False
        valid = []
        for moves in possible_moves:
            new_cell = (x + moves[0], y + moves[1])
            if new_cell[0] > -1 and new_cell[0] < self.dim and new_cell[1] > -1 and new_cell[1] < self.dim:
                new_type = self.false_neg_rate(new_cell[0], new_cell[1])[1] 
                if new_type == terrain_type:
                    check = True
                    valid.append(new_type)
        
        return check, valid

    def get_neighbors_sum(self, evidence_matrix, type1, type2):
        n_sum = np.zeros_like(self.belief)
        for i in range (self.dim):
            for j in range(self.dim):
                if evidence_matrix[i][j] != 0:
                    check, count = self.valid_neighbors(i, j, type2)
                    if self.false_neg_rate(i, j)[1] == type1 and check:
                        print(type1, i, j, count)
                        n_sum[i][j] += len(count)
                    if self.false_neg_rate(i, j)[1] == type2 and check:
                        print(type2, i, j, count)
                        n_sum[i][j] += len(count)
        
        return n_sum
    # def get_neighbors_sum(self, evidence_matrix):
    #     n_sum = np.zeros_like(self.belief)
    #     for i in range(self.dim):
    #         for j in range(self.dim):
    #             if evidence_matrix[i][j] != 0:
    #                 if i > 0 and j > 0 and i < self.dim-1 and j < self.dim-1:
    #                     n_sum[i][j] = evidence_matrix[i-1][j] + evidence_matrix[i+1][j] + evidence_matrix[i-1][j-1] + evidence_matrix[i-1][j+1] + evidence_matrix[i][j+1] + evidence_matrix[i][j+1] +  evidence_matrix[i+1][j+1] + evidence_matrix[i+1][j-1]
                    
    #                 elif i == 0 and j == 0:
    #                     n_sum[i][j] = evidence_matrix[i][j+1] + evidence_matrix[i+1][j+1] + evidence_matrix[i+1][j]
    #                 elif i == self.dim-1 and j == self.dim-1:
    #                     n_sum[i][j] = evidence_matrix[i][j-1] + evidence_matrix[i-1][j] + evidence_matrix[i-1][j-1]
    #                 elif i == 0 and j == self.dim-1:
    #                     n_sum[i][j] = evidence_matrix[i][j-1] + evidence_matrix[i+1][j-1] + evidence_matrix[i+1][j]
    #                 elif i == self.dim-1 and j == 0:
    #                     n_sum[i][j] = evidence_matrix[i][j+1] + evidence_matrix[i-1][j+1] + evidence_matrix[i-1][j]
                    
    #                 elif i == 0:
    #                     n_sum[i][j] = evidence_matrix[i][j-1] + evidence_matrix[i][j+1] + evidence_matrix[i+1][j] +evidence_matrix[i+1][j-1] + evidence_matrix[i+1][j+1]
    #                 elif i == self.dim -1:
    #                     n_sum[i][j] = evidence_matrix[i][j-1] + evidence_matrix[i][j+1] + evidence_matrix[i-1][j] +evidence_matrix[i-1][j-1] + evidence_matrix[i-1][j+1]
    #                 elif j == 0:
    #                     n_sum[i][j] = evidence_matrix[i][j+1] + evidence_matrix[i-1][j] + evidence_matrix[i+1][j] +evidence_matrix[i-1][j+1] + evidence_matrix[i+1][j+1]
    #                 elif j == self.dim -1:
    #                     n_sum[i][j] = evidence_matrix[i][j-1] + evidence_matrix[i+1][j] + evidence_matrix[i-1][j] +evidence_matrix[i+1][j-1] + evidence_matrix[i-1][j-1]
    #             else:
    #                 n_sum[i][j] = 0
    #     return n_sum

    def surveilance_report(self, type1, type2):
        evidence_points = np.zeros_like(self.belief)
        belief_sum_residual = 0
        cells_with_values = 0
        for i in range(self.dim):
            for j in range(self.dim):
                current_terrain = self.false_neg_rate(i, j)[1]
                if current_terrain == type1 and self.valid_neighbors(i, j, type2)[0]:
                    cells_with_values += 1
                    evidence_points[i][j] = 1
                elif current_terrain == type2 and self.valid_neighbors(i, j, type1)[0]:
                    cells_with_values += 1
                    evidence_points[i][j] = 1
                else:
                    belief_sum_residual += self.belief[i][j]
                    evidence_points[i][j] = 0
        
        # print("Belief\n", self.belief)
        print("Evidence points\n", evidence_points)
        evidence_sum = self.get_neighbors_sum(evidence_points, type1, type2)
        new_belief = self.belief*evidence_points
        # print(belief_sum_residual, cells_with_values)

        print("Sum of neighbors\n", evidence_sum)
        # print("Total sum of neighbors", np.sum(evidence_sum))
        belief_with_evidence = np.zeros_like(new_belief)
        for i in range(self.dim):
            for j in range(self.dim):
                if new_belief[i][j] > 0.0:
                    new_belief[i][j] += new_belief[i][j] + evidence_sum[i][j]*(belief_sum_residual/cells_with_values)
                    belief_with_evidence[i][j] = new_belief[i][j]/evidence_sum[i][j]
        
        # print("After evidence, belief\n", new_belief)
        # print("Final belief\n",belief_with_evidence) 
        # return belief_with_evidence
    
    def run_game_moving_target(self, rule_type):
        iterations = 1
        target = self.target_cell
        if rule_type == "normal":
            current_cell = self.max_prob_cell(game.rule)
            self.heat_map[current_cell[0], current_cell[1]] += 1

            while True:                
                if self.visual:
                        plt.ion()
                        plt.show()
                        plt.pause(1e-15)
                        game.generate_layout(self.belief, self.confidence, self.heat_map, iterations)

                if current_cell == target:
                    terrain_prob = self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    p = random.uniform(0, 1)
                    # print("Terrain FNR: ", terrain_prob, " Probability: ", p)
                    if p > terrain_prob:
                        return iterations
                        # break
                else:
                    # Update iterations
                    iterations += 1
                    
                    # Calculate new belief of current cell
                    self.belief[current_cell[0]][current_cell[1]] *= self.false_neg_rate(current_cell[0], current_cell[1])[0]
                    # print("New Belief Matrix: \n", self.belief)
                    
                    # Sum of the belief matrix
                    belief_sum = np.sum(self.belief)
                    # Normalize the belief matrix
                    self.belief = self.belief/belief_sum
                    # print("Normalized Belief Matrix: \n", self.belief)

                    # Target moved to new position
                    target, type1, type2 = self.target_moves(self.target_cell[0], self.target_cell[1])
                    belief_with_evidence = self.surveilance_report(type1, type2)

                    # Calculate new confidence based on new belief
                    for i in range(self.dim):
                        for j in range(self.dim):
                            self.confidence[i][j] = belief_with_evidence[i][j]*(1 - self.false_neg_rate(i, j)[0])
                    
                    # Sum of the confidence matrix
                    conf_sum = np.sum(self.confidence)
                    # Normalize the confidence matrix
                    self.confidence = self.confidence/conf_sum

                    current_cell = self.max_prob_cell(rule=game.rule, matrix=belief_with_evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic models to search and destroy")
    parser.add_argument('-n', "--grid_dimension", default=10)
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-r', "--rule", default="belief")
    parser.add_argument('-q', "--question", default="basic")
    args = parser.parse_args(sys.argv[1:])

    if args.question == "basic":
        game = SearchAndDestroy(dimensions=int(args.grid_dimension), visual=args.visual, rule=args.rule, target_type=None)
        agent = Agent(game)
        agent_iters = agent.run_game(rule_type="normal")
        print("Number of iterations: ", agent_iters)

    if args.question == "q13":
        sol_dict = {}
        save_file = "q3_analysis.csv"
        csv = open(save_file, "w")
        csv.write("Grid Size, Rule Type, Terrain Type, Iterations\n")
        
        for grid_size in range(5, 50):
            for rule in ["belief", "confidence"]:
                for terrain_target in ["flat", "hill", "forest", "cave"]:
                    agent_iters = 0
                    print("Running for Grid Dimension {}, Rule {}, Terrain Type {}".format(grid_size, rule, terrain_target))
                    for iter in range(10):
                        game = SearchAndDestroy(dimensions=grid_size, visual=args.visual, rule=rule, target_type=terrain_target)
                        agent = Agent(game)
                        agent_iters += agent.run_game(rule_type="normal")

                    agent_iters /= 10
                    if str(grid_size) not in sol_dict:
                        sol_dict[str(grid_size)] = [[rule, terrain_target, int(agent_iters)]]
                    
                    else:
                        sol_dict[str(grid_size)].append([rule, terrain_target, int(agent_iters)])
        
        for key, val in sol_dict.items():
            for v in val:
                row = key + "," + v[0] + "," + v[1] + "," + str(v[2]) + "\n"
                csv.write(row)

    if args.question == "q14":      # TO ADD: Distance covered metric
        sol_dict = {}
        save_file = "q4_analysis.csv"
        csv = open(save_file, "w")
        csv.write("Grid Size, Rule Type, Terrain Type, Iterations\n")
        
        for grid_size in range(5, 50):
            for rule in ["belief", "confidence", "belief with distance", "confidence with distance"]:
                for terrain_target in ["flat", "hill", "forest", "cave"]:
                    agent_iters = 0
                    print("Running for Grid Dimension {}, Rule {}, Terrain Type {}".format(grid_size, rule, terrain_target))
                    for iter in range(10):
                        game = SearchAndDestroy(dimensions=grid_size, visual=args.visual, rule=rule, target_type=terrain_target)
                        agent = Agent(game)
                        if rule in ["belief", "confidence"]:
                            agent_iters += agent.run_game(rule_type="normal")

                        elif rule in ["belief with distance", "confidence with distance"]:
                            agent_iters += agent.run_game(rule_type="dist")    

                    agent_iters /= 10
                    if str(grid_size) not in sol_dict:
                        sol_dict[str(grid_size)] = [[rule, terrain_target, int(agent_iters)]]
                    
                    else:
                        sol_dict[str(grid_size)].append([rule, terrain_target, int(agent_iters)])
        
        for key, val in sol_dict.items():
            for v in val:
                row = key + "," + v[0] + "," + v[1] + "," + str(v[2]) + "\n"
                csv.write(row)
        
    elif args.question == "q2":
        game = SearchAndDestroy(dimensions=int(args.grid_dimension), visual=args.visual, rule=args.rule, target_type=None)
        agent = Agent(game)
        print("Original map\n", game.original_map)
        # agent.run_game_moving_target(rule_type="normal")
        agent.surveilance_report("flat", "hill")