import numpy as np
import queue as Q


class PathFinderAlgorithm():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "astar"

    def __init__(self, environment = None, algorithm = None, visual = False, heuristic = None):
        self.environment = environment
        self.graph_maze = self.environment.graph.graph_maze
        self.algorithm = algorithm
        self.visual = visual
        self.heuristic = heuristic
        self.visited = []
        self.path = []
        self.max_fringe_length = 0

    def _get_unvisited_children(self, node_children):
        unvisited_children = []
        for child in node_children:
            if child is None:
                continue

            if child not in self.visited:
                unvisited_children.append(child)
        return unvisited_children

    def _get_final_path(self):
        node = self.graph_maze[self.environment.n - 1, self.environment.n - 1]
        while node is not None:
            self.path.append((node.row, node.column))
            node = node.parent

    def _get_euclidien_distance(self, node, dest):
        return np.sqrt((node.row - dest.row)**2 + (node.column - dest.column)**2)

    def _get_manhattan_distance(self, node, dest):
        return np.abs(node.row - dest.row) + np.abs(node.column - dest.column)

    def get_final_path_length(self):
        return len(self.path)

    def get_number_of_nodes_expanded(self):
        return len(self.visited)

    def get_maximum_fringe_length(self):
        return self.max_fringe_length

    def _create_performance_metrics(self):
        self.performance_dict = dict()
        self.performance_dict['path_length'] = self.get_final_path_length()
        self.performance_dict['maximum_fringe_size'] = self.get_maximum_fringe_length()
        self.performance_dict['number_of_nodes_expanded'] = self.get_number_of_nodes_expanded()

    def _run_dfs(self):

        root = self.graph_maze[0, 0]
        dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1]

        self.fringe = [root]
        self.visited.append(root)
        while self.fringe:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            node = self.fringe.pop()

            # update color of the cell and render the maze
            if self.visual == True :            #Added visualisation parameter
                self.environment.update_color_of_cell(node.row, node.column)
                self.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            if node not in self.visited:
                self.visited.append(node)

            # If there is no further path, then reset the color of the cell. Also, subsequently reset
            # the color of all parent cells along the path who have no other children to explore.
            flag = True
            while(flag):
                node_children = node.get_children(node = node, algorithm = self.algorithm)
                unvisited_children = self._get_unvisited_children(node_children)

                # If no unvisited children found, then reset the color of this cell in the current path
                # because there is no further path from this cell.
                if len(unvisited_children) == 0:
                    if self.visual == True:         #Added visualisation parameter --Nitin & Vedant
                        self.environment.reset_color_of_cell(node.row, node.column)
                        self.environment.render_maze()
                else:
                    for child in unvisited_children:
                        child.parent = node
                        self.fringe.append(child)
                    flag = False

                node = node.parent
                if node is None:
                    flag = False

    def _run_bfs(self):

        root = self.graph_maze[0, 0]
        dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1]

        self.fringe = [root]
        self.visited.append(root)
        while self.fringe:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            temp_path = []
            node = self.fringe.pop(0)

            if node not in self.visited:
                self.visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)
            unvisited_children = self._get_unvisited_children(node_children)

            for child in unvisited_children:

                # If child has been added to the fringe by some previous node, then dont add it again.
                if child not in self.fringe:
                    child.parent = node
                    self.fringe.append(child)

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while (flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above by popping the root first and the subsequent nodes.
            while (len(temp_path) != 0):
                temp_node = temp_path.pop()

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.update_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the entire path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()


    def _run_astar(self):

        root = self.graph_maze[0, 0]
        dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1]

        # Assign distance from each node to the destination
        for row in range(len(self.environment.maze)):
            for column in range(len(self.environment.maze)):
                if self.environment.maze[row, column] == 0:
                    continue
                if self.heuristic == "edit":
                    self.graph_maze[row, column].distance_from_dest = self._get_euclidien_distance(
                        self.graph_maze[row, column], dest)
                else:
                    self.graph_maze[row, column].distance_from_dest = self._get_manhattan_distance(
                        self.graph_maze[row, column], dest)

        # Root is at a distance of 0 from itself
        root.distance_from_source = 0

        self.fringe = Q.PriorityQueue()
        self.fringe.put(root)

        self.visited.append(root)
        while self.fringe.queue:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe.queue)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            temp_path = []
            node = self.fringe.get()

            if node not in self.visited:
                self.visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)

            for child in node_children:
                if child is None or child in self.visited:
                    continue

                # If child has been added to the fringe by some previous node, then dont add it again.
                if child not in self.fringe.queue:
                    child.parent = node
                    child.distance_from_source = node.distance_from_source + 1
                    self.fringe.put(child)
                else:
                    if child.get_heuristic() >= node.distance_from_source + child.distance_from_dest:
                        child.parent = node
                        child.distance_from_source = node.distance_from_source + 1

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while (flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above by popping the root first and the subsequent nodes.
            while (len(temp_path) != 0):
                temp_node = temp_path.pop()

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.update_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the entire path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()

    def run_path_finder_algorithm(self):
        if self.algorithm == self.DfsString:
            self._run_dfs()
        elif self.algorithm == self.BfsString:
            self._run_bfs()
        else:
            self._run_astar()

        # Get the final path
        self._get_final_path()

        # Create performance metrics
        self._create_performance_metrics()

        if len(self.path) == 1:
            print("NO PATH FOUND")
            return

        # Reverse the final saved path
        self.path = self.path[::-1]

        # Display the final highlighted path
        if self.visual == True:
            self.environment.render_maze(timer = 0.1)
