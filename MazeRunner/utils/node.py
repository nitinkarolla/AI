class Node():
    def __init__(self,
                 value = None,
                 row = None,
                 column = None,
                 left = None,
                 right = None,
                 up = None,
                 down = None,
                 parent = None,
                 distance_from_root = None,
                 previsit = None,
                 postvisit = None,
                 num_nodes_before_this_node = None):
        self.value = value
        self.row = row
        self.column = column
        self.parent = parent
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.distance_from_root = distance_from_root
        self.previsit = previsit
        self.postvisit = postvisit
        self.num_nodes_before_this_node = num_nodes_before_this_node

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def get_children(self, node):
        return [node.up, node.left, node.down, node.right]