class Node():
    def __init__(self,
                 value = None,
                 previous = None,
                 left = None,
                 right = None,
                 up = None,
                 down = None,
                 pre_visit = None,
                 post_visit = None,
                 distance_from_root = None,
                 num_nodes_before_this_node = None):
        self.value = value
        self.previous = previous
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.pre_visit = pre_visit
        self.post_visit = post_visit
        self.distance_from_root = distance_from_root
        self.num_nodes_before_this_node = num_nodes_before_this_node