class Node():
    def __init__(self,
                 value = None,
                 row = None,
                 column = None,
                 left = None,
                 right = None,
                 up = None,
                 down = None,
                 top_left = None,
                 top_right = None,
                 bottom_left = None,
                 bottom_right = None):
        self.value = value
        self.row = row
        self.column = column
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def get_neighbours(self, node):
        return [node.top_left, node.up, node.top_right, node.right, node.bottom_right, node.down, node.bottom_left, node.left]
