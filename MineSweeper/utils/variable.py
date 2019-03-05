class Variable():
    def __init__(self,
                 value = None,
                 row = None,
                 column = None,
                 has_mine = None):
        self.value = value
        self.row = row
        self.column = column
        self.has_mine = has_mine
        self.constraint_equation = []
        self.is_part_of_constraint_equations = []

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def add_constraint_variable(self, variable):
        variable.is_part_of_constraint_equations.append(self.constraint_equation)
        self.constraint_equation.append(variable)

    def get_constraint_value(self):
        return np.sum([v.has_mine for v in self.constraint_equation])
