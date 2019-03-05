class Variable():
    def __init__(self,
                 value = None,
                 row = None,
                 column = None,
                 constraint_value = None,
                 has_mine = None):
        self.value = value
        self.row = row
        self.column = column
        self.has_mine = has_mine
        self.constraint_equation = []
        self.constraint_value = constraint_value
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
        self.constraint_value = np.sum([v.has_mine for v in self.constraint_equation])
