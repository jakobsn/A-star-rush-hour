#!/usr/bin/python
# python 3.5

# By Jakob Notland
# Created for "Artificial Intelligence Programming" at norwegian university Of science and technology (NTNU)

# Assumptions + Constraint Violation Testing + Assumption Modification
# Generate complete assignments: assume/guess a value for each variable.
# Evaluate the assignment w.r.t. violated constraints.
# Modify the assignments to reduce the number of violations.

import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from gen_search import GenSearch
from time import sleep
import string

def main():
    ps = GAC("monograms/mono-cat.txt")
    ps.solve_problem()

    """
    # Takes input from command line
    parser = argparse.ArgumentParser(description='Solve nonograms')
    parser.add_argument('nonogram', type=str, help='nonogram text file')

    args = parser.parse_args()

    # Solve specific problem
    ps = ProblemSolver(args.nonogram)
    ps.solve_problem()"""


class GAC(GenSearch):

    def __init__(self, board_file, initial_node=None, algorithm="AStar", show_process=False, show_solution=False, display_time=0.5):
        self.board_file = board_file
        self.nonogram = Nonogram(board_file)
        self.initial_node = self.nonogram.nonogram
        self.algorithm = algorithm
        self.show_process = show_process
        self.show_solution = show_solution
        self.display_time = display_time

#    def solve_problem(self):
#        #TODO
#        gen
#        pass

    def list_contains_board(self, array, board):
        #TODO
        if not len(array):
            #print("============================= FALSE ===============================")
            return False
        for j in range(len(array)):
            if array[j].col_variables is board.col_variables and array[j].row_variables is board.row_variables:
                #print("============================= TRUE ===============================")
                return j
        #print("============================= FALSE ===============================")
        return False

    def is_solution(self, node):
        return not node.h
        #TODO
        pass

    def min_conflicts(self, csp):
        #TODO
        pass

    """
    CREATE nonoGRAM
    
    CHECK ALL CONSTRAINTS
    
    CONSTRAINTS VIOLATED = h
    
    GO TO THAT SHIT WITH SMALLEST HHHH
    
    """


class Nonogram:

    def __init__(self, file):
        self.file = file
        f = open(self.file)
        self.lines = f.readlines()
        self.dimensions = self.lines[0].split(" ")
        self.width = int(self.dimensions[0])
        self.height = int(self.dimensions[1])
        print("WH", self.width, self.height)
        self.row_specs, self.col_specs = self.store_specs(self.lines[1:])
        # Lists containing all the possible states of all rows and columns
        self.row_variables, self.row_constraints = self.store_segments(self.row_specs, self.width)
        self.col_variables, self.col_constraints = self.store_segments(self.col_specs, self.height)
        self.row_functions = self.create_constraint_functions(self.row_variables, self.row_constraints)
        self.col_functions = self.create_constraint_functions(self.col_variables, self.col_constraints)

        new_row_vars, new_col_vars = self.create_nonogram(self.row_specs, self.row_variables, self.col_specs, self.col_variables, [self.width, self.height])
        self.nonogram = Board(self, new_row_vars, new_col_vars, None, 0)

        for variable, constraint in zip(self.row_variables, self.row_constraints):
            print("row:", variable, constraint)

        print("")

        for variable, constraint in zip(self.col_variables, self.col_constraints):
            print("col:", variable, constraint)

        print("")



        #self.calculate_heuristic(new_row_vars, new_col_vars, self.row_functions, self.col_functions)
        #self.nonogram.expand_node(self.nonogram.row_variables, self.nonogram.col_variables)

        #self.nonogram.print_state(100)

    """
    def expand_node(self, row_variables, col_variables):
        #TODO
        children = []
        # maybe need to copy?
        best_heuristic = 99999999999999999999999
        #Traverse through every possible change and create children that has min conflicts in its domain
        print("parent row vars", row_variables)
        print("parent col vars", col_variables)
        new_row_variables = row_variables
        new_col_variables = col_variables

        for line_variables, line_domains, y in zip(self.row_variables, self.row_variables, range(len(row_variables))):
            for domain, x in zip(line_domains, range(len(line_variables))):
                for domain_variable in domain:
                    if not(row_variables[y][x] is domain_variable):
                        print("Change", x, y, "from", row_variables[y][x], "to", domain_variable)
                        new_row_variables[y][x] = domain_variable
                        if domain_variable in self.col_variables[x]:
                            new_col_variables[x][y] = domain_variable
                    else:
                        print("var", row_variables[y][x], "not in", domain)
                        print(type(row_variables[y][x]), type(domain[0]))
                    print("child row vars", row_variables)
                    print("child col vars", col_variables)
                    nonogram = Board(self, new_row_variables, new_col_variables, None, 0)
                    #nonogram.print_state(0)
                    children.append(nonogram)
        return children
    """
    # Guess nonogram shape. Picks the value of each variable to be the first in the domain
    def create_nonogram(self, row_specs, row_variables, col_specs, col_variables, dimensions):
        new_row_variables = []
        for specs_in_row, variables_in_row, i in zip(row_specs, row_variables, range(len(row_specs))):
            new_variables_in_row = []
            for variable, spec in zip(variables_in_row, specs_in_row):
                new_variables_in_row.append(variable[0])
            new_row_variables.append(new_variables_in_row)

        new_col_variables = []
        for specs_in_col, variables_in_col, i in zip(col_specs, col_variables, range(len(col_specs))):
            new_variables_in_col = []
            for variable, spec in zip(variables_in_col, specs_in_col):
                new_variables_in_col.append(variable[0])
            new_col_variables.append(new_variables_in_col)

        print("new", new_col_variables, new_col_variables)

        return new_row_variables, new_col_variables

    # Return all segments of one axis
    def store_segments(self, specs, segment_length):
        #segments = []
        variables = []
        constraints = []
        for segment in specs:
            #segments.append(self.store_segments(segment, segment_length))
            variables.append(self.store_segment_variables(segment, segment_length))
            constraints.append(self.store_segment_constraints(segment))
        return variables, constraints

    """
    # Return segments of one line with variables and local constraints
    def store_segments(self, segment, segment_length):
        #todo
        variables = self.store_segment_variables(segment, segment_length)
        constraints = self.store_segment_constraints(segment)
        return variables, constraints
    """
    # Return vars with domains for a segment
    def store_segment_variables(self, segment, segment_length):
        #todo?
        variables = []

        # Find length of the variables in the segment
        variables_length = 0
        for spec in segment[:-1]:
            variables_length += spec
            # Add blank space
            variables_length += 1
        variables_length += segment[-1]
        extra_space = segment_length - variables_length

        # Create a list containing the domain for each variable
        filled_space = 0
        for spec in segment:
            domains = []
            for i in range(filled_space, (filled_space + extra_space + 1)):
                domains.append(i)
            variables.append(domains)
            filled_space += spec + 1

        return variables

    # Return list of local constraints for a segment given segment constraints of one axis
    def store_segment_constraints(self, segment):
        constraints = []
        for i in range(len(segment) - 1):
            constraint = chr(ord('a') + i + 1) + " > " + chr(ord('a') + i) + " + " + str(segment[i])
            constraints.append(constraint)
        return constraints

    # Global cell constraint
    def cell_constraint_violated(self, x, y, row_variables, col_variables):
        not_satisfied = 0
        if x in row_variables[y]:
            if y not in col_variables[x]:
                not_satisfied += 1
        if y in col_variables[x]:
            if x not in row_variables[y]:
                not_satisfied += 1
        return not_satisfied

    """
    # Guess nonogram shape
    def create_nonogram(self):
        # todo:

        row_cons = []
        column_cons = []
        print(nonogram)
        # Every row spec
        for i in range(1, height+1):
            print(lines[i])
            # Every value in each row spec
            row_con = []
            for j in range(0, len(lines[i]), 2):
                #print(lines[i][j])
                row_con.append(lines[i][j])
                for k in range(int(lines[i][j])):
                    nonogram[height - i][k] = 1
            row_cons.append(row_con)
        # Every column spec
        for i in range(height + 1, width + 1):
            pass

#                print(char)
        print("rows", row_cons)
        print(nonogram)
        return nonogram, row_cons#, column
    """

    def create_constraint_functions(self, variables, constraints):
        constraint_functions = []
        for line_variables, line_constraints in zip(variables, constraints):
            line_functions = []
            for constraint in line_constraints:
                line_functions.append(self.create_constraint_function(line_variables, constraint))
            constraint_functions.append(line_functions)
        return constraint_functions

    def create_constraint_function(self, variables, constraint, envir=globals()):
        args = ""
        for i in range(len(variables)):
            if chr(ord('a') + i) in constraint:
                args += ",".join(chr(ord('a') + i))
                args += ","
        args = args[:-1]
        #TODO
        print("args", args)
        print("constraint", constraint)
        return eval("(lambda " + args + ": " + constraint + ")", envir)

    # Create both lists of constraints
    def store_specs(self, lines):
        row_constraints = self.store_spec(lines[:self.height])
        col_constraints = self.store_spec(lines[self.height:])
        print("row_cons", row_constraints)
        print("col_cons", col_constraints)
        return row_constraints, col_constraints

    # Create a list of constraints
    def store_spec(self, lines):
        constraints = []
        for segment in lines:
            constraint = []
            for spec in segment.replace("\n", "").split(" "):
                constraint.append(int(spec))
            constraints.append(constraint)
        return constraints

    # Create empty matrix
    def initiate_nonogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))

    def calculate_heuristic(self, row_variables, col_variables, row_functions, col_functions):
        cell_heuristic = self.calculate_cell_heuristics(row_variables, col_variables)
        row_total_heuristic, row_axis_heuristics = self.calculate_line_heuristics(row_variables, row_functions)
        col_total_heuristic, col_axis_heuristics = self.calculate_line_heuristics(col_variables, col_functions)
        print("cell h:", cell_heuristic)
        print("row h:", row_total_heuristic)
        print("col h:", col_total_heuristic)
        total_heuristic = cell_heuristic + col_total_heuristic + row_total_heuristic
        return total_heuristic, row_axis_heuristics, col_axis_heuristics

    def calculate_cell_heuristics(self, row_variables, col_variables):
        cell_heuristic = 0
        for x in range(len(col_variables)):
            for y in range(len(row_variables)):
                #print(w, h, ":", self.cell_constraint_violated(w, h, new_row_vars, new_col_vars))

                cell_heuristic += self.cell_constraint_violated(x, y, row_variables, col_variables)
        print("cell heuristic", cell_heuristic)
        return cell_heuristic

    def calculate_line_heuristics(self, variables, functions):
        # todo
        total_heuristic = 0
        axis_heuristics = []
        #print("line h")
        for con_functions, line_variables in zip(functions, variables):
            line_heuristic = 0
            for con_function in con_functions:
                varnames = con_function.__code__.co_varnames
                parameters = []
                for i in range(len(varnames)):
                #for varname in con_function.__code__.co_varnames:
                    #print("varname", varnames[i])
                    parameters.append(line_variables[string.ascii_lowercase.index(varnames[i])])
                #print("fparams:", varnames)
                #print("params:", parameters)
                if not con_function(*parameters):
                    #print("funct:", False)
                    line_heuristic += 1
                #else:
                    #print("funct:", True)
            axis_heuristics.append(line_heuristic)
            total_heuristic += line_heuristic
        return total_heuristic, axis_heuristics


class Board:

    def __init__(self, csp, row_variables, col_variables, parent, g):
        self.csp = csp
        self.row_variables = row_variables
        self.col_variables = col_variables
        self.g = g
        self.h, self.row_axis_heuristics, self.col_variables_heuristics = self.csp.calculate_heuristic(self.row_variables, self.col_variables, csp.row_functions, csp.col_functions)
        self.f = self.g + self.h
        self.children = []
        self.parent = parent
        print("")
        print("Board created with h:", self.h)


    # Print the board with help from matplotlib
    def print_state(self, sleep_time):
        colormap = self.create_colormap(self.csp.row_specs, self.row_variables, self.csp.col_specs, self.col_variables, [self.csp.width, self.csp.height])
        colorCycle = ['white', 'blue', 'red', 'green']
        cmap = colors.ListedColormap(colorCycle)
        self.plot_matrix(colormap, cmap, sleep_time)

    # Plots the graphical view
    def plot_matrix(self, colormap, cmap, sleep_time):
        title = 'Nonogram'
        plt.imshow(colormap, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.show(block=False)
        sleep(sleep_time)
        plt.close()

    def expand_node(self):
        return self.csp.expand_node(self.row_variables, self.col_variables)

    # Guess nonogram shape. Picks the value of each variable to be the first in the domain
    def create_colormap(self, row_specs, row_variables, col_specs, col_variables, dimensions):
        if row_variables == self.csp.row_variables:
            print("********************FAIL**************************")
        else:
            print("not")
            print(row_variables)
        nonogram = self.initiate_nonogram(dimensions)
        for specs_in_row, variables_in_row, i in zip(row_specs, row_variables, range(len(row_specs))):
            for variable, spec in zip(variables_in_row, specs_in_row):
                for j in range(variable, variable + spec):
                    nonogram[j][i] += 1

        for specs_in_col, variables_in_col, i in zip(col_specs, col_variables, range(len(col_specs))):
            for variable, spec in zip(variables_in_col, specs_in_col):
                for j in range(variable, variable + spec):
                    nonogram[i][j] += 2



        print("")

        print("colormap", nonogram)

        print("")

        return nonogram

    # Create empty matrix
    def initiate_nonogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))

    def expand_node(self):
        #TODO
        children = []
        # maybe need to copy?
        best_heuristic = 99999999999999999999999
        """
        Traverse through every possible change and create children that has min conflicts in its domain
        """
        #print("parent row vars", row_variables)
        #print("parent col vars", col_variables)
        #new_row_variables = self.row_variables
        #new_col_variables = self.col_variables

        for line_variables, line_domains, y in zip(self.csp.row_variables, self.csp.row_variables, range(len(self.csp.row_variables))):
            new_row_variables = self.row_variables
            new_col_variables = self.col_variables
            for domain, x in zip(line_domains, range(len(line_variables))):
                line_children = []
                for domain_variable in domain:
                    if not(self.csp.row_variables[y][x] is domain_variable):
                        print("Change", x, y, "from", new_row_variables[y][x], "to", domain_variable)
                        new_row_variables[y][x] = domain_variable
                        if domain_variable in self.csp.col_variables[x]:
                            self.csp.col_variables[x][y] = domain_variable
                    else:
                        print("var", new_row_variables[y][x], "not in", domain)
                        print(type(new_row_variables[y][x]), type(domain[0]))
                    print("child row vars", new_row_variables)
                    print("child col vars", new_col_variables)
                    nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                    line_children.append(nonogram)
                best_child = None
                for child in line_children:
                    if not child is child.parent:
                        children.append(child)
                    """
                    if child.h < best_heuristic:
                        best_heuristic = child.h
                        best_child = child
                    else:
                        print('bad h', child.h, "worse than", best_heuristic)
                                    if best_child:
                    if self.parent:
                    children.append(best_child)
                    """
                #best_child.print_state(5)

        return children

if __name__ == '__main__':
    main()

