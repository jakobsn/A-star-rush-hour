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
from random import randint, shuffle
import six.moves.cPickle as cPickle

def main():

    #ps = GAC("monograms/mono-cat.txt")
    ps = GAC(None, "AStar", "monograms/mono-cat.txt", True, True, 1)
    ps.solve_problem()
    #ps = RushHour(args.algorithm, args.board, args.display_path, args.display_agenda, args.display_time)

    """
    # Takes input from command line
    parser = argparse.ArgumentParser(description='Solve nonograms')
    parser.add_argument('nonogram', type=str, help='nonogram text file')

    args = parser.parse_args()

    # Solve specific problem
    ps = ProblemSolver(args.nonogram)
    ps.solve_problem()"""


class GAC(GenSearch):

    def __init__(self, initial_node, algorithm, board_file, show_solution=False, show_process=False, display_time=0.5):
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
        #self.nonogram.print_state(10)

        for variable, constraint in zip(self.row_variables, self.row_constraints):
            print("row:", variable, constraint)

        print("")

        for variable, constraint in zip(self.col_variables, self.col_constraints):
            print("col:", variable, constraint)

        print("")



        #self.calculate_heuristic(new_row_vars, new_col_vars, self.row_functions, self.col_functions)

        #self.nonogram.expand_node()

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
    def cell_constraint_violated(self, x, y, row_variables, col_variables, include_axis=False):
        not_satisfied = 0
        axis_violated = None

        exists_in_row = False
        exists_in_col = False
        axis = None

        for i in range(len(row_variables[y])):
            for j in range(row_variables[y][i], row_variables[y][i] + self.row_specs[y][i]):
                if j is x:
                    exists_in_row = True
                    break
            if exists_in_row:
                break

        for i in range(len(col_variables[x])):
            for j in range(col_variables[x][i], col_variables[x][i] + self.col_specs[x][i]):
                if j is y:
                    exists_in_col = True
                    break
            if exists_in_col:
                break

        if exists_in_row and not exists_in_col:
            axis = "row"
            not_satisfied = 1
        elif not exists_in_row and exists_in_col:
            axis = "col"
            not_satisfied = 1

        if include_axis:
            return not_satisfied, axis
        else:
            return not_satisfied

        """
        if x in row_variables[y]:
            if y not in col_variables[x]:
                if not include_axis:
                    return 1
                else:
                    return 1, "row"
        if y in col_variables[x]:
            if x not in row_variables[y]:
                if not include_axis:
                    return 1
                else:
                    return 1, "col"
        if not include_axis:
            return not_satisfied
        else:
            return not_satisfied, None

        """
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
        row_total_heuristic, row_axis_heuristics, row_violated_variables = self.calculate_line_heuristics(row_variables, row_functions)
        col_total_heuristic, col_axis_heuristics, col_violated_variables = self.calculate_line_heuristics(col_variables, col_functions)
        row_total_heuristic = row_total_heuristic
        col_total_heuristic = col_total_heuristic
        print("cell h:", cell_heuristic)
        print("row h:", row_total_heuristic)
        print("col h:", col_total_heuristic)
        total_heuristic = cell_heuristic + col_total_heuristic + row_total_heuristic
        return total_heuristic, row_axis_heuristics, col_axis_heuristics, row_violated_variables, col_violated_variables

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
        violated_variables = []
        for con_functions, line_variables in zip(functions, variables):
            violated_variable = []
            line_heuristic = 0
            for con_function in con_functions:
                varnames = con_function.__code__.co_varnames
                parameters = []
                for i in range(len(varnames)):
                    parameters.append(line_variables[string.ascii_lowercase.index(varnames[i])])
                if not con_function(*parameters):
                    #print("violating variables in line", i, "is", *parameters)
                    line_heuristic += 1
                    violated_variable.append(parameters)

            violated_variables.append(violated_variable)
            axis_heuristics.append(line_heuristic)
            total_heuristic += line_heuristic
        return total_heuristic, axis_heuristics, violated_variables


class Board:

    def __init__(self, csp, row_variables, col_variables, parent, g):
        self.csp = csp
        self.row_variables = row_variables
        self.col_variables = col_variables
        self.g = g
        self.h, self.row_axis_heuristics, self.col_variables_heuristics, self.row_violated_variables, self.col_violated_variables = self.csp.calculate_heuristic(self.row_variables, self.col_variables, csp.row_functions, csp.col_functions)
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

#    def expand_node(self):
#        print("vars:", self.row_variables, self.col_variables)
#        return self.csp.expand_node(self.row_variables, self.col_variables)

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
                #print("var, spec", variable, spec)
                for j in range(variable, variable + spec):
                    nonogram[i][j] += 1

        for specs_in_col, variables_in_col, i in zip(col_specs, col_variables, range(len(col_specs))):
            for variable, spec in zip(variables_in_col, specs_in_col):
                for j in range(variable, variable + spec):
                    nonogram[j][i] += 2


        """
        print("vars:")
        print("row", row_variables)
        print("col", col_variables)
        print("colormap")
        print(nonogram)

        print("")
        """
        return nonogram

    # Create empty matrix
    def initiate_nonogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))


    def min_value(self):
        pass

    def find_conflicting_axis_variable(self, axis_heuristics):
        r = list(range(len(axis_heuristics)))
        shuffle(r)
        #print(r)
        for i in r:
            s = list(range(len(axis_heuristics[i])))
            shuffle(s)
            for j in s:
                if axis_heuristics[i][j]:
                    #print("found conflicting variable")
                    return i, j

    def find_conflicting_cross_variable(self):
        x = list(range(self.csp.width))
        shuffle(x)
        y = list(range(self.csp.height))
        shuffle(y)

        #print(x)
        for i in x:
            for j in y:
                not_satisfied,  axis = self.csp.cell_constraint_violated(i, j, self.row_variables, self.col_variables, True)
                if axis:
                    #print("violating variable in", i, j, "in", axis)
                    return i, j, axis
        return
    """
    def expand_node(self):
        current = self
        method = 2#randint(0, 3)
        print("************expand**********")
        if method == 0:
            conflict = self.find_conflicting_axis_variable(self.row_axis_heuristics)
            if not conflict:
                method = randint(1, 3)
        if method == 1:
            conflict = self.find_conflicting_axis_variable(self.col_variables_heuristics)
            if not conflict:
                method = 2
        if method == 2:
            conflict_x, conflict_y = self.find_conflicting_cross_variable()
            print("conflict", conflict_x, conflict_y)


        # find min value for conflict


        # Choose random conflicted variable
        # Choose value for variable that minimizes the conflicts of the variable
        # Create child with new value for conflicted variable

    """

    def calculate_isolated_cell_heuristic(self, x, y, row_variables, col_variables, axis=None):
        cell_heuristic = self.csp.cell_constraint_violated(x, y, row_variables, col_variables)
        if (not axis or axis == "row"): #and x in self.row_violated_variables[y]:
            for j in range(len(self.row_violated_variables)):
                if x in self.row_violated_variables[y]:
                    cell_heuristic += 1
                    #print("row violation")
        if (not axis or axis == "col"): #and y in self.col_violated_variables[x]:
            for i in range(len(self.col_violated_variables[x])):
                if y in self.col_violated_variables[x][i]:
                    cell_heuristic += 1
                    #print("col violation")
        return cell_heuristic

    def generate_min_successors(self, x, y, axis):
        best_h = 999999999999
        successors = []
        new_row_variables = cPickle.loads(cPickle.dumps(self.row_variables, -1))
        new_col_variables = cPickle.loads(cPickle.dumps(self.col_variables, -1))
        old_cell_heuristic = self.calculate_isolated_cell_heuristic(x, y, new_row_variables, new_col_variables, axis)
        nonogram = None
        isolated_nonogram = None
        if axis == "row":
            #print(x, y)
            #print(self.csp.row_variables[y])
            for i in range(len(self.csp.row_variables[y])):
                if x in self.csp.row_variables[y][i]:
                    #print("old cell h:", old_cell_heuristic)
                    for variable in self.csp.row_variables[y][i]:
                        if variable is not x:
                            new_row_variables[y][i] = variable
                            successors.append(Board(self.csp, new_row_variables, new_col_variables, self, 0))
                            new_row_variables = cPickle.loads(cPickle.dumps(self.row_variables, -1))


        elif axis == "col":
            for j in range(len(self.csp.col_variables[x])):
                if y in self.csp.col_variables[x][j]:
                    for variable in self.csp.col_variables[x][j]:
                        if variable is not y:
                            new_col_variables[x][j] = variable

                            successors.append(Board(self.csp, new_row_variables, new_col_variables, self, 0))
                            new_col_variables = cPickle.loads(cPickle.dumps(self.col_variables, -1))


                            #nonogram = Board(self.csp, new_row_variables, best_col_variables, self, 0)
        #if not nonogram:
        #    self.print_state(10)
        #    return None

        return successors

    def most_conflicted_variable(self):
        most_conflicts = 0

        for x in range(len(self.col_variables)):
            for i in range(len(self.col_variables[x])):
                col_conflicts = 0
                for y in range(self.col_variables[x][i], self.col_variables[x][i] + self.csp.col_specs[x][i]):
                    col_conflicts = self.csp.cell_constraint_violated(x, y, self.row_variables, self.col_variables)
                if self.col_variables[x][i] in self.col_violated_variables[x]:
                    col_conflicts += 1
                if col_conflicts > most_conflicts:
                    most_conflicts = col_conflicts
                    conflict_x = x
                    conflict_y = self.row_variables[x][i]
                    axis = "col"
        #most_conflicted_row = None
        #most_conflicted_col = None
        for y in range(len(self.row_variables)):
            # Find most conflicting variable
            # +1 conflict for every part of segment that violates a cell constraint
            for i in range(len(self.row_variables[y])):
                row_conflicts = 0
                for x in range(self.row_variables[y][i], self.row_variables[y][i] + self.csp.row_specs[y][i]):
                    row_conflicts = self.csp.cell_constraint_violated(x, y, self.row_variables, self.col_variables)
                # +1 conflict for every local constraint violated by the variable
                if self.row_variables[y][i] in self.row_violated_variables[y]:
                    row_conflicts += 1
                if row_conflicts > most_conflicts:
                    most_conflicts = row_conflicts
                    conflict_x = self.row_variables[y][i]
                    conflict_y = y
                    axis = "row"


        print("most conflicting variable", conflict_x, conflict_y, axis, "with", most_conflicts, "conflicts")
        #self.print_state(100)


        return conflict_x, conflict_y, axis

    def expand_node(self):

        print("***********************EXPAND*********************")
        #TODO
        children = []
        axis = None
        current = self
        method = randint(0, 2)
        #print(method, "method")
        conflict_x = 0
        """
        if method == 0:

            #conflict_y, conflict_x = self.find_conflicting_axis_variable(self.row_violated_variables)
            axis = "row"
            if not conflict_x:
                method = randint(1, 2)
        if method == 1:
            #conflict_x, conflict_y = self.find_conflicting_axis_variable(self.col_violated_variables)
            axis = "col"
            if not conflict_x:
                method = 2
        if method == 2:
            conflict_x, conflict_y, axis = self.find_conflicting_cross_variable()
        print("conflict", conflict_x, conflict_y, axis)
        """
        #sleep(10)
        #self.print_state(2)
        # CHILDREN ARE THE SUCCESSORS OF THIS DOMAIN
        conflict_x, conflict_y, axis = self.most_conflicted_variable()
        print("most conflicting variable", conflict_x, conflict_y, axis)

        children = self.generate_min_successors(conflict_x, conflict_y, axis)
        #print(children, "children")
        return children





        """
        if min_succ:
            print(min_succ.h)
            #sleep(3)

        for line_domains, y in zip(self.csp.row_variables, range(len(self.csp.row_variables))):
            # Dette blir feeeeil
            for domain, x in zip(line_domains, range(len(line_domains))):
                line_children = []
                changed = False
                for domain_variable in domain:
                    new_row_variables = cPickle.loads(cPickle.dumps(self.row_variables, -1))
                    new_col_variables = cPickle.loads(cPickle.dumps(self.col_variables, -1))
                    if not(new_row_variables[y][x] is domain_variable):
                        changed = True
                        print("Change", new_row_variables[y][x], y, "from", new_row_variables[y][x], "to", domain_variable)
                        new_row_variables[y][x] = domain_variable
                        nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                        line_children.append(nonogram)


                    for col_domain, k in zip(self.csp.col_variables[domain_variable], range(len(new_col_variables))):
                    #if domain_variable in self.csp.col_variables[x]:
                        if y in col_domain:
                            if not(new_col_variables[domain_variable][k] is domain_variable):
                                changed = True
                                print(domain_variable, "in", self.csp.col_variables[domain_variable], "cor", x, y)
                                print(len(new_col_variables))
                                new_col_variables[domain_variable][k] = y
                                print("change", k, y)
                                nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                                line_children.append(nonogram)

                    if not changed:
                        print(domain_variable, "not in", self.csp.col_variables[x], "cor", x, y)
                        for i in range(self.csp.width):
                            for j in range(self.csp.height):
                                if self.csp.cell_constraint_violated(i, j, new_row_variables, new_col_variables):
                                    print("------------violated----------")
                                    for p in range(len(self.csp.row_variables[j])):
                                        if i in self.csp.row_variables[j][p] and not self.row_variables[j][p] is i:
                                            nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                                            new_row_variables[j][p] = i
                                            line_children.append(nonogram)
                                        else:
                                            for o in range(len(self.csp.col_variables[i])):
                                                if j in self.csp.col_variables[i][o] and not self.col_variables[i][o] is j:
                                                    new_col_variables[i][o] = j
                                                    nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                                                    line_children.append(nonogram)

                    #nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                    #line_children.append(nonogram)


                best_child = None
                best_heuristic = 99999999999999999999999
                for child in line_children:
                    if not child is child.parent or None:
                        if child.h <= best_heuristic:
                            best_heuristic = child.h
                            best_child = child


                if best_child:
                    #if best_heuristic < 6:
                        #best_child.print_state(6)
                    children.append(best_child)
                    print("Best change for", x, y, "with heristic", best_heuristic, "instead of", best_child.parent.h)

                if not randint(0, 30):
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print("*")
                    print(x, y)

                    x = randint(0, len(new_col_variables))
                    y = randint(0, len(new_col_variables))
                    new_value = randint(self.csp.col_variables[y][x][0], self.csp.col_variables[y][x][-1])
                    new_col_variables[x][y] = new_value
                    nonogram = Board(self.csp, new_row_variables, new_col_variables, self, 0)
                    children.append(nonogram)
                
        return children
    """

if __name__ == '__main__':
    main()

