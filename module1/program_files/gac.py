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
from copy import deepcopy

def main():

    #ps = GAC("monograms/mono-cat.txt")
    ps = GAC(None, "AStar", "monograms/mono-clover.txt", True, False, 1)
    ps.backtrack(ps.nonogram)
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
        #self.initial_node = self.nonogram.nonogram
        self.algorithm = algorithm
        self.show_process = show_process
        self.show_solution = show_solution
        self.display_time = display_time
        self.row_functions = self.nonogram.create_constraint_functions(self.nonogram.row_variables, self.nonogram.row_constraints)
        self.col_functions = self.nonogram.create_constraint_functions(self.nonogram.col_variables, self.nonogram.col_constraints)

#    def solve_problem(self):
#        #TODO
#        gen
#        pass

    def backtrack(self, nonogram):
        if self.is_solution(nonogram):
            return nonogram
        i, j, axis = self.select_unnassigned_variable(nonogram)
        print(i, j, axis)
        if axis == "col":
            domain = nonogram.col_variables[i][j]
        elif axis == "row":
            domain = nonogram.row_variables[i][j]

        for value in domain:
            new_nonogram = cPickle.loads(cPickle.dumps(self.nonogram, -1))

            if axis == "col":
                new_nonogram.col_variables[i][j] = value
            elif axis == "row":
                new_nonogram.row_variables[i][j] = value

            arcs = self.get_all_neighboring_arcs(i, j, axis)
            #inferences = self.inference(new_nonogram, self.get_all_neighboring_arcs(i, j, axis), axis)
            print(self.cell_constraint_satisfied(i, j, axis, new_nonogram))
            print("")

    def inference(self, nonogram, arcs, axis):


    def get_all_neighboring_arcs(self, i, j, axis):
        arcs = []
        if axis == "row":
            if len(self.nonogram.row_constraints[i]) > j:
                arcs.append(self.row_functions[i][j])
            if j - 1 >= 0:
                arcs.append(self.row_functions[i][j-1])
            arcs.append(self.cell_constraint_satisfied)

        if axis == "col":
            if len(self.nonogram.col_constraints[i]) > j:
                arcs.append(self.col_functions[i][j])
            if j - 1 >= 0:
                arcs.append(self.col_functions[i][j-1])
            arcs.append(self.cell_constraint_satisfied)
        return arcs

    def cell_constraint_satisfied(self, i, j, axis, nonogram):
        if axis == "row":
            for row_value in range(nonogram.row_variables[i][j], nonogram.row_variables[i][j] + nonogram.row_specs[i][j]):
                for k in range(len(nonogram.col_variables[row_value])):
                    for variable in nonogram.col_variables[row_value][k]:
                        for l in range(variable, nonogram.col_specs[row_value][k]):
                            if k == l:
                                return True

        if axis == "col":
            for col_value in range(nonogram.col_variables[i][j], nonogram.col_variables[i][j] + nonogram.col_specs[i][j]):
                for k in range(len(nonogram.row_variables[col_value])):
                    for variable in nonogram.row_variables[col_value][k]:
                        for l in range(variable, nonogram.row_specs[col_value][k]):
                            if k == l:
                                return True

        return False

    def select_unnassigned_variable(self, nonogram):
        for i in range(len(nonogram.row_variables)):
            for j in range(len(nonogram.row_variables[i])):
                if len(nonogram.row_variables[i][j]) > 1:
                    return i, j, "row"
        for i in range(len(nonogram.col_variables)):
            for j in range(len(nonogram.col_variables[i])):
                if len(nonogram.col_variables[i][j]) > 1:
                    return i, j, "col"

    def list_contains_board(self, array, board):
        #TODO
        if not len(array):
            return False
        for j in range(len(array)):
            if array[j].col_variables is board.col_variables and array[j].row_variables is board.row_variables:
                return j
        return False

    def is_solution(self, node):
        for row in node.row_variables:
            for domain in row:
                if len(domain) is not 1:
                    return False
        for col in node.col_variables:
            for domain in col:
                if len(domain) is not 1:
                    return False

        return True

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
        self.row_specs, self.col_specs = self.store_specs(self.lines[1:])
        self.row_variables, self.row_constraints = self.store_segments(self.row_specs, self.width)
        self.col_variables, self.col_constraints = self.store_segments(self.col_specs, self.height)

        #new_row_vars, new_col_vars = self.create_nonogram(self.row_specs, self.row_variables, self.col_specs, self.col_variables, [self.width, self.height])
        #self.nonogram = Board(self, new_row_vars, new_col_vars, None, 0)

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
        variables = []
        constraints = []
        for segment in specs:
            variables.append(self.store_segment_variables(segment, segment_length))
            constraints.append(self.store_segment_constraints(segment))
        return variables, constraints

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
        return eval("(lambda " + args + ": " + constraint + ")", envir)

    # Create both lists of constraints
    def store_specs(self, lines):
        row_constraints = self.store_spec(lines[:self.height])
        col_constraints = self.store_spec(lines[self.height:])
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
        total_heuristic = cell_heuristic + col_total_heuristic + row_total_heuristic
        return total_heuristic, row_axis_heuristics, col_axis_heuristics, row_violated_variables, col_violated_variables

    def calculate_cell_heuristics(self, row_variables, col_variables):
        cell_heuristic = 0
        for x in range(len(col_variables)):
            for y in range(len(row_variables)):
                cell_heuristic += self.cell_constraint_violated(x, y, row_variables, col_variables)
        return cell_heuristic

    def calculate_line_heuristics(self, variables, functions):
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

        return nonogram

    # Create empty matrix
    def initiate_nonogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))


    def min_value(self):
        pass

    def find_conflicting_axis_variable(self, axis_heuristics):
        conflicts = []
        r = list(range(len(axis_heuristics)))
        shuffle(r)
        #print(r)
        for i in r:
            s = list(range(len(axis_heuristics[i])))
            shuffle(s)
            for j in s:
                if axis_heuristics[i][j]:
                    #print("found conflicting variable")
                    conflicts.append([i, j])
        return conflicts


    def find_conflicting_cross_variable(self):
        x = list(range(self.csp.width))
        shuffle(x)
        y = list(range(self.csp.height))
        shuffle(y)
        for i in x:
            for j in y:
                not_satisfied,  axis = self.csp.cell_constraint_violated(i, j, self.row_variables, self.col_variables, True)
                if axis:
                    return i, j, axis
        return

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
        nonogram = None
        isolated_nonogram = None
        if axis == "row":
            for i in range(len(self.csp.row_variables[y])):
                if x in self.csp.row_variables[y][i]:
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

        return conflict_x, conflict_y, axis

    def expand_node(self):

        print("***********************EXPAND*********************")
        children = []
        conflict_x = 0

        print(type(self.col_violated_variables))
        for value in self.row_violated_variables:
            if value:
                conflicts = self.find_conflicting_axis_variable(self.row_violated_variables)
                axis = "row"
                for conflict in conflicts:
                    children += self.generate_min_successors(conflict[0], conflict[1], axis)
                break

        for value in self.col_violated_variables:
            if value:
                conflicts = self.find_conflicting_axis_variable(self.col_violated_variables)
                axis = "col"
                for conflict in conflicts:
                    children += self.generate_min_successors(conflict[0], conflict[1], axis)
                break

        conflict_x, conflict_y, axis = self.find_conflicting_cross_variable()
        if conflict_x:
            children += self.generate_min_successors(conflict_x, conflict_y, axis)

        conflict_x, conflict_y, axis = self.most_conflicted_variable()
        if conflict_x:
            children += self.generate_min_successors(conflict_x, conflict_y, axis)


        return children


if __name__ == '__main__':
    main()

