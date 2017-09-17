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

def main():
    ps = GAC("monograms/mono-clover.txt")
    #ps.solve_problem()

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
        #h = self.nonogram.calculate_heuristic(self.nonogram.nonogram)
        #self.initial_node = Board(self.nonogram.nonogram, self.nonogram.row_variables, self.nonogram.col_variables, None, 0)
        self.algorithm = algorithm
        self.show_process = show_process
        self.show_solution = show_solution
        self.display_time = display_time

    def solve_problem(self):
        #TODO
        pass

    def list_contains_board(self, array, board):
        #TODO
        pass

    def is_solution(self, node):
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
        #self.nonogram = self.initiate_nonogram([self.width, self.height])
        # Lists containing all the possible states of all rows and columns
        self.row_variables, self.row_constraints = self.store_segments(self.row_specs, self.width)
        self.col_variables, self.col_constraints = self.store_segments(self.col_specs, self.height)
        self.nonogram = self.create_nonogram(self.row_specs, self.row_variables, self.col_specs, self.col_variables, [self.width, self.height])
        self.nonogram = Board(self, self.nonogram, self.row_variables, self.col_variables, None, 0)

        for variable, constraint in zip(self.row_variables, self.row_constraints):
            print("row:", variable, constraint)

        print("")

        for variable, constraint in zip(self.col_variables, self.col_constraints):
            print("row:", variable, constraint)

        print("")
        for w in range(self.width):
            for h in range(self.height):
                print(w, h, ":", self.cell_constraint_violated(w, h, self.row_variables, self.col_variables))

        print("")

        self.nonogram.print_state(0.10)

        #print(self.nonogram)

    def expand_node(self, nonogram):
        #TODO
        children = []
        return children

    # Guess nonogram shape. Picks the value of each variable to be the first in the domain
    def create_nonogram(self, row_specs, row_variables, col_specs, col_variables, dimensions):
        nonogram = self.initiate_nonogram(dimensions)
        new_row_variables = []
        for specs_in_row, variables_in_row, i in zip(row_specs, row_variables, range(len(row_specs))):
            new_variables_in_row = []
            for variable, spec in zip(variables_in_row, specs_in_row):
                new_variables_in_row.append(variable[0])
                for j in range(variable[0], variable[0] + spec):
                    nonogram[i][j] = 1
            new_row_variables.append(new_variables_in_row)

        for row in nonogram:
            pass

        new_col_variables = []
        for specs_in_col, variables_in_col, i in zip(col_specs, col_variables, range(len(col_specs))):
            new_variables_in_col = []
            for variable, spec in zip(variables_in_col, specs_in_col):
                new_variables_in_col.append(variable[0])
                for j in range(variable[0], variable[0] + spec):
                    nonogram[i][j] = 1
            new_col_variables.append(new_variables_in_col)

        print("New row variables")

        for row in new_row_variables:
            print(row)

        print("")

        print(nonogram)

        print("")

        return nonogram, new_variables_in_row, new_variables_in_col

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
            constraint = str(i + 1) + " > " + str(i) + " + " + str(segment[i])
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

    def calculate_heuristic(self, nonogram):
        # todo:
        return 1

    def calculate_line_heuristic(self, line):
        # todo
        return


class Board:

    def __init__(self, csp, nonogram, row_variables, col_variables, parent, h, g=0):
        self.csp = csp
        self.nonogram = nonogram
        self.g = g
        self.h = h
        self.f = g + self.h
        self.children = []
        self.parent = parent
        self.row_variables = row_variables
        self.col_variables = col_variables

    # Print the board with help from matplotlib
    def print_state(self, sleep_time):
        colormap = self.nonogram
        colorCycle = ['white', 'blue']
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
        return self.csp.expand_node(self.nonogram)


if __name__ == '__main__':
    main()

