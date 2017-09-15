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

def main():
    ps = GAC("monograms/mono-cat.txt")
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
        self.nonogram = nonogram(board_file)
        #h = self.nonogram.calculate_heuristic(self.nonogram.nonogram)
        #self.initial_node = Board(self.nonogram.nonogram, self.nonogram.row_variables, self.nonogram.col_variables, None, h)
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


class nonogram:

    def __init__(self, file):
        self.file = file
        f = open(self.file)
        self.lines = f.readlines()
        self.width = int(self.lines[0][0])
        self.height = int(self.lines[0][2])
        self.row_specs, self.col_specs = self.store_specs(self.lines[1:])
        #self.nonogram = self.initiate_nonogram([self.width, self.height])
        # Lists containing all the possible states of all rows and columns
        self.row_variables, self.row_constraints = self.store_segments(self.row_specs, self.width)
        self.col_variables, self.col_constraints = self.store_segments(self.col_specs, self.height)
        self.create_nonogram(self.row_specs, self.row_variables, self.col_specs, self.col_variables, [self.width, self.height])

        for variable, constraint in zip(self.row_variables, self.row_constraints):
            print("row:", variable, constraint)

        print("")

        for variable, constraint in zip(self.col_variables, self.col_constraints):
            print("row:", variable, constraint)

        print("")
        for w in range(self.width):
            for h in range(self.height):
                print(w, h, ":", self.cell_constraint_satisfied(w, h))

        print("")

        #print(self.nonogram)

    # Guess nonogram shape
    def create_nonogram(self, row_specs, row_variables, col_specs, col_variables, dimensions):
        nonogram = self.initiate_nonogram(dimensions)
        new_row_variables = []
        new_col_variables = []
        for specs_in_row, variables_in_row, i in zip(row_specs, row_variables, range(len(row_specs))):
            new_variables_in_row = []
            new_variables_in_col = []
            for variable, spec in zip(variables_in_row, specs_in_row):
                new_variables_in_row.append(variable[0])
                for j in range(variable[0], variable[0] + spec):
                    nonogram[i][j] = 1

            new_row_variables.append(new_variables_in_row)

        for row in nonogram:
            pass





        print("New row variables")

        for row in new_row_variables:
            print(row)

        print("")

        print(nonogram)

        print("")

        pass

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
    def cell_constraint_satisfied(self, x, y):
        if x in self.row_variables[y]:
            if y not in self.col_variables[x]:
                return False
        if y in self.col_variables[x]:
            if x not in self.row_variables[y]:
                return False
        return True

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
        for spec in lines:
            constraint = []
            for j in range(0, len(spec) - 1, 2):
                constraint.append(int(spec[j]))
            constraints.append(constraint)
        return constraints

    # Create empty matrix
    def initiate_nonogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))

    def print_state(self, sleep_time=0):
        # todo:
        pass

    def calculate_heuristic(self, nonogram):
        # todo:
        return 1


class Board:

    def __init__(self, nonogram, row_variables, col_variables, parent, h, g=0):
        self.nonogram = nonogram
        self.g = g
        self.h = h
        self.f = g + self.h
        self.children = []
        self.parent = parent
        self.row_variables = row_variables
        self.col_variables = col_variables


if __name__ == '__main__':
    main()

