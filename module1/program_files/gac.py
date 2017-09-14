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

def main():
    ps = ProblemSolver("monograms/nono-cat.txt")
    ps.solve_problem()

    """
    # Takes input from command line
    parser = argparse.ArgumentParser(description='Solve monograms')
    parser.add_argument('monogram', type=str, help='monogram text file')

    args = parser.parse_args()

    # Solve specific problem
    ps = ProblemSolver(args.monogram)
    ps.solve_problem()"""


class ProblemSolver:

    def __init__(self, file):
        self.file = file
        self.monogram = Monogram(file)

    def solve_problem(self):
        #TODO
        pass

class Monogram:

    def __init__(self, file):
        self.file = file
        f = open(self.file)
        self.lines = f.readlines()
        self.width = int(self.lines[0][0])
        self.height = int(self.lines[0][2])
        self.row_constraints, self.col_constraints = self.store_constraints()
        self.monogram = self.initiate_monogram([self.width, self.height])
        #self.monogram = self.create_monogram()
        # Lists containing all the possible states of all rows and columns
        self.row_segments = self.store_segments(self.row_constraints, self.width)
        self.col_segments = self.store_segments(self.col_constraints, self.height)

    # Guess monogram shape
    def create_monogram(self):
        #todo?
        pass

    # Return all segments of one axis
    def store_segments(self, constraints, segment_length):
        segments = []
        for segment in constraints:
            segments.append(self.store_single_segment(segment, segment_length))
        return segments

    # Return single segment with variables and local constraints
    def store_single_segment(self, segment, segment_length):
        #todo
        variables = self.store_segment_variables(segment, segment_length)
        constraints = self.store_segment_constraints(segment)
        return variables, constraints

    # Return vars with domains for a segment
    def store_segment_variables(self, segment, segment_length):
        #todo?
        variables = []
        variables_lenght = 0
        for spec in segment[:-1]:
            variables_lenght += spec
            # Add blank space
            variables_lenght += 1
        variables_lenght += segment[-1]

        extra_space = segment_length - variables_lenght
        domains = []

        # Store first variable
        for i in range(extra_space):
            domains.append(i)
        variables.append(domains)

        return variables

    # Return list of local constraints for a segment given segment constraints of one axis
    def store_segment_constraints(self, segment):
        #todo
        constraints = []
        return constraints




    """
    # Guess monogram shape
    def create_monogram(self):
        # todo:

        row_cons = []
        column_cons = []
        print(monogram)
        # Every row spec
        for i in range(1, height+1):
            print(lines[i])
            # Every value in each row spec
            row_con = []
            for j in range(0, len(lines[i]), 2):
                #print(lines[i][j])
                row_con.append(lines[i][j])
                for k in range(int(lines[i][j])):
                    monogram[height - i][k] = 1
            row_cons.append(row_con)
        # Every column spec
        for i in range(height + 1, width + 1):
            pass

#                print(char)
        print("rows", row_cons)
        print(monogram)
        return monogram, row_cons#, column
    """

    # Create both lists of constraints
    def store_constraints(self):
        row_constraints = self.store_spec_constraints(self.lines[:self.height])
        col_constraints = self.store_spec_constraints(self.lines[self.height:])
        print("row_cons", row_constraints)
        print("col_cons", col_constraints)
        return row_constraints, col_constraints

    # Create a list of constraints
    def store_spec_constraints(self, lines):
        constraints = []
        for spec in lines:
            constraint = []
            for j in range(0, len(spec) - 1, 2):
                constraint.append(int(spec[j]))
            constraints.append(constraint)
        return constraints

    # Create empty matrix
    def initiate_monogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))

    def print_state(self, sleep_time=0):
        # todo:
        pass


if __name__ == '__main__':
    main()

