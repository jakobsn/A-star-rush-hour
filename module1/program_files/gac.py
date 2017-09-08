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
        self.monogram.create_monogram()
        pass

class Monogram:

    def __init__(self, file):
        self.file = file

    # Guess monogram shape
    def create_monogram(self):
        # todo:
        f = open(self.file)
        lines = f.readlines()
        print(lines[0])
        width = int(lines[0][0])
        height = int(lines[0][2])
        monogram = self.initiate_monogram([width, height])
        print(monogram)
        # Every row spec
        for i in range(1, height+1):
            print(lines[i])
            # Every value in each row spec
            l = 0
            for j in range(0, len(lines[i]), 2):
                print(lines[i][j])
                for k in range(l, int(lines[i][j])):
                    monogram[height - i][k + l] = 1
                    #l += 1
                l += k + 1

#                print(char)
        print(monogram)
        return monogram

    # Create empty matrix
    def initiate_monogram(self, dimensions):
        return np.zeros((dimensions[1], dimensions[0]))

    def print_state(self, sleep_time=0):
        # todo:
        pass


if __name__ == '__main__':
    main()

