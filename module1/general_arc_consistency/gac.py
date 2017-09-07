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
    # Takes input from command line
    parser = argparse.ArgumentParser(description='Solve monograms')
    parser.add_argument('monogram', type=str, help='monogram text file')

    args = parser.parse_args()

    # Solve specific problem
    ps = ProblemSolver(args.monogram)
    ps.solve_problem()


class ProblemSolver():

    def __init__(self, file):
        self.file = file

    def solve_problem(self):
        #TODO



if __name__ == '__main__':
    main()

