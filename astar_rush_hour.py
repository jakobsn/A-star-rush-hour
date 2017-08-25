# python 3.5

# Find all possible moves
# Heurestic for all moves

# Good moves:
#   Cleares space for car
#   Move car closer
#   Clear space to clear space for car?

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def main():
    # TODO: Execute tasks
    board = Board("easy-3.txt")
    board.create_board()
    print(board.create_colormap())
    board.print_board()


class ProblemSolver:
    # TODO: Implement problemsolving for different algorithms
    def __init__(self, algorithm, board):
        self.algorithm = algorithm
        self. board = board

    def solve_problem(self):
        # TODO: Solve specific problem
        return

    def best_first_search(self):
        # TODO:
        return

    def bfs(self):
        # TODO:
        return

    def dfs(self):
        # TODO:
        return

class Board:
    def __init__(self, boardFile, width=6, height=6, goal=[5,2]):
        self.boardFile = boardFile
        self.width = width
        self.height = height
        self.goal = goal
        self.board = self.create_empty_board()
        self.board = self.create_board()

    def create_empty_board(self):
        board = [ [ None for i in range(self.width) ] for j in range(self.height) ]
        return board

    def create_board(self):
        board = self.create_empty_board()
        file = open((self.boardFile), 'r')
        for line in file:
            vehicle = self.create_vehicle(line)
            self.place_vehicle(vehicle)
        return board

    # Create vehicle object from string line
    def create_vehicle(self, line):
        line = line.split(',')
        orientation = bool(line[0])
        x = int(line[1])
        y = int(line[2])
        size = int(line[3])
        return Vehicle(orientation, x, y, size)

    # Represent the vehicle in the board
    def place_vehicle(self, vehicle):
        x = vehicle.x
        y = vehicle.y
        for i in range(vehicle.size):
            self.board[x][y] = vehicle
            if(vehicle.orientation):
                x += 1
            else:
                y += 1

    def remove_vehicle(self, vehicle):
        # TODO: Remove vehicle from grid
        return

    def move_vehicle(self, id, directopm):
        # TODO: Delete vehicle from spot and create on new spot
        return

    def print_board(self):
        colormap, colormap, cars = self.create_colormap()
        print(cars)
        cmap = colors.ListedColormap(colormap)
        print(cmap)
        norm = colors.BoundaryNorm(cars, cmap.N)
        print(norm)
        plot_matrix(colormap,cmap=cmap)


    def create_colormap(self):
        colormap = self.create_empty_board()
        colors = ['C0']
        cars = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] is None:
                    colormap[i][j] = 0
                else:
                    carId = int((self.board[i][j].x+(self.board[i][j].y/10))*10)
                    if not carId in cars:
                        color = "C" + str(len(colors))
                        colors.append(color)
                        cars.append(carId)
                    colormap[i][j] = carId
        return colormap, colors, cars

def plot_matrix(rm, title='Robot World', cmap=plt.cm.Blues):
    plt.imshow(rm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()

class Vehicle: #/?
    # TODO: class for cars
    # orientation 0 = horizontal, 1 = vertical
    def __init__(self, orientation, x, y, size):
        self.orientation = orientation
        self.x = x
        self.y = y
        self.size = size

if __name__ == '__main__':
    main()
