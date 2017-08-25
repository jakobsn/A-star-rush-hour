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
import itertools

def main():
    # TODO: Execute tasks
    board = Board("easy-3.txt")
    #board.create_board()

    print("Legal moves:")
    legalMoves = board.get_legal_moves()
    for move in legalMoves:
        print(move[0].x, move[0].y, move[1])
    board.print_board()

    """
    for line in board.board:
        for vehicle in line:
            if(vehicle):
                print("vehicle")
                print(vehicle)
                board.move_vehicle(vehicle, "forward")
                break
        break
    board.print_board()
    """

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
        self.vehicles = []
        self.board = self.create_empty_board()
        self.board = self.create_board()

    def create_empty_board(self):
        board = [ [ None for i in range(self.width) ] for j in range(self.height) ]
        return board

    # Fill the board with cars
    def create_board(self):
        file = open((self.boardFile), 'r')
        for line in file:
            vehicle = self.create_vehicle(line)
            self.vehicles.append(vehicle)
            self.place_vehicle(vehicle)
        return self.board

    # Create vehicle object from string line
    def create_vehicle(self, line):
        line = line.split(',')
        orientation = int(line[0])
        x = int(line[1])
        y = int(line[2])
        size = int(line[3])
        return Vehicle(orientation, x, y, size)

    # Represent the vehicle in the board
    def place_vehicle(self, vehicle):
        x = vehicle.x
        y = vehicle.y
        for i in range(vehicle.size):
            self.board[y][x] = vehicle
            if(vehicle.orientation is 0):
                x += 1
            else:
                y += 1

    def remove_vehicle(self, vehicle):
        x = vehicle.x
        y = vehicle.y
        for i in range(vehicle.size):
            self.board[x][y] = None
            if(vehicle.orientation is 0):
                x += 1
            else:
                y += 1
        return

    def move_vehicle(self, vehicle, direction):
        self.remove_vehicle(vehicle)
        if(vehicle.orientation is 0):
            if(direction is "forward"):
                vehicle.x += 1
            else:
                vehicle.x -= 1
        else:
            if(direction is "forward"):
                vehicle.y += 1
            else:
                vehicle.y -= 1
        self.place_vehicle(vehicle)
        return

    def get_legal_moves(self):
        # TODO: Get legal moves. Which are moves that dont crash cars and dont go out of the grid
        # LegalMoves format: [[$car, $direction][...]]
        legalMoves = []
        for vehicle in self.vehicles:
            print(vehicle.x, vehicle.y)
            # Horizontal moves
            if(vehicle.orientation is 0):
                # Check forward move
                if(not self.board[vehicle.x+1][vehicle.y] and vehicle.x+1 < self.width):
                    legalMoves.append([vehicle, "forward"])
                # Check backward move
                elif(not self.board[vehicle.x-1][vehicle.y] and vehicle.x-1 > -1):
                    legalMoves.append([vehicle, "backward"])
            # Vertical moves
            else:
                # Check forward move
                if(not self.board[vehicle.x][vehicle.y+1] and vehicle.y+1 < self.height):
                    legalMoves.append([vehicle, "forward"])
                # Check backward move
                elif(not self.board[vehicle.x][vehicle.y+1] and vehicle.y-1 > -1):
                    legalMoves.append([vehicle, "backward"])
        return legalMoves

    # Print the board with help from matplotlib
    def print_board(self):
        colormap, colorCycle, cars = self.create_colormap()
        cmap = colors.ListedColormap(colorCycle)
        norm = colors.BoundaryNorm(cars, cmap.N)
        self.plot_matrix(colormap,cmap)

    # Generates:
    # Numpy matrix of car ids
    # List of car ids
    # List of colors (colorCycle) one for each car
    def create_colormap(self):
        colormap = np.zeros((self.width, self.height))
        colorCycle = ['C0']
        cars = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] is None:
                    pass
                else:
                    carId = self.board[i][j].registration
                    if not carId in cars:
                        color = "C" + str(len(colorCycle))
                        colorCycle.append(color)
                        cars.append(carId)
                    colormap[i][j] = carId
        return colormap, colorCycle, cars

    # Plots the graphical view
    def plot_matrix(self, colormap, cmap, title='Rush Hour'):
        plt.imshow(colormap, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.tight_layout()
        plt.show()

class Vehicle: #/?
    # orientation 0 = horizontal, 1 = vertical
    index = 1
    def __init__(self, orientation, x, y, size):
        self.orientation = orientation
        self.x = x
        self.y = y
        self.size = size
        self.registration = Vehicle.index
        Vehicle.index += 1

# The moves represents the nodes in the path
class Move:
    def __init__(self):
        return

if __name__ == '__main__':
    main()
