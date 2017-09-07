#!/usr/bin/python
# python 3.5

# By Jakob Notland
# Created for "Artificial Intelligence Programming" at norwegian university Of science and technology (NTNU)

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from time import sleep
import argparse
import sys
import six.moves.cPickle as cPickle
from gen_search import GenSearch


def main():
    sys.setrecursionlimit(15000)

    # Takes input from command line
    parser = argparse.ArgumentParser(description='Solve Rush Hour')
    parser.add_argument('algorithm', type=str, help='Algorithm of choice (AStar, BFS or DFS)')
    parser.add_argument('board', type=str, help='Board text file')
    parser.add_argument('display_path', type=str2bool,
                        help='True if you want to see the solution path',
                        nargs='?')
    parser.add_argument('display_agenda', type=str2bool,
                        help='True if you want to see the entire process of the nodes expanded', nargs = '?')
    parser.add_argument('display_time', type=float,
                        help='Specify how many seconds to display each frame when in display mode', nargs = '?')
    args = parser.parse_args()

    # Solve specific problem
    if args.display_time:
        ps = ProblemSolver(args.algorithm, args.board, args.display_path, args.display_agenda, args.display_time)
    else:
        ps = ProblemSolver(args.algorithm, args.board, args.display_path, args.display_agenda)
    ps.solve_problem()


# Returns string input as boolean
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Class for solving specific problem
class ProblemSolver(GenSearch):
    def __init__(self, algorithm, board, show_solution=False, show_process=False, display_time=1):
        self.board_file = board
        self.display_time = display_time
        self.algorithm = algorithm
        if self.algorithm == 'AStar':
            # Creates board with heuristics
            self.board = Board(board)
        else:
            # Creates board without heuristics
            self.board = Board(board, calculate_h=False)
        self.goal = self.board.goal
        self.driver = self.board.vehicles[self.board.driver_index]
        self.show_process = show_process
        self.show_solution = show_solution

    # Solve given algorithm
    def solve_problem(self):
        if self.algorithm == "AStar":
            self.astar()
        elif self.algorithm == "BFS":
            self.bfs()
        else:
            self.dfs()
        return

    def print_path(self, path):
        print("Path:")
        for state in reversed(path):
            state.print_state(self.display_time)

    def astar(self):
        print("Solve with AStar")
        return GenSearch.best_first_search(self, self.board, "AStar", self.board_file, self.show_process, self.show_solution)

    def bfs(self):
        print("Solve with BFS")
        return GenSearch.best_first_search(self, self.board, "BFS", self.board_file, self.show_process, self.show_solution)

    def dfs(self):
        print("Solve with DFS")
        return GenSearch.best_first_search(self, self.board, "DFS", self.board_file, self.show_process, self.show_solution)

    def best_first_search(self):
        solution = None
        closed_list = []
        open_list = []
        # Add initial node to open list
        open_list.append(self.board)
        nodes_expanded = 0
        # Find best way
        while solution is None:
            if not open_list:
                # No solution
                return None

            # Pop next node from the open list
            if self.algorithm == "DFS":
                current_node = open_list.pop()
            else:
                current_node = open_list.pop(0)
            # Check if we have arrived to the goal, by checking if the driver vehicle is at the goal
            if self.is_solution(current_node):
                print("Success, found solution for", self.algorithm, self.board_file)
                print("Nodes expanded:", nodes_expanded)
                path = self.backtrack_path(current_node)
                print("Path length:", len(path)-1)
                return path

            closed_list.append(current_node)
            if self.show_process:
                print("Current board to expand:")
                current_node.print_state(self.display_time)
            # Generate successor states
            children = current_node.expand_node()
            nodes_expanded += 1
            print("Expanding node:", nodes_expanded)

            for child in children:
                # Use custom functions to check if the new instances already exists.
                closed_list_contains_child = self.list_contains_board(closed_list, child)
                open_list_contains_child = self.list_contains_board(open_list, child)
                if closed_list_contains_child:
                    old_child = closed_list_contains_child
                elif open_list_contains_child:
                    old_child = open_list_contains_child
                # Discover new nodes and evaluate them
                if not closed_list_contains_child and not open_list_contains_child:
                    self.attach_and_eval(child, current_node)
                    open_list.append(child)
                # If node already discovered, look for cheaper path
                elif child.g < old_child.g:
                    self.attach_and_eval(child, current_node)
                    if closed_list_contains_child:
                        self.propagate_path_improvements(current_node, children)

            if self.algorithm == "AStar":
                open_list = self.merge_sort(open_list)
        return

    # Check if the state of a Board is the goal.
    def is_solution(self, node):
        return (self.goal[0] is (node.vehicles[node.driver_index].x + node.vehicles[node.driver_index].size - 1) and
                self.goal[1] is node.vehicles[node.driver_index].y)

    # Return equal instance if exits, else return false
    def list_contains_board(self, array, board):
        if not len(array):
            return False
        for instance in array:
            vehicle_missing = False
            for i in range(len(instance.vehicles)):
                if not(instance.vehicles[i].x is board.vehicles[i].x and instance.vehicles[i].y is board.vehicles[i].y):
                    vehicle_missing = True
            if not vehicle_missing:
                return instance
        return False

    def attach_and_eval(self, child, parent):
        child.parent = parent
        # All moves have the same distance, 1
        child.g = parent.g + 1
        child.f = child.g + child.h
        return

    def propagate_path_improvements(self, new_parent, children):
        for child in children:
            if child.parent is None or new_parent.g + 1 < child.g:
                child.parent = new_parent
                child.g = new_parent.g + 1
                child.f = child.g + child.h
                self.propagate_path_improvements(child, child.expand_node(self.board.matrix))
        return children

    # find path used to arrive at node
    def backtrack_path(self, node):
        path = [node]
        x = 0
        # Get parent until we reach initial node
        while path[x].parent:
            path.append(path[x].parent)
            x += 1
        return path

    # sort the open list such that the node with lowest f value is on top (merge sort)
    def merge_sort(self, some_list):
        if len(some_list) > 1:
            mid = len(some_list) // 2
            lefthalf = some_list[:mid]
            righthalf = some_list[mid:]
            self.merge_sort(lefthalf)
            self.merge_sort(righthalf)
            i = 0
            j = 0
            k = 0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i].f < righthalf[j].f:
                    some_list[k] = lefthalf[i]
                    i += 1
                # if f is the same check h
                elif lefthalf[i].f == righthalf[j].f and lefthalf[i].h < righthalf[j].h:
                    some_list[k] = lefthalf[i]
                    i += 1
                else:
                    some_list[k] = righthalf[j]
                    j += 1
                k += 1

            while i < len(lefthalf):
                some_list[k] = lefthalf[i]
                i += 1
                k += 1

            while j < len(righthalf):
                some_list[k] = righthalf[j]
                j += 1
                k += 1
        return some_list


# Class keeping track of the rush hour puzzle board instances
class Board:

    def __init__(self, boardFile, width=6, height=6, goal=[5,2], driver_index=0, parent=None, g=0, calculate_h=True):
        self.calculate_h = calculate_h
        self.boardFile = boardFile
        self.width = width
        self.height = height
        self.goal = goal
        self.vehicles = []
        self.board = self.create_empty_board()
        self.board = self.create_board()
        self.driver_index = driver_index
        self.parent = parent
        self.g = g
        if calculate_h:
            self.h = self.calculate_heuristic()
        else:
            self.h = 0
        self.f = g + self.h

    def calculate_heuristic(self):
        # h +1 for every step to goal
        h = self.goal[0] - self.vehicles[self.driver_index].x - self.vehicles[self.driver_index].size + 1
        # h +1 for cars blocking the road
        for i in range(self.vehicles[self.driver_index].x + self.vehicles[self.driver_index].size, self.goal[0] + 1):
            if not self.board[self.goal[1]][i] is 0:
                h += 1
        return h

    # Create matrix of zeros
    def create_empty_board(self):
        board = [[0 for i in range(self.width)] for j in range(self.height)]
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
            if vehicle.orientation is 0:
                x += 1
            else:
                y += 1

    def remove_vehicle(self, vehicle):
        x = vehicle.x
        y = vehicle.y
        for i in range(vehicle.size):
            self.board[y][x] = 0
            if vehicle.orientation is 0:
                x += 1
            else:
                y += 1
        return

    def move_vehicle(self, vehicle, direction):
        self.remove_vehicle(vehicle)
        new_vehicle = cPickle.loads(cPickle.dumps(vehicle, -1))

        if vehicle.orientation is 0:
            if direction is "forward":
                new_vehicle.x += 1
            else:
                new_vehicle.x -= 1
        else:
            if direction is "forward":
                new_vehicle.y += 1
            else:
                new_vehicle.y -= 1
        self.vehicles[new_vehicle.registration-1] = new_vehicle
        self.place_vehicle(new_vehicle)
        return

    # Return all moves which don't crash, goes out of the grid or are the current nodes parent
    def get_legal_moves(self):
        legalMoves = []
        for vehicle in self.vehicles:
            # Horizontal moves
            if vehicle.orientation is 0:
                # Check forward move
                if vehicle.x+vehicle.size < self.width and not self.board[vehicle.y][vehicle.x+vehicle.size]:
                    if not self.move_leads_to_parent(vehicle.registration, vehicle.x + 1, vehicle.y):
                        legalMoves.append([vehicle, "forward"])
                # Check backward move
                if not self.board[vehicle.y][vehicle.x-1] and vehicle.x-1 > -1:
                    if not self.move_leads_to_parent(vehicle.registration, vehicle.x - 1, vehicle.y):
                        legalMoves.append([vehicle, "backward"])
            # Vertical moves
            else:
                # Check forward move
                if vehicle.y+vehicle.size < self.height and not self.board[vehicle.y+vehicle.size][vehicle.x]:
                    if not self.move_leads_to_parent(vehicle.registration, vehicle.x, vehicle.y + 1):
                        legalMoves.append([vehicle, "forward"])
                # Check backward move
                if not self.board[vehicle.y-1][vehicle.x] and vehicle.y-1 > -1:
                    if not self.move_leads_to_parent(vehicle.registration, vehicle.x, vehicle.y - 1):
                        legalMoves.append([vehicle, "backward"])
        return legalMoves

    # Check if moving vehicle of id to [x, y] leads to the state of self's parent
    def move_leads_to_parent(self, vehicle_id, x, y):
        if self.parent:
            if self.parent.vehicles[vehicle_id-1].x == x and self.parent.vehicles[vehicle_id-1].y == y:
                return True
        return False

    # Create child when created from moving vehicle in a direction
    def expand_move(self, vehicle, direction):
        # Create copy of self (Board and vehicles)
        board = cPickle.loads(cPickle.dumps(self, -1))
        board.vehicles = cPickle.loads(cPickle.dumps(self.vehicles, -1))
        board.move_vehicle(vehicle, direction)
        board.g = self.g + 1
        # Only calculates h for AStar
        if board.calculate_h:
            board.h = board.calculate_heuristic()
        board.f = board.g + board.h
        board.parent = self
        return board

    # Get all possible children (board instances from legal moves) after a move
    def expand_node(self):
        legalMoves = self.get_legal_moves()
        children = []
        for move in legalMoves:
            children.append(self.expand_move(move[0], move[1]))
        return children

    # Print the board with help from matplotlib
    def print_state(self, sleep_time):
        colormap, colorCycle, cars = self.create_colormap()
        cmap = colors.ListedColormap(colorCycle)
        norm = colors.BoundaryNorm(cars, cmap.N)
        self.plot_matrix(colormap, cmap, sleep_time)

    # Generates:
    # Numpy matrix of car ids
    # List of car ids
    # List of colors (colorCycle) one for each car
    def create_colormap(self):
        colormap = np.zeros((self.width, self.height))
        colorCycle = ['white', 'red']
        colorList = ["blue", "#CC9933", "#FF99CC", "#3300CC", "black"]
        cars = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] is 0:
                    pass
                else:
                    carId = self.board[i][j].registration
                    if not carId in cars:
                        if not len(colorCycle) == 4:
                            color = "C" + str(len(colorCycle) - 1)
                        else:
                            color = "black"
                        if len(colorCycle) > 9:
                            color = colorList.pop(0)
                        colorCycle.append(color)
                        cars.append(carId)
                    colormap[i][j] = carId
        return colormap, colorCycle, cars

    # Plots the graphical view
    def plot_matrix(self, colormap, cmap, sleep_time):
        title = 'Rush Hour \n Board: ' + str(self.boardFile).strip('.txt')
        plt.imshow(colormap, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.tight_layout()
        plt.show(block=False)
        sleep(sleep_time)
        plt.close()


# Class for managing vehicles
class Vehicle:

    index = 1

    def __init__(self, orientation, x, y, size):
        self.orientation = orientation
        self.x = x
        self.y = y
        self.size = size
        self.registration = Vehicle.index
        Vehicle.index += 1

if __name__ == '__main__':
    main()
