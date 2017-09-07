#!/usr/bin/python
# python 3.5

# By Jakob Notland
# Created for "Artificial Intelligence Programming" at norwegian university Of science and technology (NTNU)


class GenSearch:

    def best_first_search(self, initial_node, algorithm, board_file, show_process, show_solution):
        solution = None
        closed_list = []
        open_list = []
        # Add initial node to open list
        open_list.append(initial_node)
        nodes_expanded = 0
        # Find best way
        while solution is None:
            if not open_list:
                # No solution
                return None

            # Pop next node from the open list
            if algorithm == "DFS":
                current_node = open_list.pop()
            else:
                current_node = open_list.pop(0)
            # Check if we have arrived to the goal, by checking if the driver vehicle is at the goal
            if self.is_solution(current_node):
                print("Success, found solution for", algorithm, board_file)
                print("Nodes expanded:", nodes_expanded)
                path = self.backtrack_path(current_node)
                print("Path length:", len(path)-1)
                if show_solution:
                    self.print_path(path)
                return path

            closed_list.append(current_node)
            print("Expanding node:", nodes_expanded)
            if show_process:
                current_node.print_state(self.display_time)
            # Generate successor states
            children = current_node.expand_node()
            nodes_expanded += 1

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

    def attach_and_eval(self, child, parent, t=0):
        child.parent = parent
        if t:
            child.g = parent.g + t
        else :
            child.g = parent.g + 1
        child.f = child.g + child.h
        return

    def propagate_path_improvements(self, new_parent, children, t=0):
        for child in children:
            if child.parent is None or new_parent.g + 1 < child.g:
                child.parent = new_parent
                if t:
                    child.g = parent.g + parent.t
                else :
                    child.g = parent.g + 1
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

    def list_contains_board(self, array, board):
        raise NotImplementedError("Please Implement this method")

    def backtrack_path(self, current_node):
        raise NotImplementedError("Please Implement this method")

    def is_solution(self, current_node):
        raise NotImplementedError("Please Implement this method")

    def print_path(self, path):
        print("Path:")
        for state in reversed(path):
            state.print_state(self.display_time)

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
