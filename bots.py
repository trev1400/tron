#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from collections import deque

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def __init__(self):
        self.unsafe_vals = {CellType.WALL, CellType.BARRIER, '1', '2'}
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order
        self.cutoff_ply = 4

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        start = asp.get_start_state()
        locs = start.player_locs
        board = start.board
        ptm = start.ptm
        loc = locs[ptm]

        action = self.abc_max_value(asp, start, ptm, float('-inf'), float('inf'), 0, self.cutoff_ply, board, loc)[0]

        return action

    def abc_max_value(self, asp, state, ptm, alpha, beta, depth, cutoff_ply, board, loc):

        # Check if a state is a terminal state before checking if it is at cutoff ply
        if asp.is_terminal_state(state):
            #print(state.board)
            #print("terminal state: ", asp.evaluate_state(state))
            return None, asp.evaluate_state(state)[ptm]

        if depth >= cutoff_ply:
            return None, self.eval_func_voronoi(asp, board, loc, ptm)

        value = float('-inf')
        best_action = None

        actions = list(asp.get_safe_actions(board, loc))
        #print("max actions: ", actions)
        if not actions:
            return None, float('-inf')
        for action in actions:
            #print(state.ptm)
            next_state = asp.transition(state, action)
            #print(next_state.ptm)
            #print(next_state.board)
            min_val = self.abc_min_value(asp, next_state, next_state.ptm, alpha, beta, depth + 1, cutoff_ply, next_state.board, next_state.player_locs[next_state.ptm])[1]
            if min_val > value:
                value = min_val
                best_action = action
            if value >= beta: 
                return best_action, value
            alpha = max(alpha, value)
        return best_action, value


    def abc_min_value(self, asp, state, ptm, alpha, beta, depth, cutoff_ply, board, loc):

        # Check if a state is a terminal state before checking if it is at cutoff ply
        if asp.is_terminal_state(state):
            return None, asp.evaluate_state(state)[ptm]

        if depth >= cutoff_ply:
            opp_ptm = abs(ptm-1)
            opp_loc = state.player_locs[opp_ptm]
            # We always want to calculate the voronoi from the perspective of the player (who is maximizing)
            return None, self.eval_func_voronoi(asp, board, opp_loc, opp_ptm)

        value = float('inf')
        best_action = None

        actions = list(asp.get_safe_actions(board, loc))
        #print("min actions: ", actions)
        if not actions:
            return None, float('inf')
        for action in actions:
            #print("action explored")
            #print(state.ptm)
            next_state = asp.transition(state, action)
            #print(next_state.ptm)
            #print(next_state.board)
            max_val = self.abc_max_value(asp, next_state, next_state.ptm, alpha, beta, depth + 1, cutoff_ply, next_state.board, next_state.player_locs[next_state.ptm])[1]
            if max_val < value:
                value = max_val
                best_action = action
            if value <= alpha: 
                return best_action, value
            beta = min(beta, value)
        return best_action, value

    def get_neighbors(self, board, curr_row, curr_col):
        # End row and col are 2 less than board length because of walls
        end_row = len(board)-2
        end_col = len(board[0])-2
        neighbors = []

        if (curr_row+1 <= end_row):
            neighbors.append((curr_row+1, curr_col))
        if (curr_row > 1):
            neighbors.append((curr_row-1, curr_col))
        if (curr_col+1 <= end_col):
            neighbors.append((curr_row, curr_col+1))
        if (curr_col > 1):
            neighbors.append((curr_row, curr_col-1))

        return neighbors

    def calc_distances(self, board, loc):
        # Distances 2-D list for keeping track of min distance to any given location
        distances = [[float('inf') for col in range(len(board[0]))] for row in range(len(board))]

        # Visited 2-D list for keeping track of locations we have already visited
        visited = [[False for col in range(len(board[0]))] for row in range(len(board))]

        start_row = loc[0]
        start_col = loc[1]

        distances[start_row][start_col] = 0

        # Get the immediate neighbors around the current location
        neighbors = self.get_neighbors(board, loc[0], loc[1])

        queue = deque([])

        # Add the immediate neighbors of the current location to the queue
        for n in neighbors:
            n_row = n[0]
            n_col = n[1]
            # If neighbor is a safe move, set its distance to 1 and add it to the queue
            if board[n_row][n_col] not in self.unsafe_vals:
               distances[n_row][n_col] = 1
               queue.appendleft(n)

        while len(queue) != 0:
            curr_loc = queue.pop()
            # Get the new distance that we would be able to reach neighboring locations
            new_distance = distances[curr_loc[0]][curr_loc[1]] + 1

            # Loop through the neighbors of the current location
            for n in self.get_neighbors(board, curr_loc[0], curr_loc[1]):
                n_row = n[0]
                n_col = n[1]

                # Once again check if neighbor is a safe move
                if board[n_row][n_col] not in self.unsafe_vals:

                    # If the new distance is shorter than current shortest distance, update
                    if new_distance < distances[n_row][n_col]:
                        distances[n_row][n_col] = new_distance

                    # If we haven't visited this location yet, add it to the queue and mark it as visited
                    if not visited[n_row][n_col]:
                        queue.appendleft(n)
                        visited[n_row][n_col] = True

        return distances

    def find_connected_component(self, board, component, curr_loc, visited):
        unsafe_vals = {CellType.WALL, CellType.BARRIER}
        # Mark current location as visited
        visited[curr_loc[0]][curr_loc[1]] = True
        # Add current location to component
        component.append(board[curr_loc[0]][curr_loc[1]])
        # Check all neighbors of current location 
        for n in self.get_neighbors(board, curr_loc[0], curr_loc[1]):
            n_row = n[0]
            n_col = n[1]
            # If neighbor hasn't been visited, recurse on the neighbor and check its neighbors
            if board[n_row][n_col] not in unsafe_vals and not visited[n_row][n_col]:
                component = self.find_connected_component(board, component, n, visited)
        return component

    def endgame_detection(self, board):
        unsafe_vals = {CellType.WALL, CellType.BARRIER}
        # Visited 2-D list for keeping track of locations we have already visited
        visited = [[False for col in range(len(board[0]))] for row in range(len(board))]

        # Loop through board and check connected components
        for row in range(len(board)):
            for col in range(len(board[0])):
                # Only check a location if it's a valid location and hasn't been visited
                if board[row][col] not in unsafe_vals and not visited[row][col]:
                    curr_component = []
                    curr_loc = (row, col)
                    # Gets the entire component that the current location is in
                    curr_component = (self.find_connected_component(board, curr_component, curr_loc, visited))
                    # If player and opponent are in same component, not in endgame
                    if '1' in curr_component and '2' in curr_component:
                        return False
                    # If player in component but opponent isn't, in endgame
                    elif '1' in curr_component and '2' not in curr_component:
                        return True
                    # If player isn't in component, but opponent is, in endgame
                    elif '1' not in curr_component and '2' in curr_component:
                        return True
                    # If neither player or opponent are in component, check next one
                    else: 
                        continue
        return False

    def eval_func_voronoi(self, asp, board, loc, ptm):
        opp_index = abs(ptm - 1)
        opp_loc = asp._player_locs_from_board(board)[opp_index]
        player_distances = self.calc_distances(board, loc)
        opp_distances = self.calc_distances(board, opp_loc)

        player_voronoi_size = 0
        opp_voronoi_size = 0
        total = 0
        for row in range(len(board)):
            for col in range(len(board[0])):
                if player_distances[row][col]!=0 and opp_distances[row][col]!=0:
                    if player_distances[row][col]==float('inf') and opp_distances[row][col]!=float('inf'):
                        opp_voronoi_size+=1
                        total+=1
                    elif player_distances[row][col]!=float('inf') and opp_distances[row][col]==float('inf'):
                        player_voronoi_size+=1
                        total+=1
                    elif player_distances[row][col]!=float('inf') and opp_distances[row][col]!=float('inf'):
                        if player_distances[row][col] < opp_distances[row][col]:
                            player_voronoi_size+=1
                            total+=1
                        if player_distances[row][col] > opp_distances[row][col]:
                            opp_voronoi_size+=1
                            total+=1
                    else:
                        continue
        voronoi = (player_voronoi_size - opp_voronoi_size)
        #voronoi = (((player_voronoi_size/total) - (opp_voronoi_size/total)) + 1.0) / 2.0
        #print("voronoi val: ", voronoi)
        return voronoi

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
