#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

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

        cutoff_ply = 10

        action = self.abc_max_value(asp, start, ptm, float('-inf'), float('inf'), 0, cutoff_ply, self.eval_func, board, loc)[0]
        return action
    
    def abc_max_value(self, asp, state, ptm, alpha, beta, depth, cutoff_ply, eval_func, board, loc):

        # Check if a state is a terminal state before checking if it is at cutoff ply
        if asp.is_terminal_state(state):
            return None, asp.evaluate_state(state)[ptm]

        actions = list(TronProblem.get_safe_actions(board, loc))

        if depth >= cutoff_ply:
            return None, eval_func(actions)

        value = float('-inf')
        bestAction = None

        for action in actions:
            nextState = asp.transition(state, action)
            minVal = self.abc_min_value(asp, nextState, ptm, alpha, beta, depth + 1, cutoff_ply, eval_func, board, loc)[1]
            if minVal > value:
                value = minVal
                bestAction = action
            if value >= beta: return bestAction, value
            alpha = max(alpha, value)
        
        return bestAction, value


    def abc_min_value(self, asp, state, ptm, alpha, beta, depth, cutoff_ply, eval_func, board, loc):

        # Check if a state is a terminal state before checking if it is at cutoff ply
        if asp.is_terminal_state(state):
            return None, asp.evaluate_state(state)[ptm]

        actions = list(TronProblem.get_safe_actions(board, loc))
        
        if depth >= cutoff_ply:
            return None, eval_func(actions)

        value = float('inf')
        bestAction = None

        for action in actions:
            nextState = asp.transition(state, action)
            maxVal = self.abc_max_value(asp, nextState, ptm, alpha, beta, depth + 1, cutoff_ply, eval_func, board, loc)[1]
            if maxVal < value:
                value = maxVal
                bestAction = action
            if value <= alpha: return bestAction, value
            beta = min(beta, value)

        return bestAction, value
    
    def eval_func(self, actions):
        # 4 is maximum number of safe actions that can be taken
        return len(actions)/(4)

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
