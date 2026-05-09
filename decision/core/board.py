"""
Board management for Three Men's Morris game.

Handles board state, operations, and display.
"""

import numpy as np


class Board:
    """Manages the board state and basic operations."""
    
    # Board structure representation:
    # | 0 | 1 | 2 |
    # | 3 | 4 | 5 |
    # | 6 | 7 | 8 |
    
    CONSTRAINT_DICT = {
        0: [1, 3, 4],
        1: [0, 2, 4],
        2: [1, 4, 5],
        3: [0, 4, 6],
        4: [0, 1, 2, 3, 5, 6, 7, 8],
        5: [2, 4, 8],
        6: [3, 4, 7],
        7: [4, 6, 8],
        8: [4, 5, 7],
    }
    
    def __init__(self):
        """Initialize empty board."""
        self.state = np.zeros((9,))
    
    def reset(self):
        """Reset board to empty state."""
        self.state = np.zeros((9,))
    
    @staticmethod
    def state_to_tuple(state):
        """Convert state array to hashable tuple."""
        return tuple(state)
    
    @staticmethod
    def show(state):
        """Display board in 3x3 grid format."""
        for i in [0, 3, 6]:
            print(f"|\t{state[i]}\t|\t{state[i+1]}\t|\t{state[i+2]}\t|")
        print()
    
    @staticmethod
    def get_available_actions(state, player_symbol):
        """
        Get available actions for a player symbol.
        
        Returns:
            list: Empty positions (deployment) or (from, to) tuples (movement)
        """
        num_pieces = np.count_nonzero(state)
        
        if num_pieces < 6:
            # Deployment phase: return empty positions
            return [i for i in range(9) if state[i] == 0]
        else:
            # Movement phase: return valid moves for player's pieces
            valid_moves = []
            for from_pos in range(9):
                if state[from_pos] == player_symbol:
                    for to_pos in Board.CONSTRAINT_DICT[from_pos]:
                        if state[to_pos] == 0:
                            valid_moves.append((from_pos, to_pos))
            return valid_moves
    
    @staticmethod
    def apply_action(state, action, player_symbol):
        """
        Apply an action to the board state.
        
        Args:
            state: Current board state
            action: int (deployment) or tuple (movement)
            player_symbol: Symbol of the player making the move
        
        Returns:
            New board state after applying action
        """
        new_state = np.array(state, copy=True)
        
        if isinstance(action, int):
            # Deployment phase
            new_state[action] = player_symbol
        elif isinstance(action, tuple):
            # Movement phase
            new_state[action[0]] = 0
            new_state[action[1]] = player_symbol
        
        return new_state
