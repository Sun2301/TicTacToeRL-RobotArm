"""
Game logic for Three Men's Morris.

Handles win conditions and game state validation.
"""

import numpy as np


class Game:
    """Manages game logic and win conditions."""
    
    def __init__(self, agent_symbol=1, human_symbol=-1):
        """Initialize game with player symbols."""
        self.agent_symbol = agent_symbol
        self.human_symbol = human_symbol
    
    def check_win(self, state):
        """
        Check if game has ended and return winner.
        
        Returns:
            tuple: (game_ended: bool, winner_symbol: int or 0)
        """
        state_matrix = np.reshape(state, (3, 3))
        
        # Check rows, columns, diagonals
        sum_by_row = state_matrix.sum(axis=1)
        sum_by_col = state_matrix.sum(axis=0)
        sum_diag1 = np.sum([state_matrix[i, i] for i in range(3)])
        sum_diag2 = np.sum([state_matrix[i, 2-i] for i in range(3)])
        
        all_sums = np.hstack((sum_by_row, sum_by_col, [sum_diag1, sum_diag2]))
        
        # Check for agent win
        if any(s == 3 * self.agent_symbol for s in all_sums):
            return True, self.agent_symbol
        
        # Check for human win
        if any(s == 3 * self.human_symbol for s in all_sums):
            return True, self.human_symbol
        
        return False, 0
