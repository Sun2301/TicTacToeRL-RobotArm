"""
Q-Learning algorithm implementation.

Implements tabular Q-learning for Three Men's Morris.
"""

import os
import pickle
from pathlib import Path

try:
    from .base import LearningAlgorithm
except ImportError:  # pragma: no cover - compatibility for direct script imports
    from algorithms.base import LearningAlgorithm


class QLearningAlgorithm(LearningAlgorithm):
    """Q-Learning algorithm implementation."""
    
    def __init__(self, board, game, alpha=0.5, gamma=0.99):
        """
        Initialize Q-Learning algorithm.
        
        Args:
            board: Board instance
            game: Game instance
            alpha: Learning rate (default: 0.5)
            gamma: Discount factor (default: 0.99)
        """
        super().__init__(board, game, alpha, gamma)
        self.q_table = {}
    
    def load_model(self, filepath):
        """Load Q-table from file."""
        if not os.path.exists(filepath):
            print(f"No Q-table found at {filepath}. Starting fresh.")
            return

        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table from {filepath} ({len(self.q_table)} states)")
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
            print(f"Failed to load Q-table from {filepath} ({exc}). Starting fresh.")
            self.q_table = {}
    
    def save_model(self, filepath):
        """Save Q-table to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filepath} ({len(self.q_table)} states)")
    
    def initialize_state(self, state):
        """Initialize state in Q-table if not present."""
        state_tuple = self.board.state_to_tuple(state)
        if state_tuple not in self.q_table:
            available = self.board.get_available_actions(state, self.player_symbol)
            self.q_table[state_tuple] = {action: 0.0 for action in available}
    
    def update_value(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation."""
        state_tuple = self.board.state_to_tuple(state)
        next_state_tuple = self.board.state_to_tuple(next_state)
        
        self.initialize_state(state)
        self.initialize_state(next_state)

        if action not in self.q_table[state_tuple]:
            self.q_table[state_tuple][action] = 0.0
        
        max_q_next = max(self.q_table[next_state_tuple].values(), default=0.0)
        
        self.q_table[state_tuple][action] += self.alpha * (
            reward + self.gamma * max_q_next - self.q_table[state_tuple][action]
        )
    
    def get_best_action(self, state):
        """Get best action according to Q-table (exploitation)."""
        state_tuple = self.board.state_to_tuple(state)
        self.initialize_state(state)
        if not self.q_table[state_tuple]:
            return None
        return max(self.q_table[state_tuple], key=self.q_table[state_tuple].get)
    
    def get_state_size(self):
        """Return number of states in Q-table."""
        return len(self.q_table)
