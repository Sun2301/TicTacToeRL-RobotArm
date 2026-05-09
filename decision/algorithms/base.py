"""
Abstract base class for learning algorithms.

Defines the interface that all algorithms must implement.
"""

from abc import ABC, abstractmethod


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms."""
    
    def __init__(self, board, game, alpha=0.5, gamma=0.99):
        """
        Initialize algorithm.
        
        Args:
            board: Board instance
            game: Game instance
            alpha: Learning rate
            gamma: Discount factor
        """
        self.board = board
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.player_symbol = game.agent_symbol
        self.opponent_symbol = game.human_symbol

    def set_player_symbols(self, player_symbol, opponent_symbol):
        """Bind algorithm to the symbols used by its owning agent."""
        self.player_symbol = player_symbol
        self.opponent_symbol = opponent_symbol
    
    @abstractmethod
    def load_model(self, filepath):
        """Load trained model from file."""
        pass
    
    @abstractmethod
    def save_model(self, filepath):
        """Save trained model to file."""
        pass
    
    @abstractmethod
    def initialize_state(self, state):
        """Initialize state in model if needed."""
        pass
    
    @abstractmethod
    def update_value(self, state, action, reward, next_state):
        """Update model with new experience."""
        pass
    
    @abstractmethod
    def get_best_action(self, state):
        """Get best action according to model (exploitation)."""
        pass
    
    @abstractmethod
    def get_state_size(self):
        """Return number of states learned."""
        pass
