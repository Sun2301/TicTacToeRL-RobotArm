#!/usr/bin/env python3
"""
Training script for Q-Learning agent.

Trains both adversary and main agent through self-play.
"""

import sys
import os

if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, REPO_ROOT)

from decision.core.board import Board
from decision.core.game import Game
from decision.core.agent import Agent
from decision.algorithms.qlearning import QLearningAlgorithm

if __name__ == "__main__":
    # Initialize components
    board = Board()
    game = Game(agent_symbol=1, human_symbol=-1)
    
    # Train adversary
    print("="*60)
    print("Training Adversary Agent")
    print("="*60)
    adversary_algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    adversary = Agent(adversary_algorithm, symbol=-1, 
                               model_filepath='models/q_table_adversary.pkl')
    
    adversary.train(episodes=800000, opponent_algorithm=None)
    adversary.algorithm.save_model(str(adversary.model_filepath))
    print(f"✓ Adversary training complete ({adversary.algorithm.get_state_size()} states learned)\n")
    
    # Train main agent
    print("="*60)
    print("Training Main Agent")
    print("="*60)
    algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    agent = Agent(algorithm, symbol=1, model_filepath='models/q_table.pkl')
    
    rewards_main = agent.train(episodes=800000, opponent_algorithm=adversary_algorithm)
    agent.algorithm.save_model(str(agent.model_filepath))
    print(f"✓ Main agent training complete ({agent.algorithm.get_state_size()} states learned)\n")
    
    # Plot results
    from decision.core.utils import plot_learning_progress

    print("Plotting learning progress...")
    plot_learning_progress(rewards_main)
    print("\n✓ Training finished! Q-tables saved to models/")
