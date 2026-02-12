#!/usr/bin/env python3
"""
Quick train and play script.

Quick training session followed by interactive play.
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
    algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    agent = Agent(algorithm, symbol=1, model_filepath='models/q_table.pkl')
    
    print("\n" + "="*60)
    print("Quick Train + Play")
    print("="*60)
    
    # Quick training
    print("Training for 50k episodes...")
    agent.train(episodes=50000)
    agent.algorithm.save_model(str(agent.model_filepath))
    print(f"✓ Training complete ({agent.algorithm.get_state_size()} states)\n")
    
    # Play
    agent.play_interactive(difficulty='medium', save_after=True)
