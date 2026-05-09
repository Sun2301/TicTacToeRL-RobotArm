#!/usr/bin/env python3
"""
Play script for interactive game against Q-Learning agent.

Play against the trained agent with selectable difficulty.
"""

import sys
import os

if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, REPO_ROOT)

from decision.core.board import Board
from decision.core.game import Game
from decision.algorithms.qlearning import QLearningAlgorithm
from decision.core.agent import Agent

if __name__ == "__main__":
    # Initialize components
    board = Board()
    game = Game(agent_symbol=1, human_symbol=-1)
    algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    agent = Agent(algorithm, symbol=1, model_filepath='models/q_table.pkl')
    
    print("\n" + "="*60)
    print("Three Men's Morris - Play Against Agent")
    print("="*60)
    print("Difficulty levels:")
    print("1. Easy   - Agent explores more")
    print("2. Medium - Balanced play")
    print("3. Hard   - Agent mostly exploits")
    print("="*60)
    
    choice = input("Select difficulty (1-3): ").strip()
    
    difficulty_map = {'1': 'easy', '2': 'medium', '3': 'hard'}
    difficulty = difficulty_map.get(choice, 'medium')
    
    agent.play_interactive(difficulty=difficulty, save_after=True)
    print("Q-table updated and saved!")
