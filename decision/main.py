#!/usr/bin/env python3
"""
Main entry point for Three Men's Morris RL Agent.

Interactive menu for training and playing against the agent.
"""

import sys
import os

if __package__ is None or __package__ == "":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, REPO_ROOT)

from decision.core.board import Board
from decision.core.game import Game
from decision.core.agent import Agent
from decision.algorithms.qlearning import QLearningAlgorithm


def main():
    """Main entry point with interactive menu."""
    
    print("\n" + "="*60)
    print("Three Men's Morris - Reinforcement Learning Agent")
    print("="*60)
    print("Algorithms: Q-Learning (extensible for DQN, etc.)")
    print("="*60 + "\n")
    
    # Initialize core components
    board = Board()
    game = Game(agent_symbol=1, human_symbol=-1)
    
    # Create algorithm (Q-Learning by default)
    algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    
    # Create agent
    agent = Agent(algorithm, symbol=1, model_filepath='models/q_table.pkl')
    
    print("="*60)
    print("MENU")
    print("="*60)
    print("1. Train agent (800k episodes with adversary)")
    print("2. Play game (easy)")
    print("3. Play game (medium)")
    print("4. Play game (hard)")
    print("5. Quick train (50k) then play")
    print("="*60)
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == '1':
        train_full(board, game, agent)
    elif choice in ['2', '3', '4']:
        difficulty_map = {'2': 'easy', '3': 'medium', '4': 'hard'}
        agent.play_interactive(difficulty=difficulty_map[choice])
    elif choice == '5':
        quick_train_and_play(board, game, agent)
    else:
        print("Invalid option!")


def train_full(board, game, agent):
    """Full training with adversary."""
    from decision.core.utils import plot_learning_progress
    
    print("\n" + "="*60)
    print("Training Adversary Agent")
    print("="*60)
    
    adversary_algorithm = QLearningAlgorithm(board, game, alpha=0.5, gamma=0.99)
    adversary = Agent(adversary_algorithm, symbol=-1, 
                     model_filepath='models/q_table_adversary.pkl')
    
    adversary.train(episodes=800000, opponent_algorithm=None)
    adversary.algorithm.save_model(str(adversary.model_filepath))
    print(f"✓ Adversary training complete\n")
    
    print("="*60)
    print("Training Main Agent Against Adversary")
    print("="*60)
    
    rewards_main = agent.train(episodes=800000, opponent_algorithm=adversary_algorithm)
    agent.algorithm.save_model(str(agent.model_filepath))
    print(f"✓ Main agent training complete\n")
    
    print("Plotting learning progress...")
    plot_learning_progress(rewards_main)
    print("\n✓ Training finished!")


def quick_train_and_play(board, game, agent):
    """Quick training followed by play."""
    
    print("\n" + "="*60)
    print("Quick Training (50k episodes)")
    print("="*60)
    
    agent.train(episodes=50000)
    agent.algorithm.save_model(str(agent.model_filepath))
    print(f"✓ Training complete ({agent.algorithm.get_state_size()} states)\n")
    
    print("Now playing...\n")
    agent.play_interactive(difficulty='medium', save_after=True)


if __name__ == "__main__":
    main()
