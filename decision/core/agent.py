"""
Agent for playing Three Men's Morris.

Algorithm-agnostic agent that can use any learning algorithm.
"""

import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


class Agent:
    """Agent that plays using a learning algorithm."""

    WIN_REWARD = 60
    LOSE_REWARD = -50
    DRAW_REWARD = 1

    DIFFICULTY_MODES = {
        "easy": {"epsilon": 0.3, "description": "Easy: Agent explores more"},
        "medium": {"epsilon": 0.12, "description": "Medium: Balanced play"},
        "hard": {"epsilon": 0.05, "description": "Hard: Agent mostly exploits"},
    }

    def __init__(self, algorithm, symbol=1, model_filepath="models/q_table.pkl"):
        """
        Initialize agent.

        Args:
            algorithm: LearningAlgorithm instance
            symbol: Agent's symbol
            model_filepath: Path to model file
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.opponent_symbol = -1 if symbol == 1 else 1
        self.model_filepath = self._resolve_model_path(model_filepath)

        self.algorithm.set_player_symbols(self.symbol, self.opponent_symbol)
        self.algorithm.load_model(str(self.model_filepath))

    @staticmethod
    def _parse_human_action(action_str):
        """Parse user input as deploy int or move tuple."""
        cleaned = action_str.strip().replace(" ", "")
        if not cleaned:
            raise ValueError("empty input")

        if "," not in cleaned:
            return int(cleaned)

        cleaned = cleaned.strip("()")
        parts = cleaned.split(",")
        if len(parts) != 2:
            raise ValueError("invalid move tuple")
        return (int(parts[0]), int(parts[1]))

    def _resolve_model_path(self, model_filepath):
        """Resolve model path relative to decision module when needed."""
        model_path = Path(model_filepath)
        if model_path.is_absolute():
            return model_path

        cwd_candidate = Path.cwd() / model_path
        if cwd_candidate.exists():
            return cwd_candidate

        package_root = Path(__file__).resolve().parents[1]
        return package_root / model_path

    def choose_action(self, state, epsilon=0.0):
        """
        Choose action using ε-greedy strategy.

        Args:
            state: Current board state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        available = self.algorithm.board.get_available_actions(state, self.symbol)
        if not available:
            return None

        if random.uniform(0, 1) < epsilon:
            return random.choice(available)

        action = self.algorithm.get_best_action(state)
        return action if action in available else random.choice(available)

    def play_game(self, opponent_algorithm=None, epsilon=0.0, alpha=0.5, learning=True):
        """
        Play a single game, optionally learning from it.

        Args:
            opponent_algorithm: Learning algorithm for opponent (None = random)
            epsilon: Exploration rate
            alpha: Learning rate for this game
            learning: Whether to update model

        Returns:
            tuple: (winner_symbol, episode_reward)
        """
        self.algorithm.alpha = alpha
        state = np.zeros((9,))
        episode_reward = 0

        while True:
            available = self.algorithm.board.get_available_actions(state, self.symbol)
            if not available:
                episode_reward += self.DRAW_REWARD
                return 0, episode_reward

            agent_state = np.array(state, copy=True)
            agent_action = self.choose_action(state, epsilon)
            state = self.algorithm.board.apply_action(state, agent_action, self.symbol)

            game_ended, winner = self.algorithm.game.check_win(state)
            if game_ended:
                reward = self.WIN_REWARD if winner == self.symbol else self.LOSE_REWARD
                episode_reward += reward
                if learning:
                    self.algorithm.update_value(agent_state, agent_action, reward, state)
                return winner, episode_reward

            available = self.algorithm.board.get_available_actions(state, self.opponent_symbol)
            if not available:
                episode_reward += self.DRAW_REWARD
                if learning:
                    self.algorithm.update_value(agent_state, agent_action, self.DRAW_REWARD, state)
                return 0, episode_reward

            if opponent_algorithm is None:
                opponent_action = random.choice(available)
            else:
                opponent_algorithm.initialize_state(state)
                opponent_action = opponent_algorithm.get_best_action(state)
                if opponent_action not in available:
                    opponent_action = random.choice(available)

            next_state = self.algorithm.board.apply_action(
                state, opponent_action, self.opponent_symbol
            )

            game_ended, winner = self.algorithm.game.check_win(next_state)
            if game_ended:
                reward = self.LOSE_REWARD if winner == self.opponent_symbol else self.WIN_REWARD
                episode_reward += reward
                if learning:
                    self.algorithm.update_value(agent_state, agent_action, reward, next_state)
                return winner, episode_reward

            if learning:
                self.algorithm.update_value(agent_state, agent_action, 0, next_state)
            state = next_state

    def train(
        self,
        episodes=800000,
        epsilon_initial=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.9995,
        alpha=0.5,
        opponent_algorithm=None,
    ):
        """
        Train the agent through self-play.

        Args:
            episodes: Number of training episodes
            epsilon_initial: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exponential decay rate per episode
            alpha: Learning rate
            opponent_algorithm: Learning algorithm for opponent (None = random)

        Returns:
            list: Episode rewards for tracking progress
        """
        self.algorithm.alpha = alpha
        episode_rewards = []

        for episode in tqdm(range(episodes)):
            current_epsilon = max(
                epsilon_min, epsilon_initial * (epsilon_decay**episode)
            )

            _, reward = self.play_game(
                opponent_algorithm=opponent_algorithm,
                epsilon=current_epsilon,
                alpha=alpha,
                learning=True,
            )

            episode_rewards.append(reward)

        return episode_rewards

    def play_interactive(self, difficulty="medium", save_after=True):
        """
        Play an interactive game with human.

        Args:
            difficulty: 'easy', 'medium', or 'hard'
            save_after: Save model after game
        """
        selected_difficulty = difficulty if difficulty in self.DIFFICULTY_MODES else "medium"
        mode = self.DIFFICULTY_MODES[selected_difficulty]
        epsilon = mode["epsilon"]

        print("\n" + "=" * 60)
        print("Welcome to Three Men's Morris!")
        print("=" * 60)
        print(f"Difficulty: {selected_difficulty.upper()} {mode['description']}")
        print(f"You: {self.opponent_symbol} | Agent: {self.symbol}")
        print("=" * 60 + "\n")

        state = np.zeros((9,))

        while True:
            self.algorithm.board.show(state)
            print("Agent's turn...")

            available = self.algorithm.board.get_available_actions(state, self.symbol)
            if not available:
                print("Agent has no moves. Draw!")
                break

            agent_state = np.array(state, copy=True)
            agent_action = self.choose_action(state, epsilon)
            print(f"Agent plays: {agent_action}\n")
            state = self.algorithm.board.apply_action(state, agent_action, self.symbol)

            game_ended, winner = self.algorithm.game.check_win(state)
            if game_ended:
                self.algorithm.board.show(state)
                if winner == self.symbol:
                    print("Agent wins!")
                    self.algorithm.update_value(
                        agent_state, agent_action, self.WIN_REWARD, state
                    )
                else:
                    print("You win!")
                break

            print("Your turn...")
            available = self.algorithm.board.get_available_actions(state, self.opponent_symbol)
            if not available:
                print("You have no moves. Draw!")
                self.algorithm.update_value(
                    agent_state, agent_action, self.DRAW_REWARD, state
                )
                break

            print(f"Available moves: {available}")
            print("Format: single number (0-8) for deploy, or from,to / (from,to) for move")

            while True:
                try:
                    action = self._parse_human_action(input("Your move: "))
                    if action in available:
                        break
                    print(f"Invalid move. Available: {available}")
                except (ValueError, IndexError):
                    print("Invalid format. Try again.")

            print(f"You play: {action}\n")
            next_state = self.algorithm.board.apply_action(state, action, self.opponent_symbol)

            game_ended, winner = self.algorithm.game.check_win(next_state)
            if game_ended:
                self.algorithm.board.show(next_state)
                if winner == self.opponent_symbol:
                    print("You win!")
                    self.algorithm.update_value(
                        agent_state, agent_action, self.LOSE_REWARD, next_state
                    )
                else:
                    print("Agent wins!")
                break

            self.algorithm.update_value(agent_state, agent_action, 0, next_state)
            state = next_state

        if save_after:
            self.algorithm.save_model(str(self.model_filepath))

        print("Thanks for playing!\n")
