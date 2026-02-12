"""
Utility functions for visualization and plotting.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_progress(episode_rewards, window_size=5000, title="Agent Learning Progress"):
    """
    Plot agent's learning progress using moving average.
    
    Args:
        episode_rewards: List of rewards for each episode
        window_size: Window size for moving average
        title: Title for the plot
    """
    rewards_series = pd.Series(episode_rewards)
    moving_avg = rewards_series.rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f"Moving Avg (window={window_size})")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
