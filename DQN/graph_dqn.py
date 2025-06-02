# plot_graph.py

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def running_avg(x, N=100):
    """Compute the running average of array x with window size N."""
    return np.convolve(x, np.ones(N)/N, mode="valid")

def plot_training_stats(npz_path):
    """
    Given a .npz file containing at least 'rewards' and 'eps' arrays (and optionally 'loss'),
    plot:
      1. Total reward per episode (raw and 100-episode moving average)
      2. Epsilon schedule over episodes
      3. TD-loss over learning steps (log scale), if present
    """
    data = np.load(npz_path)
    files = set(data.files)

    # 1) Plot rewards
    if "rewards" not in files:
        raise KeyError(f"'rewards' not found in {npz_path}. Available keys: {files}")
    rewards = data["rewards"]

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, alpha=0.3, label="Reward (raw)")
    if rewards.size >= 100:
        avg_rewards = running_avg(rewards, N=100)
        # Offset x-axis so the running average aligns with the 100th episode onward
        plt.plot(np.arange(len(avg_rewards)) + 99, avg_rewards, label="Reward (100‐episode avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Plot epsilon
    if "eps" not in files:
        raise KeyError(f"'eps' not found in {npz_path}. Available keys: {files}")
    eps = data["eps"]

    plt.figure(figsize=(8, 4))
    plt.plot(eps, color="C1")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Schedule")
    plt.tight_layout()
    plt.show()

    # 3) Plot loss if present
    if "loss" in files:
        loss = data["loss"]
        if loss.size > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(loss, color="C2")
            plt.yscale("log")
            plt.xlabel("Learning Step")
            plt.ylabel("TD‐Loss")
            plt.title("Loss Curve")
            plt.tight_layout()
            plt.show()
        else:
            print("Note: 'loss' array is present but empty, skipping loss plot.")
    else:
        print("Note: 'loss' key not found in archive; skipping loss plot.")

def main():
    parser = argparse.ArgumentParser(
        description="Plot training statistics from a .npz file containing 'rewards', 'eps', and optionally 'loss'."
    )
    parser.add_argument(
        "npz_file",
        nargs="?",
        default="training_stats.npz",
        help="Path to the .npz file (default: training_stats.npz)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz_file):
        raise FileNotFoundError(f"Could not find file '{args.npz_file}'")

    plot_training_stats(args.npz_file)

if __name__ == "__main__":
    main()
