# -*- coding: utf-8 -*-
"""
test_DQN.py — Evaluation script for a trained Mario DQN (raw 240×256×3 RGB input).

This fixes the “negative strides” error by forcing each observation to be
contiguous before converting to a torch tensor.

Usage:
    python test_DQN.py [--episodes N] [--render]

Options:
    --episodes N   Number of test episodes to run (default: 10)
    --render       If provided, render each step to the screen
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mario_model.pth"  # adjust if your file is named differently

# ------------------------------------------------------
# 1) Minimal wrappers so that each obs is the raw 240×256×3 RGB array
# ------------------------------------------------------

class OldToNew(gym.Wrapper):
    """Convert (obs, reward, done, info) → (obs, reward, terminated, truncated, info)."""
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

class NewToOld(gym.Wrapper):
    """Convert (obs, reward, terminated, truncated, info) → (obs, reward, done, info)."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, (terminated or truncated), info

class ResetInfoWrapper(gym.Wrapper):
    """Ensure env.reset() returns (obs, info) even for older‐API envs."""
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        return (res, {}) if not isinstance(res, tuple) else res

# ------------------------------------------------------
# 2) Define MarioCNN exactly as in your checkpoint
# ------------------------------------------------------

class MarioCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Layer names and shapes must match your checkpoint exactly
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

        # Compute flattened size feeding into fc1 by passing a dummy (1,3,240,256)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 240, 256)
            x = self.relu1(self.conv1(dummy))   # → (1,32,59,63)
            x = self.relu2(self.conv2(x))       # → (1,64,28,30)
            x = self.relu3(self.conv3(x))       # → (1,64,26,28)
            flat_size = x.view(1, -1).size(1)    # = 64 * 26 * 28 = 46592

        self.fc1 = nn.Linear(flat_size, 512)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (batch, 3, 240, 256), dtype=torch.float32
        x = x / 255.0
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten to (batch, 46592)
        x = self.relu_fc(self.fc1(x))
        return self.fc2(x)         # → (batch, n_actions)

# ------------------------------------------------------
# 3) Parse command‐line args
# ------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--episodes", "-e", type=int, default=10,
    help="Number of test episodes to run (default: 10)"
)
parser.add_argument(
    "--render", "-r", action="store_true",
    help="Render the environment on screen"
)
args = parser.parse_args()

# ------------------------------------------------------
# 4) Build & wrap the environment so that each obs is raw (240×256×3)
# ------------------------------------------------------

raw_env = gym_super_mario_bros.make("SuperMarioBros-v0")
# Unwrap any TimeLimit
while isinstance(raw_env, gym.wrappers.TimeLimit):
    raw_env = raw_env.env

compat_env = OldToNew(raw_env)                    # old 4‐tuple → new 5‐tuple
joy_env    = JoypadSpace(compat_env, SIMPLE_MOVEMENT)
old_api    = NewToOld(joy_env)                    # new 5‐tuple → old 4‐tuple
env        = ResetInfoWrapper(old_api)            # ensures reset()→(obs,info)

# ------------------------------------------------------
# 5) Instantiate the network, load weights, and run greedy evaluation
# ------------------------------------------------------

n_actions = env.action_space.n  # should be 7 for SIMPLE_MOVEMENT
policy_net = MarioCNN(n_actions).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Cannot find model file at '{MODEL_PATH}'.")

policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy_net.eval()
print(f"✅ Loaded network weights from '{MODEL_PATH}'")

def run_one_episode(env, net, device, render=False):
    """
    Runs one episode with ε=0 (greedy) and returns total reward.
    We force `state` to be contiguous to avoid negative-strides errors.
    """
    state, _ = env.reset()    # state: (240,256,3), dtype=uint8
    episode_reward = 0.0
    done = False

    while not done:
        if render:
            env.render()

        # Make sure state is a contiguous array before converting to tensor:
        state_c = np.ascontiguousarray(state, dtype=np.uint8)

        # Convert to torch tensor: (240,256,3) → (1,3,240,256)
        state_tensor = torch.tensor(
            state_c.transpose(2, 0, 1),  # channels-first
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            q_vals = net(state_tensor)             # shape (1, n_actions)
            action = q_vals.argmax(dim=1).item()   # greedy choice

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    return episode_reward

if __name__ == "__main__":
    total_rewards = []

    for ep in range(1, args.episodes + 1):
        ep_reward = run_one_episode(env, policy_net, DEVICE, render=args.render)
        total_rewards.append(ep_reward)
        print(f"Test Episode {ep:2d} → Total Reward: {ep_reward:.1f}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print("\n" + "=" * 40)
    print(f"Ran {args.episodes} episodes.  "
          f"Average Reward = {avg_reward:.2f}  ± {std_reward:.2f}")
    print("=" * 40)

    env.close()
