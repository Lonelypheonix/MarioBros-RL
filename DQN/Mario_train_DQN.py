"""
Super Mario Bros Reinforcement Learning Agent
This script implements a Deep Q-Learning agent to play Super Mario Bros.
The agent uses a CNN to process game states and make decisions.
"""

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class MarioEnvWrapper(gym.Wrapper):
    """
    Wrapper class for the Mario environment to handle the new gym API format.
    Converts the old gym API format to the new one with proper return values.
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

def get_model_path(model_dir="."):
    """Returns the path where the model will be saved/loaded."""
    return os.path.join(model_dir, "mario_model.pth")

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

class MarioCNN(nn.Module):
    """
    Convolutional Neural Network for processing game states and predicting actions.
    Architecture:
    - 3 convolutional layers with ReLU activation
    - 2 fully connected layers
    - Output layer for action selection
    """
    def __init__(self, num_actions):
        super(MarioCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc2 = nn.Linear(512, num_actions)

    def calculate_linear_input(self, device):
        """
        Dynamically calculate the input size for the first fully connected layer
        based on the output of the convolutional layers.
        """
        dummy_input = torch.zeros(1, 3, 240, 256).to(device)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        self.fc1 = nn.Linear(self._to_linear, 512).to(device)
        self.fc1 = self.fc1.to(device)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
while isinstance(env, gym.wrappers.TimeLimit):
    env = env.env
while isinstance(env, gym.Wrapper) and not isinstance(env, gym_super_mario_bros.SuperMarioBrosEnv):
    env = env.env
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MarioEnvWrapper(env)

observation, info = env.reset()
model_path = get_model_path()

# Load or create the model
if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = MarioCNN(env.action_space.n)
    model = model.to(device)
    model.calculate_linear_input(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print("No existing model found. Creating a new model.")
    model = MarioCNN(env.action_space.n)
    model = model.to(device)
    model.calculate_linear_input(device)

print(f"Conv1 weight device: {model.conv1.weight.device}")

# Training hyperparameters
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_episodes = 35000
batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.005

# ── NEW: stats/history & checkpoint settings ────────────────────────────
episode_lengths = []
episode_rewards = []
episode_max_x_positions = []
episode_freeze_counts = []
eps_history = []
loss_history = []             # if you want to track per-batch losses
SHOULD_TRAIN = True
CKPT_SAVE_INTERVAL = 5000       # save every 10 episodes
MODEL_DIR = "."               # where to dump checkpoints & stats

# Training statistics tracking
episode_lengths = []
episode_rewards = []
episode_max_x_positions = []
episode_freeze_counts = []

freeze_time_limit = 60
POWER_JUMP_ACTION = 1

# Main training loop
for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_next_observations = []
    total_reward = 0
    previous_vel_y = 0
    jump_button_was_held = False
    OBSTACLE_X_POSITION = 100
    POWER_JUMP_THRESHOLD = 5
    max_x_pos = 0
    freeze_count = 0
    frozen_frames = 0
    previous_x_pos = None

    while not done:
        env.render()

        # Process observation and select action
        observation_tensor = torch.tensor(observation.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action_probs = model(observation_tensor)
                action = torch.argmax(action_probs).item()

        # Execute action and get next state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Custom reward calculation
        reward = 0
        current_x_pos = info['x_pos']
        current_y_pos = info['y_pos']
        current_vel_y = info.get('y_vel', 0)

        # Reward components
        if previous_x_pos is not None and current_x_pos > previous_x_pos:
            reward += 0.1  # Reward for moving right

        if action == env.action_space.sample():
            jump_button_was_held = True
            reward += 0.05  # Reward for holding jump

        if not jump_button_was_held and current_vel_y > previous_vel_y:
            reward += 0.2  # Reward for successful jump

        if current_x_pos > OBSTACLE_X_POSITION:
            reward += 1.0  # Reward for clearing obstacle

        if current_vel_y > POWER_JUMP_THRESHOLD:
            reward += 1.5  # Reward for power jump

        if current_x_pos <= OBSTACLE_X_POSITION and current_y_pos < 0:
            reward -= 0.5  # Penalty for failing to clear obstacle

        if action == POWER_JUMP_ACTION:
            reward += 0.1  # Reward for attempting power jump

        previous_vel_y = current_vel_y
        total_reward += reward
        max_x_pos = max(max_x_pos, info['x_pos'])

        # Anti-freezing mechanism
        current_x_pos = info['x_pos']
        if previous_x_pos is not None and current_x_pos == previous_x_pos:
            frozen_frames += 1
            if frozen_frames >= freeze_time_limit:
                print("Mario froze! Restarting episode.")
                done = True
                freeze_count += 1
        else:
            frozen_frames = 0
        previous_x_pos = current_x_pos

        # Store experience in batch
        batch_observations.append(observation)
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_next_observations.append(next_state)

        observation = next_state

        # Train the model when batch is full or episode is done
        if len(batch_observations) >= batch_size or done:
            if len(batch_observations) > 0:
                # Prepare batch tensors
                batch_observations_tensor = torch.tensor(np.array(batch_observations), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long).to(device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_observations_tensor = torch.tensor(np.array(batch_next_observations), dtype=torch.float32).permute(0, 3, 1, 2).to(device)

                # Perform gradient descent
                optimizer.zero_grad()
                q_values = model(batch_observations_tensor)
                next_q_values = model(batch_next_observations_tensor).max(1)[0].detach()
                target_q_values = batch_rewards_tensor + gamma * next_q_values * (1 - torch.tensor(done, dtype=torch.float32).to(device))
                q_value = q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
                loss = nn.MSELoss()(q_value, target_q_values)
                loss.backward()
                optimizer.step()

            # Clear batch
            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_next_observations = []

    # Record episode statistics
    episode_lengths.append(len(batch_rewards) + 1)
    episode_rewards.append(total_reward)
    episode_max_x_positions.append(max_x_pos)
    episode_freeze_counts.append(freeze_count)

    # ── episode finished ───────────────────────────────────────────────
    eps_history.append(epsilon)

    print(
        f"Ep {episode+1:>3} | "
        f"reward {total_reward:>6.1f} | "
        f"ε {epsilon:.3f} | "
        f"max_x {max_x_pos:>5.1f} | "
        f"freezes {freeze_count:>2}"
    )

    # ── checkpoint & stats dump ────────────────────────────────────────
    if SHOULD_TRAIN and (episode + 1) % CKPT_SAVE_INTERVAL == 0:
        ckpt_name = f"model_ep{episode+1}.pth"
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, ckpt_name))
        print(f"✓ Saved checkpoint {ckpt_name}")

        np.savez(
            os.path.join(MODEL_DIR, "training_stats.npz"),
            rewards=np.array(episode_rewards, dtype=np.float32),
            eps=np.array(eps_history,    dtype=np.float32),
            max_x=np.array(episode_max_x_positions, dtype=np.float32),
            freezes=np.array(episode_freeze_counts, dtype=np.int32),
        )
        print("✓ Saved training_stats.npz")
    # ────────────────────────────────────────────────────────────────────

    # decay exploration
    epsilon = max(epsilon - epsilon_decay, 0.01)

# After all episodes:
print("\n--- Training Statistics ---")
print(f"Average reward: {np.mean(episode_rewards):.2f}")
print(f"Average episode length: {np.mean(episode_lengths):.2f}")
print(f"Average max x position: {np.mean(episode_max_x_positions):.2f}")
print(f"Average freeze count: {np.mean(episode_freeze_counts):.2f}")

# final model save
final_model = "mario_model.pth"
torch.save(model.state_dict(), final_model)
print(f"Model saved as: {final_model}")

env.close()