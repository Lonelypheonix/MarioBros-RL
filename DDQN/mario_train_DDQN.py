import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os
import pickle, json, numpy as np
from utils import *

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000
#CKPT_SAVE_INTERVAL = 100
#NUM_OF_EPISODES = 500

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
episode_rewards = []
eps_history     = []
loss_history    = []     # if Agent.learn() returns a TD-loss

for ep in range(1, NUM_OF_EPISODES + 1):
    state, _ = env.reset()
    done, total_reward = False, 0

    while not done:
        action = agent.choose_action(state)          # ε-greedy
        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, action, reward, next_state, done)
            loss = agent.learn()                     # returns TD-loss
            if loss is not None:
                loss_history.append(loss)

        state = next_state

    # ── episode finished ────────────────────────────────────────────────────────
    episode_rewards.append(total_reward)
    eps_history.append(agent.epsilon)

    print(f"Ep {ep:>5} | reward {total_reward:>6.1f} | ε {agent.epsilon:.3f} | "
          f"buffer {len(agent.replay_buffer):>5} | learn steps {agent.learn_step_counter}")

    # ── checkpoint & stats dump ─────────────────────────────────────────────────
    if SHOULD_TRAIN and ep % CKPT_SAVE_INTERVAL == 0:
        ckpt_name = f"model_{ep}_iter.pt"
        agent.save_model(os.path.join(model_path, ckpt_name))
        print(f"✓ Saved checkpoint {ckpt_name}")

        # save stats so far
        np.savez(os.path.join(model_path, "training_stats.npz"),
                 rewards=np.array(episode_rewards, dtype=np.float32),
                 eps=np.array(eps_history,      dtype=np.float32),
                 loss=np.array(loss_history,     dtype=np.float32))
        print("✓ Saved training_stats.npz")

env.close()
