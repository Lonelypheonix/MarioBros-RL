"""
play_mario.py  – run a trained DDQN in Super Mario Bros
"""
import torch, os, gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent          # same class you trained with

CKPT_PATH = "models/2025-05-28-02_53_33/model_35000_iter.pt"   # <-- adjust
ENV_NAME  = "SuperMarioBros-1-1-v0"
N_EPISODES = 10                # how many test runs you want

# ── Set up environment ───────────────────────────────────────────────────────────
env = gym_super_mario_bros.make(
    ENV_NAME,
    render_mode="human",
    apply_api_compatibility=True
)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# ── Initialize agent & load weights ────────────────────────────────────────────
agent = Agent(
    input_dims=env.observation_space.shape,
    num_actions=env.action_space.n
)
agent.load_model(CKPT_PATH)   # ensure this uses weights_only=True if you applied that change
agent.epsilon   = 0.0         # fully greedy at test time
agent.eps_min    = 0.0
agent.eps_decay  = 0.0

# ── Run evaluation ─────────────────────────────────────────────────────────────
for ep in range(1, N_EPISODES + 1):
    state, _   = env.reset()
    done       = False
    total_r    = 0

    while not done:
        action     = agent.choose_action(state)   # <- no unpacking
        state, rwd, done, trunc, info = env.step(action)
        total_r   += rwd

    print(f"Episode {ep:02d} — Reward: {total_r:.1f}")

env.close()