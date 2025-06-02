import numpy as np, matplotlib.pyplot as plt
stats = np.load("models/2025-05-28-02_53_33/training_stats.npz")

rewards = stats["rewards"]
eps     = stats["eps"]
loss    = stats["loss"]

def running_avg(x, N=100):
    return np.convolve(x, np.ones(N)/N, mode="valid")

plt.figure()
plt.plot(rewards, alpha=.3, label="reward (raw)")
plt.plot(running_avg(rewards), label="reward (100-episode avg)")
plt.xlabel("Episode"); plt.ylabel("Total reward"); plt.legend()
plt.title("Training performance")
plt.show()

plt.figure()
plt.plot(eps); plt.xlabel("Episode"); plt.ylabel("Epsilon")
plt.title("Exploration schedule")
plt.show()


if loss.size:
    plt.figure()
    plt.plot(loss); plt.yscale("log")
    plt.xlabel("Learn step"); plt.ylabel("TD-loss")
    plt.title("Loss curve")
    plt.show()
