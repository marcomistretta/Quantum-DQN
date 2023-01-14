import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print_path = "quantum"  # "quantum" / "noisy-simulator" / "classical" /
width = 8
height = 4

epses = np.load(print_path+"/epses.npy")
losses = np.load(print_path+"/losses.npy")
realsteps = np.load(print_path+"/realsteps.npy")
realsteps = [max(i) for i in realsteps]
rewards = np.load(print_path+"/rewards.npy")
times = np.load(print_path+"/times.npy")

plt.figure(figsize=(width, height))
plt.plot(epses)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Epsilon", fontsize=14)
plt.show()

plt.figure(figsize=(width, height))
plt.plot(losses)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.show()

plt.figure(figsize=(width, height))
plt.plot(realsteps)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Test Step", fontsize=14)
plt.show()

plt.figure(figsize=(width, height))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(times)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.show()
