import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

epses = np.load("models_and_checkpoint/epses.npy")
losses = np.load("models_and_checkpoint/losses.npy")
realsteps = np.load("models_and_checkpoint/realsteps.npy")
realsteps = [max(i) for i in realsteps]
print(np.mean(realsteps[400:]))
rewards = np.load("models_and_checkpoint/rewards.npy")
times = np.load("models_and_checkpoint/times.npy")

# epses = np.load("models_and_checkpoint/epses.npy")[:350]
# losses = np.load("models_and_checkpoint/losses.npy")[:350]
# realsteps = np.load("models_and_checkpoint/realsteps.npy")[:350]
# realsteps = [max(i) for i in realsteps]
# print(np.mean(realsteps[250:]))
# rewards = np.load("models_and_checkpoint/rewards.npy")[:350]
# times = np.load("models_and_checkpoint/times.npy")[:350]



plt.figure(figsize=(12, 4))
plt.plot(epses)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Epsilon", fontsize=14)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(realsteps)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Test Step", fontsize=14)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(times)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.show()
