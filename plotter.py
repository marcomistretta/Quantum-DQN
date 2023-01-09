import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# XXX QUANTUM 500
# epses = np.load("models_and_checkpoint/epses.npy")
# losses = np.load("models_and_checkpoint/losses.npy")
# realsteps = np.load("models_and_checkpoint/realsteps.npy")
# realsteps = [max(i) for i in realsteps]
# print(np.mean(realsteps[400:]))
# rewards = np.load("models_and_checkpoint/rewards.npy")
# times = np.load("models_and_checkpoint/times.npy")

# XXX QUANTUM 350
# epses = np.load("models_and_checkpoint/epses.npy")[:350]
# losses = np.load("models_and_checkpoint/losses.npy")[:350]
# realsteps = np.load("models_and_checkpoint/realsteps.npy")[:350]
# realsteps = [max(i) for i in realsteps]
# print(np.mean(realsteps[250:])) # tra 250 e 350
# rewards = np.load("models_and_checkpoint/rewards.npy")[:350]
# times = np.load("models_and_checkpoint/times.npy")[:350]

# XXX NOISY QUANTUM 500
# TODO set model path
# epses = np.load("models_and_checkpoint/epses.npy")
# losses = np.load("models_and_checkpoint/losses.npy")
# realsteps = np.load("models_and_checkpoint/realsteps.npy")
# realsteps = [max(i) for i in realsteps]
# print(np.mean(realsteps[400:]))
# rewards = np.load("models_and_checkpoint/rewards.npy")
# times = np.load("models_and_checkpoint/times.npy")

# XXX CLASSICAL 500
# TODO set model path
epses = np.load("classical/epses.npy")
losses = np.load("classical/losses.npy")
realsteps = np.load("classical/realsteps.npy")
realsteps = [max(i) for i in realsteps]
best_subset = realsteps[:200]
print(np.mean(best_subset[100:]))  # tra 100 e 200
rewards = np.load("classical/rewards.npy")
times = np.load("classical/times.npy")

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
