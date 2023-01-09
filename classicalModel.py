import math
import random
import time
import warnings
from collections import namedtuple, deque

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")

if gym.__version__[:4] == '0.26':
    env = gym.make('CartPole-v1')
elif gym.__version__[:4] == '0.25':
    env = gym.make('CartPole-v1', new_step_api=True)
else:
    raise ImportError(f"Requires gym v25 or v26, actual version: {gym.__version__}")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 16  # UGUALE
GAMMA = 0.99  # UGUALE
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3  # UGUALE

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, _ = env.reset()

n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(2000)

steps_done = 0


def select_action(state, epoch, epochs):
    global steps_done
    sample = random.random()
    A = 0.5
    B = 0.1
    C = 0.1
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #                 math.exp(-1. * steps_done / EPS_DECAY)
    standardized_time = (epoch - A * epochs) / (B * epochs)
    cosh = np.cosh(math.exp(-standardized_time))
    eps_threshold = 1.1 - (1 / cosh + (epoch * C / epochs))

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), eps_threshold


def select_action_test(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1)[1].view(1, 1)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 1  # TODO remove
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    my_loss = loss.item()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return my_loss


if torch.cuda.is_available():
    num_episodes = 500  # UGUALE
else:
    num_episodes = 50

save_path = "classical/"
myDumbTimeList = []
myDumbLossList = []
epsList = []
stepList = []
realStepList = []
first_time = True
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, _ = env.reset()

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    start = time.time()
    for t in range(200):

        action, my_eps = select_action(state, i_episode, num_episodes)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        my_loss = optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            stepList.append(t + 1)  # 4 fatto
            break
    if i_episode % 50 == 0:
        plot_durations()
    myDumbLossList.append(my_loss)  # 2 fatto
    epsList.append(my_eps)  # 3 fatto

    end = time.time()
    elapse = end - start
    myDumbTimeList.append(elapse)  # 1 fatto

    tmpVect = []
    for k in range(4):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for step2 in range(200):
            with torch.no_grad():
                action = select_action_test(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                state = next_state
            if done:
                break
        tmpVect.append(step2)
    realStepList.append(tmpVect)  # 5 fatto

    if len(memory) >= BATCH_SIZE:
        # if first_time:
        #     print(i_episode)
        #     print(myDumbTimeList)  # 1 fatto
        #     print(myDumbLossList)  # 2 fatto
        #     print(epsList)  # 3 fatto
        #     print(stepList)  # 4 fatto
        #     print(realStepList)  # 5 fatto
        #     first_time = False

        print("Epoch:", i_episode, "  Loss:", round(myDumbLossList[-1], 3), "  AvgLoss:",
              round(sum(myDumbLossList) / len(myDumbLossList), 3),
              "  Time:", round(elapse, 3), "  AvgTime:", round(sum(myDumbTimeList) / len(myDumbTimeList), 3),
              "  Eps:", round(my_eps, 3), "  Steps:", t + 1, "  AvgSteps:",
              round(sum(stepList) / len(stepList), 3),
              "  TestSteps:", realStepList[-1])

np.save(save_path + "times.npy", myDumbTimeList)  # 1 fatto
np.save(save_path + "losses.npy", myDumbLossList)  # 2 fatto
np.save(save_path + "epses.npy", epsList)  # 3 fatto
np.save(save_path + "rewards.npy", stepList)  # 4 fatto
np.save(save_path + "realsteps.npy", realStepList)  # 5 fatto
torch.save(policy_net, save_path+"policy_net.pth")
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
