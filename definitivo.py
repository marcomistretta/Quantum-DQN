import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import TwoLocal

import qiskit as qk
from qiskit.opflow import PauliSumOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
from torch import Tensor
from torch.optim import Adam, SGD

import gym
import time


class EncodingLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.activation = torch.arctan
        self.inputWeights = torch.nn.Parameter(Tensor(np.ones(shape=in_dim)))

    def forward(self, X):
        a = X.repeat(1, 5)
        return self.activation(a * self.inputWeights)  # todo fix


class OutputLayer(torch.nn.Module):  # todo è hardcodato
    def __init__(self, outdim):
        super().__init__()

        self.outputWeights = torch.nn.Parameter(torch.Tensor([1, 1]))
        torch.nn.init.uniform_(self.outputWeights, 35, 40)

    def forward(self, X):
        return ((X + 1) / 2) * self.outputWeights


def createCircuit(n_qbits=4, n_reps=5):
    observables = [PauliSumOp.from_list([('ZZII', 1.0)]), PauliSumOp.from_list([('IIZZ', 1.0)])]

    inputsParam, weightParam, quantumCircuitRaw = buildQuantumCircuit(n_reps, n_qbits)

    qnn = EstimatorQNN(circuit=quantumCircuitRaw, input_params=inputsParam,
                       weight_params=weightParam,
                       observables=observables, input_gradients=True)
    initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
    return TorchConnector(qnn, initial_weights)


def buildQuantumCircuit(n_reps, n_qbits):
    qr = QuantumRegister(n_qbits, 'qr')

    qc = QuantumCircuit(qr)

    parameters = qk.circuit.ParameterVector('theta', 2 * n_qbits * n_reps)
    inputParameters = qk.circuit.ParameterVector('x', n_qbits * n_reps)
    for i in range(n_reps):

        qc.barrier()

        for j in range(n_qbits):
            qc.rx(inputParameters[j + (i * n_qbits)], j)
            qc.ry(parameters[j + (2 * i * n_qbits)], j)
            qc.rz(parameters[n_qbits + j + (2 * i * n_qbits)], j)

        qc.cz(qr[0], qr[3])  # todo farlo diventare uguale per n qbit
        qc.cz(qr[0], qr[1])
        qc.cz(qr[1], qr[2])
        qc.cz(qr[2], qr[3])

        qc.barrier()

    qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.show()
    return inputParameters, parameters, qc


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def getBatchSample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dones_batch = torch.cat(batch.done)
        next_states = torch.cat(batch.next_state)

        return state_batch, action_batch, next_states, reward_batch, dones_batch

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self, capacity=10000, env_name="CartPole-v1", discount_rate=0.99, path="model_weight_ok.pt",
                 loadCheckpoint=False, saveModel=False):
        self.saveModel = saveModel
        self.discount_rate = discount_rate
        self.env_name = env_name
        self.capacity = capacity
        self.replayMemory = ReplayMemory(capacity)
        self.env = gym.make(self.env_name)
        self.oldEpoch = 0

        self.pQuantumCircuit = createCircuit()
        self.pQuantumCircuit.to(device)

        self.encodingLayer = EncodingLayer(4 * 5)
        self.encodingLayer.to(device)

        self.outputLayer = OutputLayer(2)
        self.outputLayer.to(device)

        self.optimizer_qnn = Adam(self.pQuantumCircuit.parameters(), lr=0.001)
        self.optimizer_enc = Adam(self.encodingLayer.parameters(), lr=0.01)
        self.optimizer_out = Adam(self.outputLayer.parameters(), lr=0.01)

        self.env_state = None
        self.path = path

        if loadCheckpoint:
            self.checkpoint = torch.load(self.path)
            self.pQuantumCircuit.load_state_dict(self.checkpoint['qnn_state_dict'])
            self.encodingLayer.load_state_dict(self.checkpoint['qnc_state_dict'])
            self.outputLayer.load_state_dict(self.checkpoint['out_state_dict'])
            self.oldEpoch = self.checkpoint['epoch']

        print("Initial")
        print(self.pQuantumCircuit.weight)
        print(self.encodingLayer.inputWeights)
        print(self.outputLayer.outputWeights)
        print()
        self.initReplayBuffer()

    def getQvalue(self, state):

        encodedState = self.encodingLayer(state)
        qnnstate = self.pQuantumCircuit.forward(encodedState)
        outState = self.outputLayer(qnnstate)
        return outState

    def train(self, epochs, batch_size):
        eps = 0.99
        start = time.time()
        loss_fn = torch.nn.MSELoss()
        myDumbCountList = []
        myDumbTimeList = []
        myDumbLossList = []

        for epoch in range(self.oldEpoch, epochs):

            self.optimizer_qnn.zero_grad()
            self.optimizer_enc.zero_grad()
            self.optimizer_out.zero_grad()

            states, actions, next_states, rewards, dones = self.replayMemory.getBatchSample(batch_size)

            with torch.no_grad():
                next_Q_values = self.getQvalue(states).cpu().numpy()
            max_next_Q_values = np.max(next_Q_values, axis=1)
            target_Q_values = (rewards + (1 - dones) * self.discount_rate * max_next_Q_values)

            q_values = self.getQvalue(Tensor(states))

            action_masks = []
            one_hot_actions = {0: [1, 0], 1: [0, 1]}
            actions = actions.cpu()
            for action in actions:
                action_masks.append(one_hot_actions[action.item()])

            reduced_q_vals = torch.sum(q_values.cpu() * torch.IntTensor(action_masks), dim=1)
            loss = loss_fn(reduced_q_vals, target_Q_values)
            myDumbLossList.append(loss)
            loss.backward()

            self.optimizer_qnn.step()
            self.optimizer_enc.step()
            self.optimizer_out.step()

            self.env_state, _ = self.env.reset()

            myDumbCount = 0
            for _ in range(200):
                myDumbCount = myDumbCount + 1
                if self.epsilonGreedy(eps):
                    break
            myDumbCountList.append(myDumbCount)
            eps = max(eps * 0.99, 0.01)

            end = time.time()
            myDumbTime = end - start
            myDumbTimeList.append(myDumbTime)

            if epoch > 9:
                print("Epoch: ", epoch + 1, " Loss: ", loss.item(), " AvgLoss: ",
                      (sum(myDumbLossList[-10:]) / 10).item(),
                      " Time: ", myDumbTime, "AvgTime: ", sum(myDumbTimeList[-10:]) / 10,
                      " Eps: ", eps, "Steps: ", myDumbCount,
                      "AvgSteps: ", sum(myDumbCountList[-10:]) / 10)
            else:
                print("Epoch: ", epoch + 1, " Loss: ", loss.item(), " AvgLoss: ",
                      sum(myDumbLossList) / len(myDumbLossList),
                      " Time: ", myDumbTime, "AvgTime: ", sum(myDumbTimeList) / len(myDumbTimeList),
                      " Eps: ", eps, "Steps: ", myDumbCount,
                      "AvgSteps: ", sum(myDumbCountList) / len(myDumbCountList))
            if epoch % 10 == 0:
                print()
                print("Optimized")
                print(self.pQuantumCircuit.weight)
                print(self.encodingLayer.inputWeights)
                print(self.outputLayer.outputWeights)
                print()
                if self.saveModel:
                    torch.save({
                        'epoch': epoch,
                        'qnn_state_dict': self.pQuantumCircuit.state_dict(),
                        'qnc_state_dict': self.encodingLayer.state_dict(),
                        'out_state_dict': self.outputLayer.state_dict(),
                        # 'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                    }, self.path)

            start = time.time()

            # self.env_state, _ = self.env.reset()
            # step = 0
            # for step in range(200):
            #
            #     with torch.no_grad():
            #         Q_values = self.getQvalue(Tensor(self.env_state).to(device)).cpu().numpy()
            #     action = np.argmax(Q_values)
            #
            #     self.env_state, _, terminated, truncated, _ = self.env.step(action)
            #     done = terminated or truncated
            #     if done:
            #         break
            #
            # print("Survived for ", step + 1, " steps")
            # self.env_state, _ = self.env.reset()

    def epsilonGreedy(self, epsilon=0.):

        if torch.rand(1) < epsilon:
            action = self.env.action_space.sample()

        else:

            with torch.no_grad():
                Q_values = self.getQvalue(Tensor(self.env_state).to(device)).cpu().numpy()
            action = np.argmax(Q_values)

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        s = torch.tensor(self.env_state, dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0)
        s_ = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        r = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        d = torch.tensor(done, dtype=torch.int8).unsqueeze(0)

        self.replayMemory.push(s, a, s_, r, d)
        self.env_state = next_state
        if done:
            self.env_state, _ = self.env.reset()
            return True
        return False  # todo farlo piu carino

    def initReplayBuffer(self):

        for _ in range(2):
            self.env_state, _ = self.env.reset()
            for _ in range(200):

                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                s = torch.tensor(self.env_state, dtype=torch.float32, device=device).unsqueeze(0)
                a = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0)
                s_ = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                r = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                d = torch.tensor(done, dtype=torch.int8).unsqueeze(0)

                self.replayMemory.push(s, a, s_, r, d)

                if done:
                    break

                self.env_state = next_state
        self.env_state, _ = self.env.reset()

    def test(self, repetition=10):

        for _ in range(repetition):
            state, _ = self.env.reset()
            for step in range(200):
                Q_values = self.getQvalue(Tensor(state).to(device)).cpu().detach().numpy()
                action = np.argmax(Q_values[0])

                obs, reward, done, info, _ = self.env.step(action)
                if done:
                    break
            print("Survived for ", step + 1, " steps")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: ", device)

trainer = Trainer(path="speriamoBene.pt", loadCheckpoint=False, saveModel=True)
print("Epoch ", trainer.oldEpoch)
trainer.train(3000, 16)
trainer.test(200)
