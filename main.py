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
from torch.optim import Adam

import gym
import time


class EncodingLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.inputWeights = torch.nn.Parameter(torch.rand(in_dim))
        self.activation = torch.arctan
        weights = torch.Tensor(in_dim)
        self.inputWeights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.inputWeights, -1, 1)

    def forward(self, X):
        return self.activation(X * self.inputWeights)


class QuantumDQN(torch.nn.Module):
    def __init__(self, n_qbits=4, n_reps=5):
        super().__init__()
        self.n_qbits = n_qbits
        self.encodingLayer = EncodingLayer(n_qbits)

        self.outputWeights = torch.nn.Parameter(torch.Tensor(2))
        torch.nn.init.uniform_(self.outputWeights, 35, 40)  # <-- Initialization strategy (heuristic choice)
        # self.outputWeights = torch.nn.Parameter(torch.rand(2))

        self.observables = [PauliSumOp.from_list([('ZZII', 1.0)]), PauliSumOp.from_list([('IIZZ', 1.0)])]

        inputsParam, weightParam, quantumCircuitRaw = self.buildQuantumCircuit(n_reps)

        qnn = EstimatorQNN(circuit=quantumCircuitRaw, input_params=inputsParam,
                           weight_params=weightParam,
                           observables=self.observables, input_gradients=True)
        initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
        self.quantumCircuit = TorchConnector(qnn, initial_weights)

    def embeddingLayer(self):

        qr = QuantumRegister(self.n_qbits, 'qr')
        qc = QuantumCircuit(qr)

        inputs = qk.circuit.ParameterVector('x', self.n_qbits)

        for i in range(len(inputs)):
            qc.rx(inputs[i], i)

        return inputs, qc

    def buildQuantumCircuit(self, n_reps):

        qr = QuantumRegister(self.n_qbits, 'qr')

        qc = QuantumCircuit(qr)

        inputs, embeddingLayer = self.embeddingLayer()

        for i in range(n_reps):
            qc.compose(embeddingLayer, inplace=True)
            qc.barrier()
            ansatz = TwoLocal(num_qubits=self.n_qbits, rotation_blocks=['ry', 'rz'],
                              entanglement_blocks='cz', entanglement='circular',
                              reps=1, parameter_prefix='theta' + str(i), insert_barriers=True,
                              skip_final_rotation_layer=True)

            qc.compose(ansatz, inplace=True)
            qc.barrier()

        params = [elt for elt in list(qc.parameters) if elt not in inputs]
        # print(qc.decompose().draw(output="text"))
        qc.decompose().draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
        plt.show()
        return list(inputs), params, qc

    def forward(self, X):

        res = self.quantumCircuit.forward(self.encodingLayer(X))
        res = ((res * self.outputWeights) + 1) / 2
        return res


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
    def __init__(self, capacity=2000, env_name="CartPole-v1", discount_rate=0.99, path="model.pt", loadCheckpoint=True):
        self.discount_rate = discount_rate
        self.env_name = env_name
        self.capacity = capacity
        self.replayMemory = ReplayMemory(capacity)
        self.env = gym.make(self.env_name)
        self.oldEpoch = 0
        self.model = QuantumDQN().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.env_state = None
        self.path = path

        if loadCheckpoint:
            self.checkpoint = torch.load(self.path)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.oldEpoch = self.checkpoint['epoch']
        self.initReplayBuffer()

    def train(self, epochs, batch_size):
        eps = 0.99
        start = time.time()
        for epoch in range(self.oldEpoch, epochs):

            states, actions, next_states, rewards, dones = self.replayMemory.getBatchSample(batch_size)

            with torch.no_grad():
                next_Q_values = self.model(next_states).cpu().numpy()
            max_next_Q_values = np.max(next_Q_values, axis=1)
            target_Q_values = (rewards + (1 - dones) * self.discount_rate * max_next_Q_values)

            q_values = self.model(Tensor(states))
            # q_value = torch.select(input=q_values, dim=0, index=actions)
            # loss = torch.sum((target_Q_values - q_value) ** 2)

            loss = torch.tensor(0., device=device)
            for j, state in enumerate(states):
                Q_value = q_values[j][actions[j]]
                loss += (target_Q_values[j] - Q_value) ** 2

            # Evaluate the gradients and update the parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.env_state, _ = self.env.reset()
            for _ in range(200):
                self.epsilonGreedy(eps)

            # eps -= eps / epochs
            eps = max(1 - epoch / (epochs * 0.75), 0.01)
            if epoch % 10 == 0:
                end = time.time()

                print("Epoch: ", epoch + 1, " Loss: ", loss.detach(), " Time: ", end - start, " Eps: ", eps)
                start = time.time()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, self.path)

            if epoch % 20 == 0:
                self.env_state, _ = self.env.reset()
                step = 0
                # 200 is the target score for considering the environment solved
                for step in range(200):

                    with torch.no_grad():
                        Q_values = self.model(Tensor(self.env_state).to(device)).cpu().numpy()
                    action = np.argmax(Q_values)

                    self.env_state, _, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    if done:
                        break

                print("Survived for ", step + 1, " steps")
                self.env_state, _ = self.env.reset()

    def epsilonGreedy(self, epsilon=0.):

        if torch.rand(1) < epsilon:
            action = self.env.action_space.sample()

        else:

            with torch.no_grad():
                Q_values = self.model(Tensor(self.env_state).to(device)).cpu().numpy()
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

    def initReplayBuffer(self):

        for _ in range(20):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: ", device)

Trainer().train(1000, 16)
