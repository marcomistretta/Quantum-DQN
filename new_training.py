import math
import random
import time
from collections import deque, namedtuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
import torch
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import CircuitQNN
from torch import Tensor
from torch.optim import Adam
import warnings

warnings.filterwarnings("ignore")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class EncodingLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.activation = torch.arctan
        self.inputWeights = torch.nn.Parameter(Tensor(np.ones(shape=in_dim)))
        torch.nn.init.uniform_(self.inputWeights, -1, 1)

    def forward(self, X):
        # a = X.repeat(1, 5)
        return self.activation(X * self.inputWeights)


class OutputLayer(torch.nn.Module):
    def __init__(self, outdim):
        super().__init__()

        self.outputWeights = torch.nn.Parameter(torch.Tensor([1, 1]))
        torch.nn.init.uniform_(self.outputWeights, 35, 40)

    def forward(self, X):
        return ((X + 1) / 2) * self.outputWeights


class CircuitBuilder:
    @staticmethod
    def createCircuit(n_qbits=4, n_reps=5, noisy=True):

        inputsParam, weightParam, quantumCircuitRaw = CircuitBuilder.buildQuantumCircuit(n_reps, n_qbits)

        if noisy:
            provider = qk.IBMQ.load_account()
            backend = provider.get_backend('ibm_oslo')

            backend_sim = AerSimulator.from_backend(backend)
            qi = QuantumInstance(backend_sim)
        else:
            qi = QuantumInstance(qk.Aer.get_backend('statevector_simulator'))

        qnn = CircuitQNN(quantumCircuitRaw, input_params=inputsParam, weight_params=weightParam,
                         quantum_instance=qi)
        initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
        return TorchConnector(qnn, initial_weights)

    @staticmethod
    def buildQuantumCircuit(n_reps, n_qbits):
        qr = QuantumRegister(n_qbits, 'qr')

        qc = QuantumCircuit(qr)

        parameters = qk.circuit.ParameterVector('theta', 2 * n_qbits * n_reps)
        inputParameters = qk.circuit.ParameterVector('x', n_qbits)
        for i in range(n_reps):

            qc.barrier()

            for j in range(n_qbits):
                qc.rx(inputParameters[j], j)
                qc.ry(parameters[j + (2 * i * n_qbits)], j)
                qc.rz(parameters[n_qbits + j + (2 * i * n_qbits)], j)

            qc.cz(qr[0], qr[3])
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


class Model:
    def __init__(self, capacity=2000, env_name="CartPole-v1", discount_rate=0.99, load_path=None, save_path=None,
                 loadCheckpoint=False, saveModel=False):
        self.saveModel = saveModel
        self.discount_rate = discount_rate
        self.env_name = env_name
        self.capacity = capacity
        self.replayMemory = ReplayMemory(capacity)
        self.env = gym.make(self.env_name)
        self.oldEpoch = 0

        self.pQuantumCircuit = CircuitBuilder.createCircuit()
        self.pQuantumCircuit.to(device)

        self.encodingLayer = EncodingLayer(4)
        self.encodingLayer.to(device)

        self.outputLayer = OutputLayer(2)
        self.outputLayer.to(device)

        self.optimizer_qnn = Adam(self.pQuantumCircuit.parameters(), lr=0.001)
        self.optimizer_enc = Adam(self.encodingLayer.parameters(), lr=0.001)
        self.optimizer_out = Adam(self.outputLayer.parameters(), lr=0.001)

        self.env_state = None
        self.load_path = load_path
        self.save_path = save_path

        self.mask_ZZ_12 = torch.tensor([1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1.],
                                       requires_grad=False, device=device)
        self.mask_ZZ_34 = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1.],
                                       requires_grad=False, device=device)

        if loadCheckpoint:
            self.checkpoint = torch.load(self.load_path)
            self.pQuantumCircuit.load_state_dict(self.checkpoint['qnn_state_dict'])
            self.encodingLayer.load_state_dict(self.checkpoint['qnc_state_dict'])
            self.outputLayer.load_state_dict(self.checkpoint['out_state_dict'])
            self.oldEpoch = self.checkpoint['epoch']
            try:
                self.optimizer_qnn.load_state_dict(self.checkpoint['qnn_optimizer_state_dict'])
                self.optimizer_enc.load_state_dict(self.checkpoint['enc_optimizer_state_dict'])
                self.optimizer_out.load_state_dict(self.checkpoint['out_optimizer_state_dict'])
            except:
                print("errore nel caricamente")

        print("Model Initialized")
        print(self.pQuantumCircuit.weight)
        print(self.encodingLayer.inputWeights)
        print(self.outputLayer.outputWeights)
        print()
        self.initReplayBuffer()

    @staticmethod
    def argmax(Q_values):
        if Q_values[0] >= Q_values[1]:
            ret = 0
        else:
            ret = 1
        return ret

    def getQvalue(self, state):

        encodedState = self.encodingLayer(state)
        qnnstate = self.pQuantumCircuit.forward(encodedState)

        expval_ZZ_12 = self.mask_ZZ_12 * qnnstate
        expval_ZZ_34 = self.mask_ZZ_34 * qnnstate

        # Single sample
        if len(qnnstate.shape) == 1:
            expval_ZZ_12 = torch.sum(expval_ZZ_12)
            expval_ZZ_34 = torch.sum(expval_ZZ_34)
            out = torch.cat((expval_ZZ_12.unsqueeze(0), expval_ZZ_34.unsqueeze(0)))

        # Batch of samples
        else:
            expval_ZZ_12 = torch.sum(expval_ZZ_12, dim=1, keepdim=True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34, dim=1, keepdim=True)
            out = torch.cat((expval_ZZ_12, expval_ZZ_34), 1)

        outState = self.outputLayer(out)

        return outState

    def train(self, epochs, batch_size):
        A = 0.5
        B = 0.1
        C = 0.1

        print("Number of preloads training Epoch: ", self.oldEpoch)
        print()
        best_score = -1
        loss_fn = torch.nn.MSELoss()

        myDumbLossList = []
        myDumbTimeList = []
        stepList = []
        realStepList = []
        epsList = []

        for epoch in range(self.oldEpoch + 1, epochs + 1):

            standardized_time = (epoch - A * epochs) / (B * epochs)
            cosh = np.cosh(math.exp(-standardized_time))
            eps = 1.1 - (1 / cosh + (epoch * C / epochs))

            start = time.time()
            # l'ho aggiunto anche se non penso che serva
            self.env.reset()

            # eps = max(eps * 0.99, 0.05)
            for step in range(200):
                # eps = max(epsilon - epoch / (epochs / 4 * 3), 0.01)
                if self.epsilonGreedy(eps):
                    break

            epsList.append(eps)  # epsilon
            stepList.append(step)  # step

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
            myDumbLossList.append(loss.item())  # loss
            loss.backward()

            self.optimizer_qnn.step()
            self.optimizer_enc.step()
            self.optimizer_out.step()

            self.env_state, _ = self.env.reset()

            tmpVect = []
            for k in range(4):
                state, _ = self.env.reset()
                for step2 in range(200):
                    with torch.no_grad():
                        Q_values = self.getQvalue(Tensor(state).to(device)).cpu().detach().numpy()
                        action = self.argmax(Q_values)
                    state, reward, done, info, _ = self.env.step(action)
                    if done:
                        break
                tmpVect.append(step2)
            realStepList.append(tmpVect)

            end = time.time()
            elapse = end - start
            myDumbTimeList.append(elapse)

            if epoch < 10:
                print("Epoch:", epoch, "  Loss:", round(myDumbLossList[-1], 3), "  AvgLoss:",
                      round(sum(myDumbLossList) / len(myDumbLossList), 3),
                      "  Time:", round(elapse, 3), "  AvgTime:", round(sum(myDumbTimeList) / len(myDumbTimeList), 3),
                      "  Eps:", round(eps, 3), "  Steps:", step, "  AvgSteps:",
                      round(sum(stepList) / len(stepList), 3),
                      "  TestSteps:", realStepList[-1])

            else:
                print("Epoch:", epoch, "  Loss:", round(myDumbLossList[-1], 3), "  AvgLoss:",
                      round(sum(myDumbLossList[-10:]) / 10, 3),
                      "  Time:", round(elapse, 3), "  AvgTime:", round(sum(myDumbTimeList[-10:]) / 10, 3),
                      "  Eps:", round(eps, 3), "  Steps:", step, "  AvgSteps:", round(sum(stepList[-10:]) / 10, 3),
                      "  TestSteps:", realStepList[-1])

            if any(i >= best_score for i in realStepList[-1]):
                # print(self.pQuantumCircuit.weight)
                # print(self.encodingLayer.inputWeights)
                # print(self.outputLayer.outputWeights)
                # print()
                if self.saveModel:
                    torch.save({
                        'epoch': epoch,
                        'qnn_state_dict': self.pQuantumCircuit.state_dict(),
                        'qnc_state_dict': self.encodingLayer.state_dict(),
                        'out_state_dict': self.outputLayer.state_dict(),
                        'qnn_optimizer_state_dict': self.optimizer_qnn.state_dict(),
                        'enc_optimizer_state_dict': self.optimizer_enc.state_dict(),
                        'out_optimizer_state_dict': self.optimizer_out.state_dict(),
                        'loss': loss,
                    }, self.save_path + str(max(realStepList[-1])) + 'r-' + str(epoch) + '-model.pth')
                best_score = step
                np.save(self.save_path + "times.npy", myDumbTimeList)
                np.save(self.save_path + "losses.npy", myDumbLossList)
                np.save(self.save_path + "epses.npy", epsList)
                np.save(self.save_path + "rewards.npy", stepList)
                np.save(self.save_path + "realsteps.npy", realStepList)
            elif step >= best_score:
                # print(self.pQuantumCircuit.weight)
                # print(self.encodingLayer.inputWeights)
                # print(self.outputLayer.outputWeights)
                # print()
                if self.saveModel:
                    torch.save({
                        'epoch': epoch,
                        'qnn_state_dict': self.pQuantumCircuit.state_dict(),
                        'qnc_state_dict': self.encodingLayer.state_dict(),
                        'out_state_dict': self.outputLayer.state_dict(),
                        'qnn_optimizer_state_dict': self.optimizer_qnn.state_dict(),
                        'enc_optimizer_state_dict': self.optimizer_enc.state_dict(),
                        'out_optimizer_state_dict': self.optimizer_out.state_dict(),
                        'loss': loss,
                    }, self.save_path + str(step) + 'f-' + str(epoch) + '-model.pth')
                best_score = step
                np.save(self.save_path + "times.npy", myDumbTimeList)
                np.save(self.save_path + "losses.npy", myDumbLossList)
                np.save(self.save_path + "epses.npy", epsList)
                np.save(self.save_path + "rewards.npy", stepList)
                np.save(self.save_path + "realsteps.npy", realStepList)
            if epoch % 50 == 0:
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
                        'qnn_optimizer_state_dict': self.optimizer_qnn.state_dict(),
                        'enc_optimizer_state_dict': self.optimizer_enc.state_dict(),
                        'out_optimizer_state_dict': self.optimizer_out.state_dict(),
                        'loss': loss,
                    }, self.save_path + 'checkpoint-' + str(epoch) + '-model.pth')
                np.save(self.save_path + "times.npy", myDumbTimeList)
                np.save(self.save_path + "losses.npy", myDumbLossList)
                np.save(self.save_path + "epses.npy", epsList)
                np.save(self.save_path + "rewards.npy", stepList)
                np.save(self.save_path + "realsteps.npy", realStepList)

            start = time.time()

        if self.saveModel:
            torch.save({
                'epoch': epoch,
                'qnn_state_dict': self.pQuantumCircuit.state_dict(),
                'qnc_state_dict': self.encodingLayer.state_dict(),
                'out_state_dict': self.outputLayer.state_dict(),
                'qnn_optimizer_state_dict': self.optimizer_qnn.state_dict(),
                'enc_optimizer_state_dict': self.optimizer_enc.state_dict(),
                'out_optimizer_state_dict': self.optimizer_out.state_dict(),
                'loss': loss,
            }, self.save_path + 'checkpoint-' + str(epoch) + '-model.pth')
        np.save(self.save_path + "times.npy", myDumbTimeList)
        np.save(self.save_path + "losses.npy", myDumbLossList)
        np.save(self.save_path + "epses.npy", epsList)
        np.save(self.save_path + "rewards.npy", stepList)
        np.save(self.save_path + "realsteps.npy", realStepList)

    def epsilonGreedy(self, epsilon=0.):

        if torch.rand(1) < epsilon:
            action = self.env.action_space.sample()

        else:

            with torch.no_grad():
                Q_values = self.getQvalue(Tensor(self.env_state).to(device)).cpu().numpy()
            action = self.argmax(Q_values)

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
        return False

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
        stepList = []
        for _ in range(repetition):
            state, _ = self.env.reset()
            for step in range(200):
                Q_values = self.getQvalue(Tensor(state).to(device)).cpu().detach().numpy()
                action = self.argmax(Q_values)

                state, reward, done, info, _ = self.env.step(action)
                if done or info:
                    break
            print("Survived for ", step + 1, " steps")
            stepList.append(step + 1)
        print("Average: ", str(sum(stepList) / len(stepList)), " steps")

    def load(self, load_path=None):
        if load_path is not None:
            print("loading:", load_path)

            self.checkpoint = torch.load(load_path)
            self.pQuantumCircuit.load_state_dict(self.checkpoint['qnn_state_dict'])
            self.encodingLayer.load_state_dict(self.checkpoint['qnc_state_dict'])
            self.outputLayer.load_state_dict(self.checkpoint['out_state_dict'])
            self.oldEpoch = self.checkpoint['epoch']
            try:
                self.optimizer_qnn.load_state_dict(self.checkpoint['qnn_optimizer_state_dict'])
                self.optimizer_enc.load_state_dict(self.checkpoint['enc_optimizer_state_dict'])
                self.optimizer_out.load_state_dict(self.checkpoint['out_optimizer_state_dict'])
            except:
                print("errore nel caricamente")
        elif self.load_path is not None:
            print("loading:", load_path)

            self.checkpoint = torch.load(self.load_path)
            self.pQuantumCircuit.load_state_dict(self.checkpoint['qnn_state_dict'])
            self.encodingLayer.load_state_dict(self.checkpoint['qnc_state_dict'])
            self.outputLayer.load_state_dict(self.checkpoint['out_state_dict'])
            self.oldEpoch = self.checkpoint['epoch']
            try:
                self.optimizer_qnn.load_state_dict(self.checkpoint['qnn_optimizer_state_dict'])
                self.optimizer_enc.load_state_dict(self.checkpoint['enc_optimizer_state_dict'])
                self.optimizer_out.load_state_dict(self.checkpoint['out_optimizer_state_dict'])
            except:
                print("errore nel caricamente")
        else:
            print("errore nel caricamente")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: ", device)

NUM_EPOCHS = 3
BATCH_SIZE = 16
load_path = "models_and_checkpoint/checkpoint-350-model.pth"

model = Model(save_path="test_se_funziona/", loadCheckpoint=False, saveModel=True)
model.train(NUM_EPOCHS, BATCH_SIZE)
model.load(load_path)
model.test(repetition=3)
