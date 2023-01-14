# Exploring the Potential of Quantum Computing for Reinforcement Learning: A Case Study on the Cart Pole Environment!

Deep Reinforcement Learning (RL) has  recently achieved impressive results on  a variety of challenging tasks. However, the computational resources required for Deep RL can be huge, especially when the RL tasks involve high dimensional observation spaces or long  
time  horizons.  To overcome these limitations, a promising approach is to use Quantum computers to accelerate the training of DNNs for  RL. 

In this paper, we re-implement the quantum  circuit  introduced  in [Quantum  agents in the Gym:  a variational quantum algorithm for deep Q-learning](https://arxiv.org/abs/2103.15084) to learn the [Cart Pole Environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). The algorithm combines Variational Quantum Algorithms (VQAs) and  Deep Q-learning (DQL). Our Quantum  agent is implemented in [Qiskit](https://qiskit.org/), an open-source quantum computing framework.  


## UML diagram

![UML](/images/uml.jpeg)

## Quantum Circuit

![QuantumCircuit](/images/CircuitoQML.png)

### Ansatz

![Ansatz](/images/ansatz.png)

## Comparison Result with Classical Approach

![Comparison](/images/comparison.png)

Comparison between Quantum (Left) and Classical (Right) Approach.

From the graphs above, it’s evident that quantum model training is globally more
robust than its classical counterpart. However the classic model is undoubtedly
faster than the quantum approach despite it has 17’000 trainable parameters, unlike
the quantum approach which has only 46 trainable parameters.


### Contributors
- [Marco Mistretta](https://github.com/marcomistretta)
- [Girolamo Macaluso](https://github.com/ganjiro/)

### Supervisor
- [Filippo Caruso](https://www.unifi.it/p-doc2-2019-200010-C-3f2b342f38282c-0.html)

