# ES-HyperNEAT in Jax

This is a jax implementation of the [ES-HyperNEAT algorithm](https://pubmed.ncbi.nlm.nih.gov/22938563/) by Risi and Stanley using Jax. This enables easy parallelization using GPUs.

The code is divided into four components:

1. `neat.py`: Implements the NEAT algorithm by wrapping [tensorneat's version](https://github.com/EMI-Group/tensorneat/tree/main) so that the genotype encoding is decoupled from the developmental step.
2. `dev.py`: Implements the "hypercube" developmental step using the CPPNs produced by NEAT. The patterns produced by the CPPNs are used to initialise the hidden nodes using the substrate partitioner. The class HyperCubeDev can be used inside a policy network class, which initialises it's parameters using it. An example is given called `DevelopedPolicyNetwork`.
3. `cppn.py`: Contains the CPPN implementation, which is graph where nodes have bias and activation function values drawn from a list. Connections are directional and parameterized by the adjacency matrix. Maximum number of nodes nneed to be specified, though not necessarily used.
4. `substrate.py`: Implements the initialisation of the substrate based on the variance of values within multi-resultion patches.

This code uses the same conventions as Evosax, with an ask-tell interface such that instantiation and evaluation are handled by different classes, unlike the tensorneat version where they are all coupled.
