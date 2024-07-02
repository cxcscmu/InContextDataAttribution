# Influence Readme

This readme contains a brief overview of the influence module. Note that this code is adapted from [kronfluence](https://github.com/pomonam/kronfluence).

# Inner Product Influence

Running ```compute_influences_ip.py``` will run the inner product influence (see our paper for details). It can be adapted to [TracIn](https://arxiv.org/abs/2002.08484) by including multiple checkpoints and their respective learning rates in arguments.

# Influence Function

Running ```compute_influences_ip.py``` will run influence function with the EK-FAC approximation as noted in [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296).