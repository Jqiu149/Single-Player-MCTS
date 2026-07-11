import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """

    def __init__(self, n_vectors, vector_dim, n_hidden, n_actions):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.dense1 = nn.Linear(n_vectors*vector_dim, n_hidden)
        self.dense_p = nn.Linear(n_hidden, n_actions)
        self.dense_v = nn.Linear(n_hidden, 1)

    def forward(self, obs):
        inp = obs.reshape(obs.shape[0], -1)
        inp = torch.tensor(inp)
        

        h_relu = F.relu(self.dense1(inp))

        logits = self.dense_p(h_relu)

        policy = F.softmax(logits, dim=1)

        value = self.dense_v(h_relu).view(-1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()
