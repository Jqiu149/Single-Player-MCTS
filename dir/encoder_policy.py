import torch
import math

from torch import nn, Tensor
import numpy as n
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class Policy(nn.Module):
  def __init__(self, num_encoder_layers, vector_dim, encoder_nhead, num_actions):
    super().__init__()
    self.pos_enc= Summer(PositionalEncoding1D(vector_dim))

    encoder_layer =nn.TransformerEncoderLayer(d_model=vector_dim, nhead=encoder_nhead,batch_first=True)
    self.encoder = transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

    self.linear_p = nn.Linear(vector_dim, num_actions)
    self.linear_v = nn.Linear(vector_dim, 1)

    self.extra_colp =  nn.Parameter(torch.randn(vector_dim))
    self.extra_colv =  nn.Parameter(torch.randn(vector_dim))

  def forward(self,x):
    inp = torch.cat([x,
                    self.extra_colp.unsqueeze(0).expand(x.size(0), -1, -1),
                    self.extra_colv.unsqueeze(0).expand(x.size(0), -1, -1)
                    ],
                    dim=1
                    )
    inp = self.pos_enc(inp)
    inp = self.encoder(inp)

    #compute logits and p
    logits = self.linear_p(inp[:, -1, :])
    policy = F.softmax(logits, dim=1)

    #compute v
    v = self.linear_v(inp[:, -2, :]).view(-1)

    return logits,policy, v

  def step(self, obs):
    """
    Returns policy and value estimates for given observations.
    :param obs: Array of shape [N] containing N observations.
    :return: Policy estimate [N, n_actions] and value estimate [N] for
    the given observations.
    """
    obs = torch.from_numpy(obs)
    _, pi, v = self.forward(obs)

    return pi.detach().numpy(), v.detach().numpy()
