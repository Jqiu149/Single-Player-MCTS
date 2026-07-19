import torch
import math

from torch import nn, Tensor
import numpy as n
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pe = torch.zeros(max_len, 1, d_model)
      pe[:, 0, 0::2] = torch.sin(position * div_term)
      pe[:, 0, 1::2] = torch.cos(position * div_term)
      self.register_buffer('pe', pe)


  def forward(self, x: Tensor) -> Tensor:
      """
      Arguments:
          x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
      """
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)



class Policy(nn.Module):
  def __init__(self, num_encoder_layers, vector_dim, encoder_nhead, num_actions):
    super().__init__()
    self.pos_emb= PositionalEncoding(vector_dim)

    encoder_layer =nn.TransformerEncoderLayer(d_model=vector_dim, nhead=encoder_nhead)
    self.encoder = transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)


    self.linear_p = nn.Linear(vector_dim, num_actions)
    self.linear_v = nn.Linear(vector_dim, 1)

    self.extra_colp =  nn.Parameter(torch.randn(vector_dim))
    self.extra_colv =  nn.Parameter(torch.randn(vector_dim))

  def forward(self,x):
    x= x.to(device)
    inp = torch.cat([x,
                    self.extra_colp.unsqueeze(0).expand(x.size(0), -1, -1),
                    self.extra_colv.unsqueeze(0).expand(x.size(0), -1, -1)
                    ],
                    dim=1
                    ).float()
    inp = self.pos_emb(inp)
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

    return pi.detach().cpu().numpy(), v.detach().cpu().numpy()
