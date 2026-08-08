import torch
import math

from torch import nn, Tensor
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


#leaving this as a comment b/c idk what i'm supposed to be doing for now...
#!pip install positional-encodings[pytorch]
#https://github.com/tatp22/multidim-positional-encoding



# okay so input dim is the dimension of the vectors being given
# transformer_dim is the dimension we want to map that input to using a linera layer, before feeding it into the transformer and the rest of the model ig
class Policy(nn.Module):
  def __init__(self, num_encoder_layers, input_dim, emb_dim, transformer_feedforward_dim, encoder_nhead, num_actions):
    super().__init__()

    #using to take the input and map to larger dimension mainly. thought we might as well do a linear
    self.input_linear= nn.Linear(input_dim,emb_dim, bias=False)
    
    self.pos_emb= Summer(PositionalEncoding1D(emb_dim))

    encoder_layer =nn.TransformerEncoderLayer(d_model=emb_dim, nhead=encoder_nhead,dim_feedforward = transformer_feedforward_dim, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

    #extra columns we pass together with input into transformer
    self.extra_colp =  nn.Parameter(torch.randn(emb_dim))
    self.extra_colv =  nn.Parameter(torch.randn(emb_dim))

    #layers to get final output in desired shape
    self.linear_p = nn.Linear(emb_dim, num_actions)
    self.linear_v = nn.Linear(emb_dim, 1)


  def forward(self,x):

    #print("weights and biases?", self.input_linear.weight, self.input_linear.bias)
    #print(x)



    magnitudes = torch.linalg.vector_norm(x,dim = -1)
    magnitudes[magnitudes ==0] = torch.inf
    min_magnitudes = magnitudes.min(dim=-1).values
    x= x/ min_magnitudes.reshape(x.size(0),1,1)


    inp = self.input_linear(x)

    #print("after input_linear", inp)

     
    inp = torch.cat([inp,
                    self.extra_colp.unsqueeze(0).expand(x.size(0), -1, -1),
                    self.extra_colv.unsqueeze(0).expand(x.size(0), -1, -1)
                    ],
                    dim=1
                    )

    #mask = (inp==0).all(axis=-1)

    #print("after cat", inp)

    inp = self.pos_emb(inp)

    #print("after pos_emb", inp)
   
    #, src_key_padding_mask=mask
    inp = self.encoder(inp)

    #print("after encoder", inp)

    #compute logits and p
    logits = self.linear_p(inp[:, -2, :])
    policy = F.softmax(logits, dim=1)

    #compute v
    v = self.linear_v(inp[:, -1, :]).view(-1)

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
