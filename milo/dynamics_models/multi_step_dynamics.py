import torch
import torch.nn as nn
import numpy as np
from milo.nn import *

class MultiStepDynamicsModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 embed_dim,
                 deter_dim,
                 stoc_dim,
                 hidden_sizes,
                 activation='relu',
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoc_dim = stoc_dim
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        
        # make Dreamer model
        self.encoder = mlp(self.state_dim, self.hidden_sizes, self.deter_dim)