import torch
import torch.nn as nn
import numpy as np
from milo.nn import *

class MultiStepDynamicsModel(nn.Module):
    def __init__(self,
                 n_models,
                 state_dim,
                 action_dim,
                 embed_dim,
                 deter_dim,
                 stoc_dim,
                 stoc_discrete_dim,
                 gru_type,
                 hidden_sizes,
                 activation='relu',
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n_models = n_models
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoc_dim = stoc_dim
        self.stoc_discrete_dim = stoc_discrete_dim
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        
        # make Dreamer model
        self.encoder = mlp(self.state_dim, self.hidden_sizes, self.deter_dim)
        self.rssm = RSSMCell(n_models, embed_dim, deter_dim, stoc_dim,
                             stoc_discrete_dim, hidden_sizes[0], action_dim,
                             stoc_discrete_dim != 0, gru_type, activation)
        self.decoder = mlp(self.rssm.dist_dim, self.hidden_sizes, self.state_dim)
        
    def initial_state(self, batch_size: int):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim),
            'stoc': torch.zeros(batch_size, self.stoc_dim * self.stoc_discrete_dim)
        }
        
    def forward(self, state_seq, action_seq, state=None):
        pass