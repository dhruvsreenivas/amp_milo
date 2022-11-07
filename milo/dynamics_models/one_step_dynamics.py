import torch
import torch.nn as nn
import numpy as np
from utils import mlp

class OneStepDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation='relu', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.net = mlp(state_dim + action_dim, hidden_sizes, state_dim, activation)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)