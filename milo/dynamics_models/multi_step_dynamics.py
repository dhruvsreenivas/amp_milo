import torch
import torch.nn as nn
import numpy as np

class MultiStepDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, recurrent_dim, hidden_sizes, activation='relu', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        