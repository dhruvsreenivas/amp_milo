import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from milo.nn import mlp, ResidualMLP, DenseMLP
import wandb

# TODO if this still sucks, see if probabilistic modeling with Gaussian mean/std is useful like in MOPO, and maximize log likelihood of next state
class OneStepDynamicsModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 transformations,
                 learning_rate=3e-4,
                 activation='relu',
                 optim_name='adam',
                 grad_clip=0.0,
                 train_for_diff=True,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.net = mlp(state_dim + action_dim, hidden_sizes, state_dim, activation)
        self.optim_name = optim_name
        self.lr = learning_rate
        if optim_name == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # taken from original MILO repo
        
        self.train_for_diff = train_for_diff
        self.grad_clip = grad_clip
        self.transformations = transformations
        
        self.name = 'reg'
        
    def forward(self, state, action, transform_out=True):
        sa = torch.cat([state, action], dim=-1)
        out = self.net(sa)
        if transform_out:
            target_mean, target_std = self.transformations[-2:]
            out = (out * target_std) + target_mean
        return out
    
    def train_model(self, dataloader, n_epochs, normalize_inputs, id=0):
        '''Train model for n_epochs epochs on the offline dataset.'''
        device = next(self.parameters()).device
        epoch_losses = []
        
        # project for specific model run
        name=f'{self.name}_{"diff_in_state" if self.train_for_diff else "next_state"}_{self.optim_name}_{self.lr}/dyn_model_{id}'
        wandb.init(project='amp onestep dynamics model training', entity='dhruv_sreenivas', name=name)
        
        for _ in range(n_epochs):
            train_losses = []
            for batch in dataloader:
                state, action, next_state = batch
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                
                if self.train_for_diff:
                    target = next_state - state
                else:
                    target = next_state
                
                if normalize_inputs:
                    assert self.transformations is not None, 'cannot normalize if not given dataset stats.'
                    state_mean, state_std, action_mean, action_std, target_mean, target_std = self.transformations
                    
                    state = (state - state_mean) / (state_std + 1e-8)
                    action = (action - action_mean) / (action_std + 1e-8)
                    target = (target - target_mean) / (target_std + 1e-8)
                    
                next_pred = self.forward(state, action, transform_out=False)
                loss = F.mse_loss(next_pred, target)
                
                self.optim.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                    
                self.optim.step()
                train_losses.append(loss.detach().cpu().item())
            
            avg_loss = sum(train_losses) / len(train_losses)
            epoch_losses.append(avg_loss)
            wandb.log({f'avg_model_loss_{id}': avg_loss})
            
        wandb.finish()
        return epoch_losses
    
class ResNetDynamicsModel(OneStepDynamicsModel):
    def __init__(self, state_dim, action_dim, hidden_sizes, transformations, learning_rate=3e-4, activation='relu', optim_name='adam', grad_clip=0.0, train_for_diff=True, seed=0):
        super().__init__(state_dim, action_dim, hidden_sizes, transformations, learning_rate, activation, optim_name, grad_clip, train_for_diff, seed)
        
        self.net = ResidualMLP(state_dim + action_dim, hidden_sizes, state_dim, activation)
        if optim_name == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.net.parameters(), lr=learning_rate)
            
        self.name = 'resnet'
            
class DenseNetDynamicsModel(OneStepDynamicsModel):
    def __init__(self, state_dim, action_dim, hidden_sizes, transformations, learning_rate=3e-4, activation='relu', optim_name='adam', grad_clip=0.0, train_for_diff=True, seed=0):
        super().__init__(state_dim, action_dim, hidden_sizes, transformations, learning_rate, activation, optim_name, grad_clip, train_for_diff, seed)
        
        self.net = DenseMLP(state_dim + action_dim, hidden_sizes, state_dim, activation)
        if optim_name == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.net.parameters(), lr=learning_rate)
            
        self.name = 'densenet'