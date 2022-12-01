import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from milo.nn import mlp
import wandb

class OneStepDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, learning_rate=3e-4, activation='relu', grad_clip=0.0, train_for_diff=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.net = mlp(state_dim + action_dim, hidden_sizes, state_dim, activation)
        self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        self.train_for_diff = train_for_diff
        self.grad_clip = grad_clip
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)
    
    def train_model(self, dataloader, n_epochs, normalize_inputs, id=0):
        '''Train model for n_epochs epochs on the offline dataset.'''
        device = next(self.parameters()).device
        epoch_losses = []
        
        # project for specific model run
        name=f'{"diff_in_state" if self.train_for_diff else "next_state"}/dyn_model_{id}'
        wandb.init(project='amp dynamics model training', entity='dhruv_sreenivas', name=name)
        
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
                    state_mean, state_std = state.mean(dim=0), state.std(dim=0)
                    state = (state - state_mean) / (state_std + 1e-8)
                    
                    action_mean, action_std = action.mean(dim=0), action.std(dim=0)
                    action = (action - action_mean) / (action_std + 1e-8)
                    
                    target_mean, target_std = target.mean(dim=0), target.std(dim=0)
                    target = (target - target_mean) / (target_std + 1e-8)
                    
                next_pred = self.forward(state, action)
                loss = F.mse_loss(next_pred, target)
                
                self.optim.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                    
                self.optim.step()
                train_losses.append(loss.detach().cpu().item())
            
            avg_loss = sum(train_losses) / len(train_losses)
            epoch_losses.append(avg_loss)
            wandb.log({'avg_model_loss': avg_loss})
            
        wandb.finish()
        return epoch_losses