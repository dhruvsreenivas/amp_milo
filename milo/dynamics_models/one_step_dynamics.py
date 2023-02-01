import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import numpy as np
from milo.nn import mlp, ResidualMLP, DenseMLP
from milo.utils import move_to
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
                 probabilistic=False,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.probabilistic = probabilistic
        if not probabilistic:
            self.net = mlp(state_dim + action_dim, hidden_sizes, state_dim, activation)
        else:
            # mopo-like setup
            self.features = mlp(state_dim + action_dim, hidden_sizes[:-1], hidden_sizes[-1], activation)
            self.mean = nn.Linear(hidden_sizes[-1], state_dim)
            self.std = nn.Linear(hidden_sizes[-1], state_dim)
            
        self.optim_name = optim_name
        self.lr = learning_rate
        if optim_name == 'adam':
            if hasattr(self, 'net'):
                self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.Adam(params, lr=learning_rate)
        elif optim_name == 'adamw':
            if hasattr(self, 'net'):
                self.optim = optim.AdamW(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.AdamW(params, lr=learning_rate)
        else:
            if hasattr(self, 'net'):
                self.optim = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # taken from original MILO repo
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov=True)
        
        self.train_for_diff = train_for_diff
        self.grad_clip = grad_clip
        self.transformations = transformations
        
        self.name = 'reg'
        
    def forward(self, state, action, transform_out=True):
        assert state.dim() == action.dim() == 2, 'not sequence of (s, a)!'
        sa = torch.cat([state, action], dim=-1)
        if not self.probabilistic:
            out = self.net(sa)
            extras = None
        else:
            features = self.features(sa)
            mean = self.mean(features)
            std = self.std(features)
            
            dist = td.Normal(mean, std.exp())
            out = dist.sample() # no need to pass gradients as this is not used in loss (MOPO loss)
            extras = (mean, std)
            
        if transform_out:
            target_mean, target_std = self.transformations[-2:]
            out = (out * target_std) + target_mean
            
        return out, extras
    
    def multistep_loss(self, state, action, next_state, done):
        assert state.dim() == action.dim() == 3, 'sequence of (s, a) required here!'
        assert done.dim() == 2
        
        loss = 0.0
        n = state.size(1)
        s = state[:, 0, :]
        for i in range(n):
            a = action[:, i, :]
            sa = torch.cat([s, a], dim=-1)
            
            pred = self.net(sa)
            if self.train_for_diff:
                diff = pred - next_state[:, i, :] + state[:, i, :]
            else:
                diff = pred - s - next_state[:, i, :] + state[:, i, :]
                
            loss_i = diff.square().mean()
            loss += loss_i
            
            ns = pred + s if self.train_for_diff else pred
            if i < n - 1:
                s = state[:, i + 1, :] + (1.0 - done)[i].unsqueeze(-1) * (ns - state[:, i + 1, :])
                assert s.size() == state[:, i + 1, :].size()
                
        return loss
    
    def train_model(self, train_dataloader, n_epochs, normalize_inputs, id=0, val_dataloader=None, logprob=True):
        '''Train model for n_epochs epochs on the offline dataset.'''
        device = next(self.parameters()).device # should be cuda
        epoch_losses = []
        
        # project for specific model run
        name=f'{self.name}_{"diff_in_state" if self.train_for_diff else "next_state"}_{self.optim_name}_{self.lr}_prob_{self.probabilistic}/dyn_model_{id}'
        wandb.init(project='amp onestep dynamics model training', entity='dhruv_sreenivas', name=name)
        
        for _ in range(n_epochs):
            train_losses = []
            
            self.to(device)
            self.train() # set in training mode
            
            for batch in train_dataloader:
                state, action, _, next_state, done = batch
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                done = done.to(device)
                
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
                    
                next_pred, extras = self.forward(state, action, transform_out=False)
                if state.dim() == 2:
                    if not self.probabilistic:
                        loss = F.mse_loss(next_pred, target)
                    else:
                        mean, std = extras
                        if logprob:
                            dist = td.Normal(mean, std.exp())
                            loss = -dist.log_prob(target).mean()
                        else:
                            mse_loss = F.mse_loss(mean, target)
                            var_loss = std.exp().mean()
                            loss = mse_loss + var_loss
                else:
                    loss = self.multistep_loss(state, action, next_state, done)
                
                self.optim.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                    
                self.optim.step()
                train_losses.append(loss.detach().cpu().item())
            
            avg_loss = sum(train_losses) / len(train_losses)
            epoch_losses.append(avg_loss)
            wandb.log({f'avg_model_loss_{id}': avg_loss})
            
            # validation losses
            if val_dataloader is not None:
                # evaluate on cpu, so move everything there
                self.to('cpu')
                self.eval()
                
                val_loss = 0.0
                count = 0
                for batch in val_dataloader:
                    state, action, _, next_state, _ = batch
                    state = state.to('cpu')
                    action = action.to('cpu')
                    next_state = next_state.to('cpu')
                    
                    if self.train_for_diff:
                        target = next_state - state
                    else:
                        target = next_state
                    
                    if normalize_inputs:
                        assert self.transformations is not None, 'cannot normalize if not given dataset stats.'
                        state_mean, state_std, action_mean, action_std, target_mean, target_std = move_to(self.transformations, 'cpu')
                        
                        state = (state - state_mean) / (state_std + 1e-8)
                        action = (action - action_mean) / (action_std + 1e-8)
                        target = (target - target_mean) / (target_std + 1e-8)
                    
                    next_pred, extras = self.forward(state, action, transform_out=False)
                    if not self.probabilistic:
                        loss = F.mse_loss(next_pred, target)
                    else:
                        # TODO fix: should this just be maximum likelihood?
                        mean, std = extras
                        if logprob:
                            dist = td.Normal(mean, std.exp())
                            loss = -dist.log_prob(target).mean()
                        else:
                            mse_loss = F.mse_loss(mean, target)
                            var_loss = std.exp().mean()
                            loss = mse_loss + var_loss
                        
                    val_loss += loss.detach().item()
                    count += 1 # number of batches
                    
                avg_val_loss = val_loss / count
                wandb.log({f'avg_model_loss_{id}_val': avg_val_loss})
                    
            
        wandb.finish()
        return epoch_losses
    
class ResNetDynamicsModel(OneStepDynamicsModel):
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
                 probabilistic=False,
                 seed=0):
        super().__init__(state_dim, action_dim, hidden_sizes, transformations, learning_rate, activation, optim_name, grad_clip, train_for_diff, probabilistic, seed)
        
        self.net = ResidualMLP(state_dim + action_dim, hidden_sizes, state_dim, activation)
        if optim_name == 'adam':
            if hasattr(self, 'net'):
                self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.Adam(params, lr=learning_rate)
        elif optim_name == 'adamw':
            if hasattr(self, 'net'):
                self.optim = optim.AdamW(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.AdamW(params, lr=learning_rate)
        else:
            if hasattr(self, 'net'):
                self.optim = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # taken from original MILO repo
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov=True)
            
        self.name = 'resnet'
            
class DenseNetDynamicsModel(OneStepDynamicsModel):
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
                 probabilistic=False,
                 seed=0):
        super().__init__(state_dim, action_dim, hidden_sizes, transformations, learning_rate, activation, optim_name, grad_clip, train_for_diff, probabilistic, seed)
        
        self.net = DenseMLP(state_dim + action_dim, hidden_sizes, state_dim, activation)
        if optim_name == 'adam':
            if hasattr(self, 'net'):
                self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.Adam(params, lr=learning_rate)
        elif optim_name == 'adamw':
            if hasattr(self, 'net'):
                self.optim = optim.AdamW(self.net.parameters(), lr=learning_rate)
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.AdamW(params, lr=learning_rate)
        else:
            if hasattr(self, 'net'):
                self.optim = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # taken from original MILO repo
            else:
                params = list(self.features.parameters()) + list(self.mean.parameters()) + list(self.std.parameters())
                self.optim = optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov=True)
            
        self.name = 'densenet'