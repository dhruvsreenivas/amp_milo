import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
from typing import Tuple

class AMPExpertDataset(Dataset):
    '''AMP expert dataset.'''
    def __init__(self, expert_states: torch.Tensor, expert_next_states: torch.Tensor, device: str='cuda'):
        super().__init__()
        self.device = device
        
        assert expert_states.size(0) == expert_next_states.size(0), "don't have the same amount of (s, s')"
        self.states = expert_states
        self.next_states = expert_next_states
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx: int):
        state = self.states[idx].to(self.device)
        next_state = self.next_states[idx].to(self.device)
        return state, next_state
    
    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.states.size(0), size=batch_size)
        states = self.states[idxs]
        next_states = self.next_states[idxs]
        return states, next_states
    
class OldOfflineDataset(Dataset):
    '''old offline dataset with (s, a) pairs so as not to break anything in the training script yet.'''
    def __init__(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        assert states.size(0) == actions.size(0) == next_states.size(0)
        self.states = states.detach()
        self.actions = actions.detach()
        self.next_states = next_states.detach()
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx):
        s = self.states[idx].to(self.device)
        a = self.actions[idx].to(self.device)
        ns = self.next_states[idx].to(self.device)
        
        return s, a, ns
    
class OfflineDataset(Dataset):
    '''Offline, model-based dataset with actions.'''
    def __init__(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 rewards: torch.Tensor, # currently GAIL-based rewards, don't use in real algo
                 next_states: torch.Tensor,
                 dones: torch.Tensor,
                 n: int = 1,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        assert states.size(0) == actions.size(0) == rewards.size(0) == next_states.size(0) == dones.size(0), "don't have the same amount of (s, a, r, s', d)"
        self.states = states.detach()
        self.actions = actions.detach()
        self.rewards = rewards.detach()
        self.next_states = next_states.detach()
        self.dones = dones.detach()
        
        self.n = n # number of consecutive datapoints to sample
        
    def __len__(self):
        return self.states.size(0)
    
    def correct_seq(self, idx):
        # in the case of i <= n, we must return [:n], otherwise we return [i-n+1:i+1], so as to keep index i
        if idx <= self.n - 1:
            idx = self.n - 1
        
        min_idx = idx - (self.n - 1)
        max_idx = idx + 1
        return min_idx, max_idx
    
    def __getitem__(self, idx: int):
        min_idx, max_idx = self.correct_seq(idx)
        state = self.states[min_idx:max_idx].squeeze().to(self.device)
        action = self.actions[min_idx:max_idx].squeeze().to(self.device)
        reward = self.rewards[min_idx:max_idx].squeeze().to(self.device)
        next_state = self.next_states[min_idx:max_idx].squeeze().to(self.device)
        done = self.dones[min_idx:max_idx].squeeze().to(self.device)
        
        return state, action, reward, next_state, done
    
class SamplingOfflineDataset(IterableDataset):
    def __init__(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 rewards: torch.Tensor, # currently GAIL-based rewards, don't use in real algo
                 next_states: torch.Tensor,
                 dones: torch.Tensor,
                 n: int = 1,
                 max_steps: int = 600000,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        assert states.size(0) == actions.size(0) == rewards.size(0) == next_states.size(0) == dones.size(0), "don't have the same amount of (s, a, r, s', d)"
        self.states = states.detach()
        self.actions = actions.detach()
        self.rewards = rewards.detach()
        self.next_states = next_states.detach()
        self.dones = dones.detach()
        
        self.size = states.size(0)
        self.n = n # number of consecutive datapoints to sample
        self.max_steps = max_steps # hardcode for now
        
    def sample_seq(self):
        valid_idx = False
        while not valid_idx:
            start_idx = np.random.randint(0, self.size - self.n)
            states = self.states[start_idx : start_idx + self.n].squeeze().to(self.device)
            actions = self.actions[start_idx : start_idx + self.n].squeeze().to(self.device)
            rewards = self.rewards[start_idx : start_idx + self.n].squeeze().to(self.device)
            next_states = self.next_states[start_idx : start_idx + self.n].squeeze().to(self.device)
            dones = self.dones[start_idx : start_idx + self.n].squeeze().to(self.device)
            
            valid_idx = (dones[:-1] == 0).all().item() # have to all be 0 before the ending one
            
        return states, actions, rewards, next_states, dones
    
    def __iter__(self):
        for _ in range(self.max_steps):
            yield self.sample_seq()
    
def iterative_dataloader(dataset, batch_size: int, shuffle: bool=True, drop_remainder: bool=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_remainder)
    
def get_dataset_transformations(dataset: OfflineDataset, diff: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = dataset.states
    actions = dataset.actions
    next_states = dataset.next_states
    
    sm, sstd = states.mean(dim=0).float().requires_grad_(False), states.std(dim=0).float().requires_grad_(False)
    am, astd = actions.mean(dim=0).float().requires_grad_(False), actions.std(dim=0).float().requires_grad_(False)
    
    targets = next_states - states if diff else next_states
    tm, tstd = targets.mean(dim=0).float().requires_grad_(False), targets.std(dim=0).float().requires_grad_(False)
    
    return sm, sstd, am, astd, tm, tstd