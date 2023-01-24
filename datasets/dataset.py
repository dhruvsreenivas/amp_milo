import torch
from torch.utils.data import Dataset, DataLoader
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
    
class OfflineDataset(Dataset):
    '''Offline, model-based dataset with actions.'''
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        assert states.size(0) == actions.size(0) == next_states.size(0), "don't have the same amount of (s, a, s')"
        self.states = states.detach()
        self.actions = actions.detach()
        self.next_states = next_states.detach()
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx: int):
        state = self.states[idx].to(self.device)
        action = self.actions[idx].to(self.device)
        next_state = self.next_states[idx].to(self.device)
        return state, action, next_state
    
    def sample_seq(self, seq_len: int, device: str = 'cuda'):
        '''Naive sequence sampling--make sure to fix to add done flags into mix.'''
        start_idx = np.random.randint(0, self.states.size(0) - seq_len)
        states = self.states[start_idx : start_idx + seq_len].to(device)
        actions = self.actions[start_idx : start_idx + seq_len].to(device)
        next_states = self.next_states[start_idx : start_idx + seq_len].to(device)
        
        return states, actions, next_states
    
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