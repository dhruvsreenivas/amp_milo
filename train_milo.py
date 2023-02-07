import isaacgym
import torch
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import isaacgymenvs
import wandb

from datasets.dataset import AMPExpertDataset, OfflineDataset
from milo.dynamics_models.ensembles import DynamicsEnsemble

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class AMPWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dyn_cfg = cfg.dynamics_training
        
        # set up
        self.setup()
        
    def setup(self):
        # seed + device
        set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        
        # data setup
        data_dir = Path(to_absolute_path('data'))
        
        expert_dataset_dir = data_dir / 'expert' / self.cfg.task / 'expert_dataset.pt'
        expert_data = torch.load(expert_dataset_dir)
        self.expert_dataset = AMPExpertDataset(*expert_data, device=self.cfg.device)
        
        offline_dataset_dir = data_dir / 'offline' / self.cfg.task / f'{self.cfg.level}_dataset_all.pt'
        offline_data = torch.load(offline_dataset_dir)
        self.offline_dataset = OfflineDataset(*offline_data, n=self.cfg.seq_len, device=self.cfg.device)
        
        # set up state and action dims
        self.cfg.state_dim = offline_data[0].size(1)
        self.cfg.action_dim = offline_data[1].size(1)
        
        # dynamics model setup
        self.dynamics_dir = Path(to_absolute_path('pretrained_dynamics_models')) / self.cfg.task / self.cfg.level / self.cfg.dynamics_training.model_type
        self.dynamics_dir.mkdir(parents=True, exist_ok=True)
        self.dynamics_ensemble = DynamicsEnsemble(self.offline_dataset, self.cfg.dynamics_training)
        
        # TODO agent setup
        
    def train_dynamics(self):
        loss_log = self.dynamics_ensemble.train_models()
        # save dynamics ensemble to directory
        dynamics_save_path = Path(self.dynamics_dir) / f'ensemble_{self.dyn_cfg.n_models}_{self.dyn_cfg.train_for_diff}_{self.dyn_cfg.optim}_{self.dyn_cfg.lr}_{self.dyn_cfg.probabilistic}.pt'
        torch.save(self.dynamics_ensemble, dynamics_save_path)
        print('SAVED DYNAMICS ENSEMBLE TO DISK')
        
        return loss_log
    
    def run_mbrl(self):
        '''Run model-based reinforcement learning, with GAIL rewards specified by the trained AMP agent.'''
        pass
    
@hydra.main(config_path='./milo_cfgs', config_name='config')
def main(cfg):
    amp_ws = AMPWorkspace(cfg)
    
    if cfg.train_dynamics:
        _ = amp_ws.train_dynamics()
    
if __name__ == '__main__':
    main()