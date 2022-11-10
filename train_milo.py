import torch
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import isaacgymenvs
import wandb

from milo.dynamics_models.ensembles import DynamicsEnsemble

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class AMPWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # set up
        self.setup()
        
    def setup(self):
        # seed + device
        set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        
        # data setup
        data_dir = Path(to_absolute_path('data'))
        self.expert_dataset = data_dir / 'expert' / f'dataset_{self.cfg.data.expert.n_samples}.npz'
        self.offline_dataset = data_dir / 'offline' / f'dataset_{self.cfg.data.offline.n_samples}.npz'
        
        # dynamics model setup
        self.dynamics_dir = Path(to_absolute_path('pretrained_dynamics_models')) / f'model_{self.cfg.data.offline.n_samples}'
        self.dynamics_dir.mkdir(parents=True, exist_ok=True)
        self.dynamics_ensemble = DynamicsEnsemble(self.offline_dataset, self.cfg.dynamics_training)
        
        # TODO agent setup
        
    def train_dynamics(self):
        loss_log = self.dynamics_ensemble.train_models()
        return loss_log
    
@hydra.main(config_path='./milo_cfgs', config_name='config')
def main(cfg):
    amp_ws = AMPWorkspace(cfg)
    
    if cfg.train_dynamics:
        _ = amp_ws.train_dynamics()
    
if __name__ == '__main__':
    main()