import torch
from milo.dynamics_models.one_step_dynamics import OneStepDynamicsModel
from milo.dynamics_models.multi_step_dynamics import MultiStepDynamicsModel

class DynamicsEnsemble:
    def __init__(self,
                 offline_dataset,
                 cfg):
        
        # set up
        self.dataset = offline_dataset
        self.n_models = cfg.n_models
        self.normalize_inputs = cfg.normalize_inputs
        self.base_seed = cfg.seed
        self.device = torch.device(cfg.device)
        
        if cfg.single_step:
            self.models = [
                OneStepDynamicsModel(cfg.state_dim,
                                     cfg.action_dim,
                                     cfg.hidden_dims,
                                     activation=cfg.activation,
                                     seed=self.base_seed + k)
                for k in range(self.n_models)
            ]
        else:
            self.models = [
                MultiStepDynamicsModel(cfg.state_dim,
                                       cfg.action_dim,
                                       cfg.hidden_dims,
                                       activation=cfg.activation,
                                       seed=self.base_seed + k)
                for k in range(self.n_models)
            ]