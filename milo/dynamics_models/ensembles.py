import torch
from milo.dynamics_models.one_step_dynamics import OneStepDynamicsModel
from milo.dynamics_models.multi_step_dynamics import MultiStepDynamicsModel
from datasets.dataset import iterative_dataloader
from pathlib import Path

class DynamicsEnsemble:
    def __init__(self,
                 offline_dataset,
                 cfg):
        
        # set up
        self.cfg = cfg
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
                                     learning_rate=cfg.lr,
                                     activation=cfg.activation,
                                     grad_clip=cfg.grad_clip,
                                     train_for_diff=cfg.train_for_diff,
                                     seed=self.base_seed + k).to(self.device)
                for k in range(self.n_models)
            ]
        else:
            self.models = [
                MultiStepDynamicsModel(cfg.state_dim,
                                       cfg.action_dim,
                                       cfg.hidden_dims,
                                       activation=cfg.activation,
                                       seed=self.base_seed + k).to(self.device)
                for k in range(self.n_models)
            ]
            
        # discrepancy args
        self.discrepancy_cfg = cfg.discrepancy
            
    def train_models(self):
        loader = iterative_dataloader(self.dataset, self.cfg.batch_size)
        trained_models = []
        loss_log = {}
        for idx in range(self.n_models):
            print('=' * 20 + f' Training model {idx} ... ' + '=' * 20)
            model = self.models[idx]
            
            epoch_losses = model.train_model(loader, self.cfg.n_epochs, self.normalize_inputs, id=idx)
            loss_log[f'dynamics_training/model_{idx}_losses'] = epoch_losses
            
            trained_models.append(model)
        
        self.models = trained_models
        print('=' * 20 + ' Saved trained models! ' + '=' * 20)
        return loss_log
    
    @torch.no_grad()
    def compute_onestep_discrepancy(self, states, actions, to_cpu=False):
        '''Computes discrepancy for a batch of (s, a) pairs.'''
        assert type(self.models[0]) is OneStepDynamicsModel, "This requires one-step dynamics model to run."
        
        outs = torch.stack([model(states, actions) for model in self.models], dim=0) # (n_models, batch_size, state_dim)
        if self.discrepancy_cfg.max_diff:
            # get L2 distance between all pairs and take max difference
            diffs = [torch.norm(outs[i] - outs[j], p=2, dim=-1) for i in range(self.n_models) for j in range(i + 1, self.n_models)]
            diffs = torch.stack(diffs, dim=0) # (n_pairs, batch_size)
            diffs = diffs.max(0).values # (batch_size)
        else:
            # look at variance of output across models (dim=0)
            logvars = torch.var(torch.log(outs), dim=0) # (batch_size, state_dim)
            diffs = logvars.mean(dim=-1)
            
        if to_cpu:
            diffs = diffs.to('cpu')
        
        return diffs

    def save(self, save_path):
        with Path(save_path).open('wb') as f:
            torch.save(self.models, f)
            
    def load(self, save_path):
        with Path(save_path).open('rb') as f:
            self.models = torch.load(f)