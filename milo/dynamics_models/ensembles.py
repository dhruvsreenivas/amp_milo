import torch
from torch.utils.data import random_split
from milo.dynamics_models.one_step_dynamics import OneStepDynamicsModel, ResNetDynamicsModel, DenseNetDynamicsModel
from milo.dynamics_models.multi_step_dynamics import MultiStepDynamicsModel
from datasets.dataset import iterative_dataloader, get_dataset_transformations
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
        
        self.transformations = get_dataset_transformations(self.dataset, diff=cfg.train_for_diff)
        
        if cfg.single_step:
            if cfg.model_type == 'reg':
                model_class = OneStepDynamicsModel
            elif cfg.model_type ==  'resnet':
                model_class = ResNetDynamicsModel
            else:
                model_class = DenseNetDynamicsModel
            
            self.models = [
                model_class(cfg.state_dim,
                            cfg.action_dim,
                            cfg.hidden_dims,
                            transformations=self.transformations,
                            learning_rate=cfg.lr,
                            activation=cfg.activation,
                            optim_name=cfg.optim,
                            grad_clip=cfg.grad_clip,
                            train_for_diff=cfg.train_for_diff,
                            probabilistic=cfg.probabilistic,
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
        self.validation = cfg.validation
        
    def train_models(self):
        # add train + validation
        if self.validation:
            num_train = int(0.9 * len(self.dataset))
            train_dataset, val_dataset = random_split(self.dataset, [num_train, len(self.dataset) - num_train])
            train_loader = iterative_dataloader(train_dataset, self.cfg.batch_size)
            val_loader = iterative_dataloader(val_dataset, self.cfg.batch_size)
        else:
            train_loader = iterative_dataloader(self.dataset, self.cfg.batch_size)
            val_loader = None
        
        trained_models = []
        loss_log = {}
        for idx in range(self.n_models):
            print('=' * 20 + f' Training model {idx} ... ' + '=' * 20)
            model = self.models[idx]
            
            epoch_losses = model.train_model(train_loader, self.cfg.n_epochs, self.normalize_inputs, id=idx, val_dataloader=val_loader, logprob=True)
            loss_log[f'dynamics_training/model_{idx}_losses'] = epoch_losses
            
            trained_models.append(model)
        
        self.models = trained_models
        print('=' * 20 + ' Saved trained models! ' + '=' * 20)
        return loss_log
    
    @torch.no_grad()
    def compute_onestep_discrepancy(self, states, actions, to_cpu=False):
        '''Computes discrepancy for a batch of (s, a) pairs.'''
        assert type(self.models[0]) in [OneStepDynamicsModel, ResNetDynamicsModel, DenseNetDynamicsModel], "This requires one-step dynamics model to run."
        
        outs = torch.stack([model(states, actions) for model in self.models], dim=0) # (n_models, batch_size, state_dim)
        print(f'out size (n_models, batch_size, state_dim): {outs.size()}')
        if self.discrepancy_cfg.max_diff:
            # get L2 distance between all pairs and take max difference
            diffs = [torch.norm(outs[i] - outs[j], p=2, dim=-1) for i in range(self.n_models) for j in range(i + 1, self.n_models)]
            diffs = torch.stack(diffs, dim=0) # (n_pairs, batch_size)
            print(f'diffs size (n_pairs, batch_size): {diffs.size()}')
            diffs = diffs.max(0).values # (batch_size)
        else:
            # look at standard deviation of output across models (dim=0)
            stds = torch.std(outs, dim=0) # (batch_size, state_dim)
            print(f'std diffs size (batch_size, state_dim): {stds.size()}')
            diffs = stds.mean(dim=-1)
            
        if to_cpu:
            diffs = diffs.to('cpu')
        
        return diffs

    def save(self, save_path):
        with Path(save_path).open('wb') as f:
            torch.save(self.models, f)
            
    def load(self, save_path):
        with Path(save_path).open('rb') as f:
            self.models = torch.load(f)